# py/tello_watch/run.py
import argparse
import os
import time

import cv2

from .control import ControlConfig, horizontal_offset_norm
from .flight import Flight
from .safety import SafetyAction, assess
from .statemachine import StateConfig, StateMachine
from .target import select_target
from .telemetry import FlightLogger
from .vision import PersonDetector, frame_is_live

MODELS = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT = os.path.join(MODELS, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL = os.path.join(MODELS, "MobileNetSSD_deploy.caffemodel")

NUDGE = 30  # cm, manual override step
MANUAL_KEYS = {ord(c) for c in "wasdqerf"}

# Control/scan defaults — single source so meta.json matches what flew.
KP, DEADZONE, MAX_YAW = 0.3, 0.1, 25
SCAN_YAW, REACQUIRE_AFTER_S = 20, 2.5
CONF_THRESHOLD = 0.5


def _drone_info(flight):
    """Query best-effort SDK version and serial number from the Tello."""
    info = {"sdk_version": None, "serial_number": None}
    try:
        info["sdk_version"] = flight.tello.query_sdk_version()
    except Exception:
        pass
    try:
        info["serial_number"] = flight.tello.query_serial_number()
    except Exception:
        pass
    return info


# move_* calls block the loop until the Tello acknowledges — intended for a
# deliberate manual takeover.
def _manual_override(flight, key):
    if not flight.fly:
        return
    t = flight.tello
    if key == ord("w"):
        t.move_forward(NUDGE)
    elif key == ord("s"):
        t.move_back(NUDGE)
    elif key == ord("a"):
        t.move_left(NUDGE)
    elif key == ord("d"):
        t.move_right(NUDGE)
    elif key == ord("e"):
        t.rotate_clockwise(NUDGE)
    elif key == ord("q"):
        t.rotate_counter_clockwise(NUDGE)
    elif key == ord("r"):
        t.move_up(NUDGE)
    elif key == ord("f"):
        t.move_down(NUDGE)


def main():
    parser = argparse.ArgumentParser(description="Tello stationary toddler watcher")
    parser.add_argument("--no-fly", action="store_true", help="run detection/logic without arming motors")
    parser.add_argument("--battery-floor", type=int, default=20)
    parser.add_argument("--log", default=None, metavar="DIR", help="write a timestamped flight log under DIR")
    parser.add_argument("--note", default="", help="free-text note stored in the log metadata")
    args = parser.parse_args()

    flight = Flight(fly=not args.no_fly)
    detector = PersonDetector(PROTOTXT, CAFFEMODEL, conf_threshold=CONF_THRESHOLD)
    sm = StateMachine(
        StateConfig(reacquire_after_s=REACQUIRE_AFTER_S, scan_yaw=SCAN_YAW),
        ControlConfig(kp=KP, deadzone=DEADZONE, max_yaw=MAX_YAW),
    )

    flight.connect()
    print("battery:", flight.battery())
    flight.start_video()

    logger = None
    if args.log:
        first = flight.get_frame()
        frame_size = [first.shape[0], first.shape[1]] if first is not None else [None, None]
        info = _drone_info(flight)
        meta = {
            "fly": not args.no_fly,
            "note": args.note,
            "frame_size": frame_size,
            "sdk_version": info["sdk_version"],
            "serial_number": info["serial_number"],
            "config": {
                "kp": KP, "deadzone": DEADZONE, "max_yaw": MAX_YAW,
                "scan_yaw": SCAN_YAW, "reacquire_after_s": REACQUIRE_AFTER_S,
                "battery_floor": args.battery_floor, "conf_threshold": CONF_THRESHOLD,
            },
        }
        logger = FlightLogger(args.log, meta)
        print("logging to", logger.run_dir)

    flight.takeoff()

    start = time.monotonic()
    prev = start
    frame_idx = 0
    try:
        try:
            while True:
                now_m = time.monotonic()
                loop_dt_ms = int((now_m - prev) * 1000)
                prev = now_m
                t_rel = now_m - start

                frame = flight.get_frame()
                video_ok = frame_is_live(frame)

                detect_t0 = time.monotonic()
                detections = detector.detect(frame) if video_ok else []
                detect_ms = int((time.monotonic() - detect_t0) * 1000)

                target = select_target(detections) if video_ok else None
                if video_ok:
                    state, cmd = sm.update(target, frame.shape[1], frame.shape[0], time.time())
                    offset_norm = horizontal_offset_norm(target, frame.shape[1])
                else:
                    state, cmd, offset_norm = None, None, None

                key = cv2.waitKey(1) & 0xFF
                kill = key == 27  # ESC

                # djitellopy's get_battery reads the cached background state stream,
                # not a blocking query, so polling each loop is cheap; an exception
                # here means the link/state is unavailable.
                try:
                    battery = flight.battery()
                    connected = True
                except Exception:
                    battery, connected = 0, False

                action = assess(
                    battery=battery,
                    connected=connected,
                    video_ok=video_ok,
                    kill_requested=kill,
                    battery_floor=args.battery_floor,
                )
                land_reason = None
                if action == SafetyAction.LAND:
                    if kill:
                        land_reason = "kill (ESC)"
                    elif not connected:
                        land_reason = "connection loss"
                    elif not video_ok:
                        land_reason = "video loss"
                    else:
                        land_reason = "low battery"

                if logger is not None:
                    logger.log_frame(
                        t=t_rel, frame_idx=frame_idx, loop_dt_ms=loop_dt_ms,
                        detect_ms=detect_ms, state=state, detections=detections,
                        target=target, offset_norm=offset_norm, cmd=cmd,
                        battery=battery, video_ok=video_ok, connected=connected,
                        kill=kill, action=action.name, land_reason=land_reason,
                        drone=flight.state(),
                    )
                frame_idx += 1

                if action == SafetyAction.LAND:
                    print("LAND triggered:", land_reason)
                    break

                if key in MANUAL_KEYS:
                    _manual_override(flight, key)
                elif cmd is not None:
                    flight.send(cmd)

                if video_ok:
                    if target is not None:
                        cv2.rectangle(frame, (target.x, target.y),
                                      (target.x + target.w, target.y + target.h), (0, 255, 0), 2)
                    cv2.putText(frame, str(state), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("tello-watch", frame)
        except KeyboardInterrupt:
            print("LAND triggered: Ctrl-C")
        except Exception as e:
            print("LAND triggered: unhandled error:", e)
    finally:
        flight.land()
        cv2.destroyAllWindows()
        flight.end()
        if logger is not None:
            logger.close()
            if logger.errors:
                print("WARNING: logger dropped", logger.errors, "records")


if __name__ == "__main__":
    main()
