# py/tello_watch/run.py
import argparse
import os
import time

import cv2

from .control import ControlConfig
from .flight import Flight
from .safety import SafetyAction, assess
from .statemachine import StateConfig, StateMachine
from .target import select_target
from .vision import PersonDetector

MODELS = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT = os.path.join(MODELS, "MobileNetSSD_deploy.prototxt")
CAFFEMODEL = os.path.join(MODELS, "MobileNetSSD_deploy.caffemodel")

NUDGE = 30  # cm, manual override step


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
    args = parser.parse_args()

    flight = Flight(fly=not args.no_fly)
    detector = PersonDetector(PROTOTXT, CAFFEMODEL)
    sm = StateMachine(
        StateConfig(reacquire_after_s=2.5, scan_yaw=20),
        ControlConfig(kp=0.3, deadzone=0.1, max_yaw=25),
    )

    flight.connect()
    print("battery:", flight.battery())
    flight.start_video()
    flight.takeoff()

    try:
        while True:
            frame = flight.get_frame()
            video_ok = frame is not None
            target = select_target(detector.detect(frame)) if video_ok else None
            state, cmd = sm.update(target, frame.shape[1], frame.shape[0], time.time()) if video_ok else (None, None)

            key = cv2.waitKey(1) & 0xFF
            kill = key == 27  # ESC

            action = assess(
                battery=flight.battery(),
                connected=True,
                video_ok=video_ok,
                kill_requested=kill,
                battery_floor=args.battery_floor,
            )
            if action == SafetyAction.LAND:
                print("LAND triggered:", "kill" if kill else "safety")
                break

            _manual_override(flight, key)
            if cmd is not None:
                flight.send(cmd)

            if video_ok:
                if target is not None:
                    cv2.rectangle(frame, (target.x, target.y),
                                  (target.x + target.w, target.y + target.h), (0, 255, 0), 2)
                cv2.putText(frame, str(state), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("tello-watch", frame)
    finally:
        flight.land()
        cv2.destroyAllWindows()
        flight.end()


if __name__ == "__main__":
    main()
