import json
import os
from datetime import datetime

FLUSH_EVERY = 30


class FlightLogger:
    def __init__(self, base_dir, meta, now=None):
        now = now or datetime.now
        stamp = now().strftime("%Y-%m-%dT%H-%M-%S")
        self.run_dir = os.path.join(base_dir, stamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self.errors = 0
        self._count = 0
        self._closed = False
        meta = dict(meta)
        meta["started_at"] = now().isoformat()
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._fh = open(os.path.join(self.run_dir, "events.jsonl"), "a")

    def log_frame(self, **fields):
        try:
            rec = build_record(**fields)
            self._fh.write(json.dumps(rec) + "\n")
            self._count += 1
            if self._count % FLUSH_EVERY == 0:
                self._fh.flush()
        except Exception:
            self.errors += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            self.errors += 1


def build_record(*, t, frame_idx, loop_dt_ms, detect_ms, state, detections,
                 target, offset_norm, cmd, battery, video_ok, connected, kill,
                 action, land_reason, drone):
    return {
        "t": round(t, 3),
        "frame_idx": frame_idx,
        "loop_dt_ms": loop_dt_ms,
        "detect_ms": detect_ms,
        "state": state.name if state is not None else None,
        "n_detections": len(detections),
        "detections": [[b.x, b.y, b.w, b.h, round(b.conf, 3)] for b in detections],
        "target": ([target.x, target.y, target.w, target.h, round(target.conf, 3)]
                   if target is not None else None),
        "offset_norm": round(offset_norm, 4) if offset_norm is not None else None,
        "cmd": ({"lr": cmd.lr, "fb": cmd.fb, "ud": cmd.ud, "yaw": cmd.yaw}
                if cmd is not None else None),
        "battery": battery,
        "video_ok": video_ok,
        "connected": connected,
        "kill": kill,
        "action": action,
        "land_reason": land_reason,
        "drone": drone,
    }
