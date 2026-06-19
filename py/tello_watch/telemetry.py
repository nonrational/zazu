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
