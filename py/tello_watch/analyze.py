import argparse
import json
import math
import os


def load_run(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    records = []
    path = os.path.join(run_dir, "events.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records, meta


def effective_fps(records):
    if len(records) < 2:
        return 0.0
    span = records[-1]["t"] - records[0]["t"]
    return (len(records) - 1) / span if span > 0 else 0.0


def detect_ms_stats(records):
    vals = [r["detect_ms"] for r in records if r.get("detect_ms") is not None]
    if not vals:
        return {"avg": 0.0, "max": 0.0}
    return {"avg": sum(vals) / len(vals), "max": max(vals)}


def target_present_ratio(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("target") is not None) / len(records)


def state_durations(records):
    durations = {}
    for a, b in zip(records, records[1:]):
        dt = b["t"] - a["t"]
        if dt < 0:
            continue
        key = a.get("state") if a.get("state") is not None else "NONE"
        durations[key] = durations.get(key, 0.0) + dt
    return durations


def time_to_first_lock(records):
    if not records:
        return None
    t0 = records[0]["t"]
    for r in records:
        if r.get("state") == "TRACK":
            return r["t"] - t0
    return None


def rms_offset(records):
    vals = [r["offset_norm"] for r in records
            if r.get("state") == "TRACK" and r.get("offset_norm") is not None]
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def in_deadzone_ratio(records, deadzone):
    track = [r for r in records
             if r.get("state") == "TRACK" and r.get("offset_norm") is not None]
    if not track:
        return 0.0
    return sum(1 for r in track if abs(r["offset_norm"]) < deadzone) / len(track)


def battery_summary(records):
    vals = [r["battery"] for r in records if r.get("battery") is not None]
    if not vals:
        return {"start": None, "end": None, "min": None}
    return {"start": vals[0], "end": vals[-1], "min": min(vals)}


def land_reason(records):
    for r in records:
        if r.get("action") == "LAND":
            return r.get("land_reason")
    return None


def _unwrap_delta(prev, cur):
    d = cur - prev
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def yaw_sign_check(records):
    xs, ys = [], []
    for a, b in zip(records, records[1:]):
        cmd = a.get("cmd")
        da, db = a.get("drone") or {}, b.get("drone") or {}
        if cmd is None or "yaw" not in da or "yaw" not in db:
            continue
        cyaw = cmd.get("yaw", 0)
        if cyaw == 0:
            continue
        xs.append(cyaw)
        ys.append(_unwrap_delta(da["yaw"], db["yaw"]))
    n = len(xs)
    if n < 5:
        return {"verdict": "inconclusive", "correlation": None, "n": n}
    corr = _pearson(xs, ys)
    if corr is None or abs(corr) < 0.3:
        verdict = "inconclusive"
    elif corr > 0:
        verdict = "ok"
    else:
        verdict = "likely flipped — try negating kp"
    return {"verdict": verdict, "correlation": corr, "n": n}


def summarize(run_dir):
    records, meta = load_run(run_dir)
    deadzone = meta.get("config", {}).get("deadzone", 0.1)
    dms = detect_ms_stats(records)
    ttl = time_to_first_lock(records)
    rms = rms_offset(records)
    bat = battery_summary(records)
    ys = yaw_sign_check(records)
    lr = land_reason(records)
    sd = state_durations(records)
    sd_str = ", ".join(f"{k} {v:.1f}" for k, v in sorted(sd.items())) if sd else "n/a"
    corr = f"{ys['correlation']:.2f}" if ys["correlation"] is not None else "n/a"

    lines = [
        f"Run: {run_dir}",
        f"Frames: {len(records)}   Effective FPS: {effective_fps(records):.1f}",
        f"detect_ms: avg {dms['avg']:.0f}   max {dms['max']:.0f}",
        f"Target present: {target_present_ratio(records) * 100:.0f}%",
        f"State durations (s): {sd_str}",
        f"Time to first lock: {ttl:.1f}s" if ttl is not None else "Time to first lock: never",
        f"RMS centering error (TRACK): {rms:.3f}" if rms is not None else "RMS centering error: n/a",
        f"Time within deadzone ({deadzone}): {in_deadzone_ratio(records, deadzone) * 100:.0f}%",
        f"Battery: start {bat['start']}  end {bat['end']}  min {bat['min']}",
        f"Land reason: {lr}" if lr is not None else "Land reason: n/a",
        f"Yaw-sign check (best-effort): {ys['verdict']} (corr {corr}, n={ys['n']})",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize a tello-watch flight log")
    parser.add_argument("run_dir", help="path to a flights/<timestamp> run directory")
    args = parser.parse_args()
    print(summarize(args.run_dir))


if __name__ == "__main__":
    main()
