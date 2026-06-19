import json

from tello_watch.analyze import (
    effective_fps, detect_ms_stats, target_present_ratio, state_durations,
    time_to_first_lock, rms_offset, in_deadzone_ratio, battery_summary,
    land_reason, yaw_sign_check, summarize,
)


def rec(t, state=None, target=None, offset=None, cmd_yaw=0, dyaw=0.0,
        battery=80, detect_ms=10, action="CONTINUE", land_reason=None):
    return {
        "t": t, "state": state, "target": target, "offset_norm": offset,
        "cmd": {"lr": 0, "fb": 0, "ud": 0, "yaw": cmd_yaw}, "drone": {"yaw": dyaw},
        "battery": battery, "detect_ms": detect_ms, "action": action,
        "land_reason": land_reason,
    }


def test_effective_fps():
    assert abs(effective_fps([rec(0.0), rec(1.0), rec(2.0)]) - 1.0) < 1e-9


def test_detect_ms_stats():
    s = detect_ms_stats([rec(0, detect_ms=10), rec(1, detect_ms=30)])
    assert s == {"avg": 20.0, "max": 30}


def test_target_present_ratio():
    assert target_present_ratio([rec(0, target=[1, 2, 3, 4, 0.9]), rec(1, target=None)]) == 0.5


def test_state_durations():
    d = state_durations([rec(0, state="SCAN"), rec(1, state="TRACK"), rec(3, state="TRACK")])
    assert d == {"SCAN": 1.0, "TRACK": 2.0}


def test_time_to_first_lock():
    assert time_to_first_lock([rec(0, state="SCAN"), rec(0.5, state="SCAN"), rec(1.0, state="TRACK")]) == 1.0


def test_time_to_first_lock_never():
    assert time_to_first_lock([rec(0, state="SCAN")]) is None


def test_rms_offset_track_only():
    recs = [rec(0, state="TRACK", offset=0.3), rec(1, state="TRACK", offset=-0.3),
            rec(2, state="SCAN", offset=0.9)]
    assert abs(rms_offset(recs) - 0.3) < 1e-9


def test_in_deadzone_ratio():
    recs = [rec(0, state="TRACK", offset=0.05), rec(1, state="TRACK", offset=0.5)]
    assert in_deadzone_ratio(recs, 0.1) == 0.5


def test_battery_summary():
    assert battery_summary([rec(0, battery=80), rec(1, battery=60), rec(2, battery=70)]) == {
        "start": 80, "end": 70, "min": 60}


def test_land_reason():
    assert land_reason([rec(0), rec(1, action="LAND", land_reason="low battery")]) == "low battery"


def _yaw_series(slope):
    recs, t, y = [], 0.0, 0.0
    for cy in [10, 20, -10, 15, -20, 25, -15, 10, 20, -10]:
        recs.append(rec(t, cmd_yaw=cy, dyaw=y))
        y += slope * cy * 0.1
        t += 0.1
    recs.append(rec(t, cmd_yaw=0, dyaw=y))
    return recs


def test_yaw_sign_check_ok():
    res = yaw_sign_check(_yaw_series(+1))
    assert res["verdict"] == "ok" and res["correlation"] > 0 and res["n"] >= 5


def test_yaw_sign_check_flipped():
    res = yaw_sign_check(_yaw_series(-1))
    assert res["verdict"].startswith("likely flipped") and res["correlation"] < 0


def test_yaw_sign_check_inconclusive_few_samples():
    res = yaw_sign_check([rec(0, cmd_yaw=10, dyaw=0), rec(0.1, cmd_yaw=10, dyaw=1)])
    assert res["verdict"] == "inconclusive" and res["n"] < 5


def test_summarize_runs(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"config": {"deadzone": 0.1}}))
    (d / "events.jsonl").write_text(
        json.dumps(rec(0, state="TRACK", offset=0.05, target=[1, 2, 3, 4, 0.9])) + "\n"
        + json.dumps(rec(1, state="TRACK", offset=0.05, target=[1, 2, 3, 4, 0.9])) + "\n"
    )
    out = summarize(str(d))
    assert "Effective FPS" in out
    assert "Yaw-sign check" in out
