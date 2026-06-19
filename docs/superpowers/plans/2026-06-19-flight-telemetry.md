# Flight Telemetry & Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add numeric-only per-flight telemetry to the `tello_watch` watcher — a JSONL event log + run metadata written during flight, plus a `tello-watch-analyze` command that prints a text performance summary for tuning.

**Architecture:** Two new modules (`telemetry.py`, `analyze.py`) plus three small additive seams in existing code (`control.py` extracts the offset helper, `flight.py` adds `state()`, `run.py` gains `--log`/`--note` and per-frame logging). Logging is a no-op when `--log` is absent and can never raise into the flight loop.

**Tech Stack:** Python ≥3.11, `uv`, standard library only (`json`, `os`, `math`, `datetime`, `argparse`), `pytest`. No new third-party dependencies.

## Global Constraints

- **Use `uv` for everything** — never pip. Run tests with `uv run pytest`.
- **Python ≥ 3.11.**
- **Numeric-only v1** — log numbers only, no images/video/frames. No new dependencies (stdlib only).
- **Logging must never crash the flight loop** — `FlightLogger.log_frame` and `close` swallow and count errors (`logger.errors`), never raise.
- **`--log` absent → identical behavior to today** — no logger constructed, zero change to the existing flight path.
- **TDD throughout.** Pure functions (`build_record`, `horizontal_offset_norm`, all `analyze` functions) fully unit-tested; `FlightLogger` tested against a tmp dir; `flight.state()` tested with a mocked Tello; `run.py` wiring verified by import + `--help` + full suite (live flight is operator-run, no hardware here).
- **Config values logged in `meta.json` must be the exact values the runner uses** — single source so the log is reproducible.

---

### Task 1: Extract `horizontal_offset_norm` in `control.py`

**Files:**
- Modify: `py/tello_watch/control.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: `Box` (vision).
- Produces: `horizontal_offset_norm(target: Box | None, frame_w: int) -> float | None` — the normalized horizontal error `(target.cx - frame_w/2) / (frame_w/2)`, or `None` when `target is None`. `compute_command` is refactored to call it; its behavior and existing tests are unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_control.py
from tello_watch.control import horizontal_offset_norm


def test_offset_norm_none_when_no_target():
    assert horizontal_offset_norm(None, 200) is None


def test_offset_norm_zero_at_center():
    b = Box(x=80, y=0, w=40, h=40, conf=0.9)  # cx=100, frame center=100
    assert horizontal_offset_norm(b, 200) == 0.0


def test_offset_norm_right_of_center():
    b = Box(x=120, y=0, w=20, h=40, conf=0.9)  # cx=130 -> (130-100)/100
    assert abs(horizontal_offset_norm(b, 200) - 0.3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_control.py -v`
Expected: FAIL — `ImportError: cannot import name 'horizontal_offset_norm'`.

- [ ] **Step 3: Refactor the implementation**

Replace the body of `compute_command` and add the helper in `py/tello_watch/control.py` (keep `Command` and `ControlConfig` unchanged):

```python
def horizontal_offset_norm(target, frame_w):
    if target is None:
        return None
    center = frame_w / 2
    return (target.cx - center) / center


def compute_command(target, frame_w, frame_h, cfg):
    offset_norm = horizontal_offset_norm(target, frame_w)
    if offset_norm is None or abs(offset_norm) < cfg.deadzone:
        return Command()
    yaw = cfg.kp * offset_norm * 100
    yaw = max(-cfg.max_yaw, min(cfg.max_yaw, yaw))
    return Command(yaw=int(yaw))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_control.py -v`
Expected: PASS — the 3 new tests plus the 4 existing `compute_command` tests (7 total).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/control.py tests/test_control.py
git commit -m "Extract horizontal_offset_norm from compute_command"
```

---

### Task 2: `build_record` in `telemetry.py`

**Files:**
- Create: `py/tello_watch/telemetry.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: duck-typed `Box` (has `.x/.y/.w/.h/.conf`), `Command` (has `.lr/.fb/.ud/.yaw`), `State` (has `.name`) — no imports needed; values are passed in.
- Produces: `build_record(*, t, frame_idx, loop_dt_ms, detect_ms, state, detections, target, offset_norm, cmd, battery, video_ok, connected, kill, action, land_reason, drone) -> dict` — a pure, JSON-serializable record. `state` → its `.name` or `None`; `cmd` → `{"lr","fb","ud","yaw"}` or `None`; `target` → `[x,y,w,h,conf]` or `None`; `detections` → list of `[x,y,w,h,conf]`; `n_detections` = `len(detections)`; floats rounded (`t` 3dp, `offset_norm` 4dp, `conf` 3dp).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
import json

from tello_watch.telemetry import build_record
from tello_watch.vision import Box
from tello_watch.control import Command
from tello_watch.statemachine import State


def test_build_record_shape_and_serializable():
    rec = build_record(
        t=12.8442, frame_idx=193, loop_dt_ms=71, detect_ms=58,
        state=State.TRACK, detections=[Box(120, 40, 30, 80, 0.8231)],
        target=Box(120, 40, 30, 80, 0.8231), offset_norm=0.42344,
        cmd=Command(0, 0, 0, 19), battery=74, video_ok=True, connected=True,
        kill=False, action="CONTINUE", land_reason=None, drone={"yaw": -3, "tof": 120},
    )
    json.loads(json.dumps(rec))  # must be serializable
    assert rec["t"] == 12.844
    assert rec["state"] == "TRACK"
    assert rec["n_detections"] == 1
    assert rec["detections"] == [[120, 40, 30, 80, 0.823]]
    assert rec["target"] == [120, 40, 30, 80, 0.823]
    assert rec["offset_norm"] == 0.4234
    assert rec["cmd"] == {"lr": 0, "fb": 0, "ud": 0, "yaw": 19}
    assert rec["drone"] == {"yaw": -3, "tof": 120}


def test_build_record_handles_none_target_state_cmd():
    rec = build_record(
        t=1.0, frame_idx=1, loop_dt_ms=70, detect_ms=0,
        state=None, detections=[], target=None, offset_norm=None,
        cmd=None, battery=50, video_ok=False, connected=True,
        kill=False, action="CONTINUE", land_reason=None, drone={},
    )
    assert rec["state"] is None
    assert rec["target"] is None
    assert rec["cmd"] is None
    assert rec["offset_norm"] is None
    assert rec["n_detections"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.telemetry`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/telemetry.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/telemetry.py tests/test_telemetry.py
git commit -m "Add build_record telemetry record builder"
```

---

### Task 3: `FlightLogger` in `telemetry.py`

**Files:**
- Modify: `py/tello_watch/telemetry.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `build_record` (same module).
- Produces: `FlightLogger`:
  - `__init__(self, base_dir: str, meta: dict, now=None)` — `now` is a zero-arg callable returning a `datetime` (defaults to `datetime.now`). Creates `base_dir/<YYYY-MM-DDTHH-MM-SS>/`, writes `meta.json` (a copy of `meta` plus `"started_at"` = `now().isoformat()`), opens `events.jsonl` for append. Exposes `run_dir: str` and `errors: int`.
  - `log_frame(self, **fields) -> None` — `build_record(**fields)` → one JSON line; flushes every 30 frames; on any exception increments `self.errors` and returns (never raises).
  - `close(self) -> None` — flush + close; idempotent; never raises.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_telemetry.py
import os
from datetime import datetime

from tello_watch.telemetry import FlightLogger


def _fixed_now():
    return datetime(2026, 6, 19, 15, 30, 0)


def test_logger_creates_timestamped_dir_and_meta(tmp_path):
    logger = FlightLogger(str(tmp_path), {"fly": True, "config": {"kp": 0.3}}, now=_fixed_now)
    assert logger.run_dir.endswith("2026-06-19T15-30-00")
    meta = json.load(open(os.path.join(logger.run_dir, "meta.json")))
    assert meta["fly"] is True
    assert meta["config"]["kp"] == 0.3
    assert meta["started_at"] == "2026-06-19T15:30:00"
    logger.close()


def test_logger_appends_valid_jsonl(tmp_path):
    logger = FlightLogger(str(tmp_path), {}, now=_fixed_now)
    logger.log_frame(
        t=1.0, frame_idx=1, loop_dt_ms=70, detect_ms=10, state=State.TRACK,
        detections=[Box(1, 2, 3, 4, 0.9)], target=Box(1, 2, 3, 4, 0.9),
        offset_norm=0.1, cmd=Command(0, 0, 0, 5), battery=80, video_ok=True,
        connected=True, kill=False, action="CONTINUE", land_reason=None, drone={"yaw": 1},
    )
    logger.close()
    lines = open(os.path.join(logger.run_dir, "events.jsonl")).read().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["state"] == "TRACK" and rec["cmd"]["yaw"] == 5


def test_log_frame_never_raises_and_counts_errors(tmp_path):
    logger = FlightLogger(str(tmp_path), {}, now=_fixed_now)
    logger.log_frame(t=1.0)  # missing required kwargs -> build_record raises -> swallowed
    assert logger.errors == 1
    logger.close()


def test_close_is_idempotent(tmp_path):
    logger = FlightLogger(str(tmp_path), {}, now=_fixed_now)
    logger.close()
    logger.close()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL — `ImportError: cannot import name 'FlightLogger'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top and body of `py/tello_watch/telemetry.py`:

```python
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
```

(Place the `import` lines at the top of the file, above `build_record`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: PASS (6 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/telemetry.py tests/test_telemetry.py
git commit -m "Add FlightLogger writing meta.json and events.jsonl"
```

---

### Task 4: `Flight.state()` in `flight.py`

**Files:**
- Modify: `py/tello_watch/flight.py`
- Test: `tests/test_flight.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Flight.state(self) -> dict` — returns `self.tello.get_current_state()`; returns `{}` on any exception (never raises). Always runs regardless of `fly`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_flight.py
@patch("tello_watch.flight.Tello")
def test_state_returns_current_state(MockTello):
    from tello_watch.flight import Flight
    f = Flight(fly=False)
    f.tello.get_current_state.return_value = {"yaw": 10, "tof": 100}
    assert f.state() == {"yaw": 10, "tof": 100}


@patch("tello_watch.flight.Tello")
def test_state_returns_empty_dict_on_error(MockTello):
    from tello_watch.flight import Flight
    f = Flight(fly=False)
    f.tello.get_current_state.side_effect = RuntimeError("boom")
    assert f.state() == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flight.py -v`
Expected: FAIL — `AttributeError: 'Flight' object has no attribute 'state'`.

- [ ] **Step 3: Write minimal implementation**

Add a method to the `Flight` class in `py/tello_watch/flight.py`:

```python
    def state(self):
        try:
            return self.tello.get_current_state()
        except Exception:
            return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_flight.py -v`
Expected: PASS (5 tests: 3 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/flight.py tests/test_flight.py
git commit -m "Add Flight.state() drone-state reader"
```

---

### Task 5: `analyze.py` summary module + console script

**Files:**
- Create: `py/tello_watch/analyze.py`
- Modify: `pyproject.toml`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Consumes: a list of record dicts (as written by `build_record`) and a meta dict.
- Produces (all pure unless noted):
  - `effective_fps(records) -> float`
  - `detect_ms_stats(records) -> dict` (`{"avg","max"}`)
  - `target_present_ratio(records) -> float`
  - `state_durations(records) -> dict` (state-name → seconds; missing state keyed `"NONE"`)
  - `time_to_first_lock(records) -> float | None`
  - `rms_offset(records) -> float | None`
  - `in_deadzone_ratio(records, deadzone) -> float`
  - `battery_summary(records) -> dict` (`{"start","end","min"}`)
  - `land_reason(records) -> str | None`
  - `yaw_sign_check(records) -> dict` (`{"verdict","correlation","n"}`)
  - `load_run(run_dir) -> (records, meta)`
  - `summarize(run_dir) -> str`
  - `main() -> None` (console script `tello-watch-analyze`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.analyze`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/analyze.py
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
```

- [ ] **Step 4: Add the console script to `pyproject.toml`**

In the `[project.scripts]` table, add the analyze entry alongside the existing `tello-watch`:

```toml
[project.scripts]
tello-watch = "tello_watch.run:main"
tello-watch-analyze = "tello_watch.analyze:main"
```

- [ ] **Step 5: Re-sync so the new console script installs**

Run: `uv sync --extra dev`
Expected: re-installs the editable package; `tello-watch-analyze` becomes available.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: PASS (all tests in the file).
Run: `uv run tello-watch-analyze --help`
Expected: prints usage with a `run_dir` positional argument.

- [ ] **Step 7: Commit**

```bash
git add py/tello_watch/analyze.py tests/test_analyze.py pyproject.toml uv.lock
git commit -m "Add tello-watch-analyze flight-log summary"
```

---

### Task 6: Wire logging into `run.py` + document it

**Files:**
- Modify: `py/tello_watch/run.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `FlightLogger` (telemetry), `horizontal_offset_norm` (control), `Flight.state` (flight), and all previously-wired modules.
- Produces: `main()` gains `--log DIR` and `--note TEXT`; when `--log` is set it constructs a `FlightLogger` (printing its `run_dir`), logs one record per loop, closes it in `finally`, and warns if `logger.errors` is non-zero. When `--log` is absent, behavior is unchanged.

- [ ] **Step 1: Rewrite `py/tello_watch/run.py`**

Replace the file with this version (adds config constants so `meta.json` matches the runtime config, detect timing, and the logging calls; the safety/state/control logic is unchanged):

```python
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
        meta = {
            "fly": not args.no_fly,
            "note": args.note,
            "frame_size": frame_size,
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
```

- [ ] **Step 2: Verify the full suite + smoke checks**

Run: `uv run pytest -q`
Expected: PASS — all tests across the package (Tasks 1–5 plus prior work).
Run: `uv run python -c "import tello_watch.run"`
Expected: no output, exit 0.
Run: `uv run tello-watch --help`
Expected: usage lists `--no-fly`, `--battery-floor`, `--log`, `--note`.

- [ ] **Step 3: Document logging + analysis in `README.md`**

Add this section immediately before the `## How it works` heading:

```markdown
## Logging & analysis

Record a flight (works with or without `--no-fly`) and get a tuning summary.

```bash
# numeric-only log written to flights/<timestamp>/
uv run tello-watch --log flights --note "crawling, afternoon sun"

# summarize a run
uv run tello-watch-analyze flights/2026-06-19T15-30-00
```

Each run directory holds `meta.json` (the exact config that flew) and
`events.jsonl` (one numeric record per frame — no imagery). The summary reports
effective FPS, time-to-first-lock, RMS centering error, battery curve, and a
best-effort yaw-sign check. Start on the ground with `--no-fly --log flights` to
tune detection safely before arming the motors. The logs contain only numbers, so
they're safe to share for analysis; keep identifying detail out of `--note`.
```

- [ ] **Step 4: Commit**

```bash
git add py/tello_watch/run.py README.md
git commit -m "Wire flight logging into runner and document it"
```

---

## Notes

- `flight.state()` is called once per loop only when logging is active (inside the
  `if logger is not None` block), so non-logging flights are unaffected.
- The config constants in `run.py` (`KP`, `DEADZONE`, …) are now the single source
  for both the live `StateMachine`/`ControlConfig` and the logged `meta.json`,
  keeping logs reproducible. Tuning still happens by editing these constants.
