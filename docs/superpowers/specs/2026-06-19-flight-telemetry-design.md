# Flight Telemetry & Analysis — Design

**Date:** 2026-06-19\
**Status:** Approved, pending implementation plan\
**Builds on:** the `tello_watch` stationary watcher (Version A)

## Goal

Collect structured per-flight data from real Tello flights so tracking performance
can be analyzed and the control/scan/safety knobs tuned with evidence instead of
guesses. v1 is **numeric-only** (no video): a JSONL event log plus run metadata,
and a local `analyze` command that prints a text summary.

Out of scope for v1 (future slices): video / annotated-MP4 recording, plots,
compression, automatic parameter search.

## Why these data

Every logged field maps to a tuning decision:

- `offset_norm` over time vs commanded `yaw` → tune `kp`, `deadzone`, `max_yaw`
  (oscillation = gain too high; lag = too low; visible dead band; saturation).
- `state` timeline, time-to-first-lock, target-present ratio → tune `scan_yaw`,
  `reacquire_after_s`.
- `detections` / `n_detections` / confidences → tune `conf_threshold`, spot
  multi-person mislocks.
- `loop_dt_ms` / `detect_ms` → latency budget (whether to detect every Nth frame).
- `drone.yaw` change vs commanded `yaw` → detect the yaw-sign gotcha and estimate
  gain (best-effort).

## Architecture

Two new modules plus three small seams in existing code.

```
py/tello_watch/
  telemetry.py   # build_record (pure) + FlightLogger (writes meta.json + events.jsonl)
  analyze.py     # pure summary functions + `tello-watch-analyze` entry point
```

Existing-code seams:

- **`control.py`** — extract `horizontal_offset_norm(target, frame_w) -> float | None`
  (the normalized horizontal control error). `compute_command` calls it so the
  logger records the exact number the controller used. Existing `compute_command`
  behavior and tests are unchanged.
- **`flight.py`** — add `state(self) -> dict` wrapping `tello.get_current_state()`;
  returns `{}` on any error (never raises).
- **`run.py`** — add `--log DIR` and `--note TEXT` flags; time the detect call;
  construct a `FlightLogger` when `--log` is set; call `logger.log_frame(...)` each
  iteration and `logger.close()` in the `finally`. When `--log` is absent, no
  logger is constructed and behavior is identical to today.

### Module responsibilities

**`telemetry.py`**

- `build_record(*, t, frame_idx, loop_dt_ms, detect_ms, state, detections, target,
  offset_norm, cmd, battery, video_ok, connected, kill, action, land_reason,
  drone) -> dict` — a pure function assembling one JSON-serializable record. No I/O.
  `state` is logged as its name (e.g. `"TRACK"`) or `null`; `cmd` as
  `{"lr","fb","ud","yaw"}`; `detections`/`target` as `[x,y,w,h,conf]` lists (or
  `null` target); `action` as `"CONTINUE"`/`"LAND"`.
- `FlightLogger`:
  - `__init__(self, base_dir: str, meta: dict, now=datetime.now)` — creates a
    timestamped run directory `base_dir/<YYYY-MM-DDTHH-MM-SS>/`, opens
    `events.jsonl` for buffered append. `meta` is augmented with `started_at`
    (ISO-8601) and written to `meta.json` immediately.
  - `log_frame(self, **fields) -> None` — calls `build_record(**fields)`,
    `json.dumps` it, writes one line. Wrapped in try/except: on failure it
    increments `self.errors` and returns — it MUST never raise into the flight
    loop. Flushes every `FLUSH_EVERY` (30) frames.
  - `close(self) -> None` — flush and close; safe to call twice; never raises.
  - `run_dir` attribute exposes the created directory path (printed by the runner).

**`analyze.py`**

- Pure functions over loaded records (list of dicts) + meta, each independently
  testable:
  - `effective_fps(records) -> float`
  - `detect_ms_stats(records) -> {"avg","max"}`
  - `target_present_ratio(records) -> float`
  - `state_durations(records) -> {state_name: seconds}`
  - `time_to_first_lock(records) -> float | None` (first `t` with state `TRACK`,
    relative to first record's `t`)
  - `rms_offset(records) -> float | None` (RMS of `offset_norm` over TRACK frames
    with a non-null offset)
  - `in_deadzone_ratio(records, deadzone) -> float` (TRACK frames with
    `abs(offset_norm) < deadzone`)
  - `battery_summary(records) -> {"start","end","min"}`
  - `land_reason(records) -> str | None`
  - `yaw_sign_check(records) -> {"verdict","correlation","n"}` — correlation
    between commanded `yaw` and the per-frame change in `drone.yaw` (unwrapped
    across the −180..180 boundary). Verdict is `"ok"`, `"likely flipped — try
    negating kp"`, or `"inconclusive"` (too few samples / near-zero correlation).
    Clearly labeled best-effort.
  - `load_run(run_dir) -> (records, meta)` — reads `events.jsonl` + `meta.json`.
  - `summarize(run_dir) -> str` — composes the text report.
  - `main()` — argparse `run_dir`, prints `summarize(run_dir)`. Console script
    `tello-watch-analyze`.

## Data layout

`--log flights` creates `flights/<YYYY-MM-DDTHH-MM-SS>/`:

- `meta.json` (written once at start):
  ```json
  {"started_at":"2026-06-19T15:30:00Z","fly":true,"note":"crawling, afternoon sun",
   "frame_size":[720,960],"sdk_version":"<best-effort>","serial_number":"<best-effort>",
   "config":{"kp":0.3,"deadzone":0.1,"max_yaw":25,"scan_yaw":20,
             "reacquire_after_s":2.5,"battery_floor":20,"conf_threshold":0.5}}
  ```
- `events.jsonl` (one record per loop):
  ```json
  {"t":12.84,"frame_idx":193,"loop_dt_ms":71,"detect_ms":58,"state":"TRACK",
   "n_detections":2,"detections":[[120,40,30,80,0.82]],"target":[120,40,30,80,0.82],
   "offset_norm":0.42,"cmd":{"lr":0,"fb":0,"ud":0,"yaw":19},
   "battery":74,"video_ok":true,"connected":true,"kill":false,
   "action":"CONTINUE","land_reason":null,
   "drone":{"yaw":-3,"vgx":0,"vgy":0,"vgz":0,"tof":120,"h":110,"templ":50,"time":42}}
  ```

`t` is a monotonic seconds value relative to logger start. For `frame_size`, the
runner reads one frame after `start_video()` (before the main loop), takes its
`shape` as `[height, width]`, and includes it in the `meta` dict passed to the
`FlightLogger` constructor — so `meta.json` is complete when first written.
`sdk_version`/`serial_number` are best-effort and may be `null` if the SDK does not
expose them.

## Error handling

- Logging must never crash the flight loop: `log_frame` and `close` swallow and
  count errors. The runner checks `logger.errors` after the loop and prints a
  warning if non-zero.
- `flight.state()` returns `{}` on any error; `drone` may therefore be `{}`.
- `--log` absent → no logger, identical behavior to today.
- `analyze` on a malformed/empty run reports what it can and notes missing data
  rather than crashing.

## Privacy

v1 logs no imagery, only numbers — safe to send for analysis. The `--note` field
is operator-supplied free text; keep identifying detail out of it.

## Testing

- `telemetry.build_record` — pure; unit-tested for field shape and serializability.
- `telemetry.FlightLogger` — writes to a tmp dir; assert `meta.json` + valid JSONL
  lines; assert `log_frame` never raises on bad input and increments `errors`; no
  drone involved.
- `analyze` summary functions — deterministic unit tests over synthetic record
  lists with known answers (FPS, ratios, RMS, time-to-lock, yaw-sign verdict on
  crafted positive/negative-correlation data).
- `control.horizontal_offset_norm` — unit tests; existing `compute_command` tests
  remain green (refactor only).
- `run.py` — `--log`/`--note` wiring verified by `import tello_watch.run`,
  `tello-watch --help`, and full suite; live flight is operator-run (no hardware).
