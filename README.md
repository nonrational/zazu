# zazu

There's one in every family.

A MacBook harness for a Ryze Tech Tello that takes off, scans a room, locks onto
the shortest person it sees (a toddler-height heuristic), and yaws to keep them
centered in frame — a stationary "watcher" (Version A). It does not chase or fly
around the room; it rotates in place and watches.

## ⚠️ Safety first

The Tello has exposed props and no obstacle avoidance. Fly with **prop guards**,
in a clear space, and keep a hand on the kill switch (**ESC**). The harness lands
automatically on low battery, lost video, or lost connection — but those signals
are best-effort, so you are the real-time failsafe. Always do a props-off dry run
(`--no-fly`) before arming the motors.

## Getting started

### Prerequisites

- macOS with [`uv`](https://docs.astral.sh/uv/) installed.
- A Ryze Tello, powered on, with your Mac joined to the drone's `TELLO-XXXXXX`
  Wi-Fi network.
- The MobileNet-SSD model files are checked into `py/tello_watch/models/` — no
  separate download needed.

### Setup

```bash
uv sync --extra dev
```

This creates the virtualenv, installs dependencies, and installs the `tello-watch`
command (editable). Verify the toolchain:

```bash
uv run python -c "import cv2, djitellopy; print('ok', cv2.__version__)"
uv run pytest -q
```

### Run

**Always start with a props-off dry run.** Motors stay disarmed; you get the live
video feed with detection overlays so you can confirm it finds and boxes the
shortest person:

```bash
uv run tello-watch --no-fly
```

When that looks right, arm the motors (clear space, prop guards, hand on ESC):

```bash
uv run tello-watch
```

It takes off, yaw-scans the room, locks onto the shortest person, and rotates to
keep them centered. Leaving the frame for ~2.5s makes it hover and resume
scanning — it does not land on a lost target.

Flags:

- `--no-fly` — run detection and the control logic without arming motors.
- `--battery-floor N` — auto-land threshold in percent (default `20`).

Controls (in the video window):

- **ESC** — immediate land and exit (kill switch). **Ctrl-C** also lands.
- `w`/`a`/`s`/`d` — manual nudge forward/left/back/right
- `q`/`e` — rotate counter-clockwise/clockwise
- `r`/`f` — up/down

### Tests

```bash
uv run pytest
```

The detection, target-selection, control, safety, and state-machine logic are
unit-tested with no hardware. The flight wrapper is tested with a mocked Tello.

## Tuning

The defaults are starting guesses — expect to adjust them once you watch it fly.
Most knobs live where the runner constructs its configs in `py/tello_watch/run.py`:

```python
sm = StateMachine(
    StateConfig(reacquire_after_s=2.5, scan_yaw=20),
    ControlConfig(kp=0.3, deadzone=0.1, max_yaw=25),
)
```

| Knob | Where | Default | Effect |
|---|---|---|---|
| `kp` | `run.py` `ControlConfig` | `0.3` | Yaw aggressiveness. Higher = snaps to center faster but can overshoot/oscillate; lower = smoother but lags a moving toddler. |
| `deadzone` | `run.py` `ControlConfig` | `0.1` | Fraction of half-frame around center where it holds still (no yaw). Bigger = steadier, ignores small drift; smaller = twitchier. |
| `max_yaw` | `run.py` `ControlConfig` | `25` | Hard cap on yaw rate (−100..100). Keep modest so it never whips around near the child. |
| `scan_yaw` | `run.py` `StateConfig` | `20` | Yaw rate while scanning/reacquiring. Lower = slower, more thorough sweep; too high blurs the feed and misses people. |
| `reacquire_after_s` | `run.py` `StateConfig` | `2.5` | Seconds a target can be lost before it stops grace-hovering and resumes scanning. |
| `--battery-floor` | CLI flag | `20` | Auto-land battery threshold (%). |
| `conf_threshold` | `PersonDetector(...)` in `run.py:54` | `0.5` | Detection confidence. Raise (e.g. `0.6`) to cut false positives; lower to catch a partially-visible toddler. Pass it: `PersonDetector(PROTOTXT, CAFFEMODEL, conf_threshold=0.6)`. |
| `NUDGE` | `run.py` | `30` | Manual-override step in cm. |

### Yaw-sign gotcha

If on the first flight it rotates **away** from your toddler instead of toward
him, the yaw sign is backwards for your setup — **flip the sign of `kp`** (e.g.
`kp=-0.3`). Positive yaw is meant to turn the drone toward a target right of
center; it's easy to have this inverted until you see it move.

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

## How it works

The `tello_watch` package is split into single-purpose modules:

- `vision.py` — video frame → person bounding boxes (OpenCV / MobileNet-SSD)
- `target.py` — boxes → the shortest box (toddler heuristic)
- `control.py` — target offset → yaw command (the only module that decides motion)
- `safety.py` — battery / connection / video / kill checks → land decision
- `statemachine.py` — SCAN → TRACK → REACQUIRE
- `flight.py` — thin `djitellopy.Tello` wrapper with a `FLY` toggle
- `run.py` — wires it together; the `tello-watch` entry point

Active following (the drone translating to follow someone) is a future Version B —
it would change only `control.py`.

## Legacy

`py/main.py` (a manual keyboard-flight prototype) and `rb/` (a Ruby Tello console)
predate this harness and are kept for reference. `py/requirements.txt` is
superseded by `pyproject.toml` + `uv` — use `uv sync`, not pip.
