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
