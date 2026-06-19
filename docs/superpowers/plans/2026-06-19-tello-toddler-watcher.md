# Tello Toddler Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A MacBook-run harness that makes a Ryze Tello take off, scan a room to find a toddler (shortest detected person), and yaw to keep him centered in frame — a stationary watcher (Version A).

**Architecture:** Small single-responsibility modules under `py/tello_watch/`. A per-frame loop reads the Tello video stream, detects people (OpenCV DNN / MobileNet-SSD), selects the shortest box as the target, and a state machine (SCAN → TRACK → REACQUIRE) plus a control module produce yaw commands. A safety layer evaluated every loop can override any state and land. The only module that decides *how the drone moves* is `control.py`, so Version B (active following) is an additive change there.

**Tech Stack:** Python ≥3.11, `uv` for dependency management and execution, `djitellopy` (Tello SDK), `opencv-python` (video + DNN inference), `numpy`, `pytest`.

## Global Constraints

- **Use `uv` for everything** — deps via `pyproject.toml` + `uv.lock`, execution via `uv run`. Do NOT use pip or `py/requirements.txt` (legacy).
- **Python ≥ 3.11.**
- **Version A only:** yaw-only motion. No translation (lr/fb/ud stay 0). Translation is Version B, out of scope.
- **Target = shortest detected person** above the confidence threshold.
- **Failsafe policy:** uncertainty (lost target / timeout) → **hover**, never auto-land. Land ONLY on: manual kill (ESC / Ctrl-C), battery **< 20%**, connection loss, video loss, or unhandled error.
- **No sudden moves:** yaw is rate-capped (`max_yaw`), with a center deadzone to avoid jitter.
- **`FLY` toggle** gates all motor commands so logic runs without arming the drone.
- **TDD throughout.** Pure-logic modules (`vision.parse_detections`, `target`, `control`, `safety`, `statemachine`) are fully unit-tested with no hardware. `flight.py` is tested with a mocked `Tello`. `run.py` is verified live.

---

### Task 1: Project setup with uv

**Files:**
- Create: `pyproject.toml`
- Create: `py/tello_watch/__init__.py` (empty package marker)
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore` (add `.venv/`, `__pycache__/`, `*.pyc`)

**Interfaces:**
- Consumes: nothing.
- Produces: an installed editable package `tello_watch` importable from tests and `uv run`; `uv run pytest` runs the suite.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "tello-watch"
version = "0.1.0"
description = "Stationary Tello watcher that yaws to keep a toddler centered in frame"
requires-python = ">=3.11"
dependencies = [
    "djitellopy>=2.5.0",
    "opencv-python>=4.9.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
tello-watch = "tello_watch.run:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["py"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package + test markers**

Create empty `py/tello_watch/__init__.py` and `tests/__init__.py`. Add to `.gitignore`:

```
.venv/
__pycache__/
*.pyc
```

- [ ] **Step 3: Sync the environment**

Run: `uv sync --extra dev`
Expected: creates `.venv/`, resolves deps, writes `uv.lock`, installs `tello-watch` editable. Ends with a summary like `Installed N packages`.

- [ ] **Step 4: Verify the toolchain**

Run: `uv run python -c "import cv2, numpy, djitellopy; print('ok', cv2.__version__)"`
Expected: prints `ok 4.x.x` with no ImportError.
Run: `uv run pytest -q`
Expected: `no tests ran` (exit 5) — confirms pytest is wired before any tests exist.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock py/tello_watch/__init__.py tests/__init__.py .gitignore
git commit -m "Set up tello_watch package with uv"
```

---

### Task 2: Detection — `vision.py`

**Files:**
- Create: `py/tello_watch/vision.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Box` dataclass: fields `x: int, y: int, w: int, h: int, conf: float`; properties `cx -> float`, `cy -> float` (box center).
  - `PERSON_CLASS_ID = 15` (MobileNet-SSD VOC person class).
  - `parse_detections(raw: np.ndarray, frame_w: int, frame_h: int, conf_threshold: float = 0.5) -> list[Box]` — pure parser of MobileNet-SSD output `(1,1,N,7)` rows `[img_id, class_id, conf, x1, y1, x2, y2]` (normalized 0–1). Keeps only `class_id == PERSON_CLASS_ID` rows with `conf >= conf_threshold`.
  - `PersonDetector` class: `__init__(self, prototxt: str, model: str, conf_threshold: float = 0.5)`, method `detect(self, frame: np.ndarray) -> list[Box]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vision.py
import numpy as np
from tello_watch.vision import parse_detections, Box, PERSON_CLASS_ID


def test_box_center():
    b = Box(x=10, y=20, w=40, h=60, conf=0.9)
    assert b.cx == 30.0
    assert b.cy == 50.0


def test_parse_detections_keeps_person_above_threshold():
    # raw shape (1, 1, N, 7): [img_id, class_id, conf, x1, y1, x2, y2] normalized
    raw = np.array([[[
        [0, PERSON_CLASS_ID, 0.9, 0.25, 0.5, 0.75, 1.0],  # person, kept
        [0, 7,               0.99, 0.0, 0.0, 1.0, 1.0],     # car, dropped (wrong class)
        [0, PERSON_CLASS_ID, 0.2, 0.0, 0.0, 0.1, 0.1],      # person, dropped (low conf)
    ]]], dtype=np.float32)

    boxes = parse_detections(raw, frame_w=200, frame_h=100, conf_threshold=0.5)

    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x, b.y, b.w, b.h) == (50, 50, 100, 50)
    assert abs(b.conf - 0.9) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vision.py -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError: tello_watch.vision`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/vision.py
from dataclasses import dataclass

import cv2
import numpy as np

PERSON_CLASS_ID = 15  # MobileNet-SSD VOC class index for "person"


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def parse_detections(raw, frame_w, frame_h, conf_threshold=0.5):
    boxes = []
    detections = raw[0, 0]  # shape (N, 7)
    for det in detections:
        class_id = int(det[1])
        conf = float(det[2])
        if class_id != PERSON_CLASS_ID or conf < conf_threshold:
            continue
        x1 = int(det[3] * frame_w)
        y1 = int(det[4] * frame_h)
        x2 = int(det[5] * frame_w)
        y2 = int(det[6] * frame_h)
        boxes.append(Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1, conf=conf))
    return boxes


class PersonDetector:
    def __init__(self, prototxt, model, conf_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        raw = self.net.forward()
        return parse_detections(raw, w, h, self.conf_threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vision.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Download the model files**

The `PersonDetector` needs MobileNet-SSD Caffe weights. Download into `py/tello_watch/models/`:

```bash
mkdir -p py/tello_watch/models
curl -L -o py/tello_watch/models/MobileNetSSD_deploy.prototxt \
  https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt
curl -L -o py/tello_watch/models/MobileNetSSD_deploy.caffemodel \
  https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel
```

Verify the caffemodel is the real ~23 MB binary, not an HTML error page:

Run: `uv run python -c "import cv2; cv2.dnn.readNetFromCaffe('py/tello_watch/models/MobileNetSSD_deploy.prototxt','py/tello_watch/models/MobileNetSSD_deploy.caffemodel'); print('model loads')"`
Expected: prints `model loads`. If it errors, the download URL rotted — find another raw-hosted `MobileNetSSD_deploy.caffemodel`/`.prototxt` pair (VOC, 21 classes, person=15) and re-run.

- [ ] **Step 6: Commit**

```bash
git add py/tello_watch/vision.py tests/test_vision.py py/tello_watch/models/
git commit -m "Add MobileNet-SSD person detector"
```

---

### Task 3: Target selection — `target.py`

**Files:**
- Create: `py/tello_watch/target.py`
- Test: `tests/test_target.py`

**Interfaces:**
- Consumes: `Box` from `tello_watch.vision`.
- Produces: `select_target(boxes: list[Box]) -> Box | None` — returns the box with the smallest `h` (toddler height heuristic), or `None` if the list is empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_target.py
from tello_watch.vision import Box
from tello_watch.target import select_target


def test_select_target_returns_shortest_box():
    tall = Box(x=0, y=0, w=40, h=120, conf=0.9)
    short = Box(x=100, y=50, w=30, h=60, conf=0.8)
    assert select_target([tall, short]) is short


def test_select_target_none_when_empty():
    assert select_target([]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_target.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.target`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/target.py
def select_target(boxes):
    if not boxes:
        return None
    return min(boxes, key=lambda b: b.h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_target.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/target.py tests/test_target.py
git commit -m "Add shortest-person target selector"
```

---

### Task 4: Control — `control.py`

**Files:**
- Create: `py/tello_watch/control.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: `Box` from `tello_watch.vision`.
- Produces:
  - `Command` dataclass: `lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0` (RC velocities, range −100..100). Version A only sets `yaw`.
  - `ControlConfig` dataclass: `kp: float, deadzone: float, max_yaw: int`.
  - `compute_command(target: Box | None, frame_w: int, frame_h: int, cfg: ControlConfig) -> Command` — proportional yaw to recenter the target horizontally; `Command()` (all zeros) when `target is None` or within the deadzone; yaw clamped to `±cfg.max_yaw`. Positive yaw turns toward a target right of center.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control.py
from tello_watch.vision import Box
from tello_watch.control import compute_command, ControlConfig, Command

CFG = ControlConfig(kp=0.3, deadzone=0.1, max_yaw=25)


def test_no_target_yields_hover():
    assert compute_command(None, 200, 100, CFG) == Command()


def test_centered_target_yields_no_yaw():
    b = Box(x=80, y=0, w=40, h=40, conf=0.9)  # cx=100 == frame center
    assert compute_command(b, 200, 100, CFG) == Command(0, 0, 0, 0)


def test_target_right_of_center_yields_proportional_positive_yaw():
    b = Box(x=120, y=0, w=20, h=40, conf=0.9)  # cx=130, offset_norm=0.3
    cmd = compute_command(b, 200, 100, CFG)
    assert cmd.yaw == 9  # int(0.3 * 0.3 * 100)
    assert cmd.lr == 0 and cmd.fb == 0 and cmd.ud == 0


def test_far_target_yaw_is_clamped():
    b = Box(x=198, y=0, w=2, h=40, conf=0.9)  # cx=199, large offset
    cmd = compute_command(b, 200, 100, CFG)
    assert cmd.yaw == CFG.max_yaw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_control.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.control`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/control.py
from dataclasses import dataclass


@dataclass
class Command:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0


@dataclass
class ControlConfig:
    kp: float
    deadzone: float
    max_yaw: int


def compute_command(target, frame_w, frame_h, cfg):
    if target is None:
        return Command()
    center = frame_w / 2
    offset_norm = (target.cx - center) / center
    if abs(offset_norm) < cfg.deadzone:
        return Command()
    yaw = cfg.kp * offset_norm * 100
    yaw = max(-cfg.max_yaw, min(cfg.max_yaw, yaw))
    return Command(yaw=int(yaw))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_control.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/control.py tests/test_control.py
git commit -m "Add proportional yaw control"
```

---

### Task 5: Safety — `safety.py`

**Files:**
- Create: `py/tello_watch/safety.py`
- Test: `tests/test_safety.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SafetyAction` enum: `CONTINUE`, `LAND`.
  - `assess(*, battery: int, connected: bool, video_ok: bool, kill_requested: bool, battery_floor: int) -> SafetyAction` — returns `LAND` if `kill_requested`, not `connected`, not `video_ok`, or `battery < battery_floor`; otherwise `CONTINUE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safety.py
from tello_watch.safety import assess, SafetyAction

BASE = dict(battery=80, connected=True, video_ok=True, kill_requested=False, battery_floor=20)


def test_continue_when_all_nominal():
    assert assess(**BASE) == SafetyAction.CONTINUE


def test_land_when_battery_below_floor():
    assert assess(**{**BASE, "battery": 19}) == SafetyAction.LAND


def test_land_on_kill_request():
    assert assess(**{**BASE, "kill_requested": True}) == SafetyAction.LAND


def test_land_on_disconnect():
    assert assess(**{**BASE, "connected": False}) == SafetyAction.LAND


def test_land_on_video_loss():
    assert assess(**{**BASE, "video_ok": False}) == SafetyAction.LAND
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safety.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.safety`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/safety.py
from enum import Enum, auto


class SafetyAction(Enum):
    CONTINUE = auto()
    LAND = auto()


def assess(*, battery, connected, video_ok, kill_requested, battery_floor):
    if kill_requested or not connected or not video_ok or battery < battery_floor:
        return SafetyAction.LAND
    return SafetyAction.CONTINUE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safety.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/safety.py tests/test_safety.py
git commit -m "Add failsafe assessment"
```

---

### Task 6: State machine — `statemachine.py`

**Files:**
- Create: `py/tello_watch/statemachine.py`
- Test: `tests/test_statemachine.py`

**Interfaces:**
- Consumes: `Box` (vision); `compute_command`, `Command`, `ControlConfig` (control).
- Produces:
  - `State` enum: `SCAN`, `TRACK`, `REACQUIRE`.
  - `StateConfig` dataclass: `reacquire_after_s: float`, `scan_yaw: int`.
  - `StateMachine` class: `__init__(self, state_cfg: StateConfig, control_cfg: ControlConfig)`; method `update(self, target: Box | None, frame_w: int, frame_h: int, now: float) -> tuple[State, Command]`.

Behavior: a present `target` → `TRACK` with `compute_command(...)`. No target while never-seen → stays `SCAN`, emits `Command(yaw=scan_yaw)`. No target after a prior lock → `TRACK` grace-hover (`Command()`) until lost longer than `reacquire_after_s`, then `REACQUIRE` emitting `Command(yaw=scan_yaw)`. Never returns a landing decision — that is the safety layer's job.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_statemachine.py
from tello_watch.vision import Box
from tello_watch.statemachine import StateMachine, State, StateConfig
from tello_watch.control import ControlConfig, Command


def make_sm():
    return StateMachine(
        StateConfig(reacquire_after_s=2.0, scan_yaw=20),
        ControlConfig(kp=0.3, deadzone=0.1, max_yaw=25),
    )


def test_starts_in_scan_and_emits_scan_yaw():
    sm = make_sm()
    state, cmd = sm.update(None, 200, 100, now=0.0)
    assert state == State.SCAN
    assert cmd == Command(yaw=20)


def test_locks_on_target_and_tracks():
    sm = make_sm()
    state, cmd = sm.update(Box(x=120, y=0, w=20, h=40, conf=0.9), 200, 100, now=1.0)
    assert state == State.TRACK
    assert cmd.yaw == 9 and cmd.lr == 0  # proportional yaw toward right


def test_brief_loss_hovers_then_reacquires():
    sm = make_sm()
    sm.update(Box(x=120, y=0, w=20, h=40, conf=0.9), 200, 100, now=1.0)  # TRACK, last_seen=1.0
    state, cmd = sm.update(None, 200, 100, now=2.0)  # lost 1.0s < 2.0 -> grace hover
    assert state == State.TRACK and cmd == Command()
    state, cmd = sm.update(None, 200, 100, now=3.5)  # lost 2.5s > 2.0 -> REACQUIRE scan
    assert state == State.REACQUIRE and cmd == Command(yaw=20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_statemachine.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.statemachine`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/statemachine.py
from dataclasses import dataclass
from enum import Enum, auto

from .control import Command, compute_command


class State(Enum):
    SCAN = auto()
    TRACK = auto()
    REACQUIRE = auto()


@dataclass
class StateConfig:
    reacquire_after_s: float
    scan_yaw: int


class StateMachine:
    def __init__(self, state_cfg, control_cfg):
        self.state_cfg = state_cfg
        self.control_cfg = control_cfg
        self.state = State.SCAN
        self._last_seen = None

    def update(self, target, frame_w, frame_h, now):
        if target is not None:
            self._last_seen = now
            self.state = State.TRACK
            return self.state, compute_command(target, frame_w, frame_h, self.control_cfg)

        lost_for = float("inf") if self._last_seen is None else now - self._last_seen
        if lost_for > self.state_cfg.reacquire_after_s:
            # never-seen -> keep scanning; previously locked -> reacquire scan
            self.state = State.SCAN if self._last_seen is None else State.REACQUIRE

        if self.state in (State.SCAN, State.REACQUIRE):
            return self.state, Command(yaw=self.state_cfg.scan_yaw)
        return self.state, Command()  # TRACK grace window, no target -> hover
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_statemachine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/statemachine.py tests/test_statemachine.py
git commit -m "Add SCAN/TRACK/REACQUIRE state machine"
```

---

### Task 7: Flight wrapper — `flight.py`

**Files:**
- Create: `py/tello_watch/flight.py`
- Test: `tests/test_flight.py`

**Interfaces:**
- Consumes: `Command` from `tello_watch.control`.
- Produces: `Flight` class wrapping `djitellopy.Tello`:
  - `__init__(self, fly: bool = True)`
  - `connect(self) -> None`, `battery(self) -> int`, `start_video(self) -> None`, `get_frame(self) -> np.ndarray`
  - `takeoff(self) -> None`, `send(self, cmd: Command) -> None`, `land(self) -> None`, `end(self) -> None`
  - When `fly is False`, `takeoff`/`send`/`land` are no-ops (logic runs without arming motors). `connect`/`battery`/video always execute.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flight.py
from unittest.mock import patch
from tello_watch.control import Command


@patch("tello_watch.flight.Tello")
def test_fly_false_does_not_command_motors(MockTello):
    from tello_watch.flight import Flight
    f = Flight(fly=False)
    f.takeoff()
    f.send(Command(yaw=20))
    f.land()
    f.tello.takeoff.assert_not_called()
    f.tello.send_rc_control.assert_not_called()
    f.tello.land.assert_not_called()


@patch("tello_watch.flight.Tello")
def test_fly_true_sends_rc_control(MockTello):
    from tello_watch.flight import Flight
    f = Flight(fly=True)
    f.send(Command(lr=0, fb=0, ud=0, yaw=15))
    f.tello.send_rc_control.assert_called_once_with(0, 0, 0, 15)


@patch("tello_watch.flight.Tello")
def test_battery_reads_through(MockTello):
    from tello_watch.flight import Flight
    f = Flight(fly=False)
    f.tello.get_battery.return_value = 73
    assert f.battery() == 73
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flight.py -v`
Expected: FAIL — `ModuleNotFoundError: tello_watch.flight`.

- [ ] **Step 3: Write minimal implementation**

```python
# py/tello_watch/flight.py
from djitellopy import Tello


class Flight:
    def __init__(self, fly=True):
        self.fly = fly
        self.tello = Tello()

    def connect(self):
        self.tello.connect()

    def battery(self):
        return self.tello.get_battery()

    def start_video(self):
        self.tello.streamoff()
        self.tello.streamon()

    def get_frame(self):
        return self.tello.get_frame_read().frame

    def takeoff(self):
        if self.fly:
            self.tello.takeoff()

    def send(self, cmd):
        if self.fly:
            self.tello.send_rc_control(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

    def land(self):
        if self.fly:
            self.tello.land()

    def end(self):
        self.tello.end()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_flight.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/flight.py tests/test_flight.py
git commit -m "Add Tello flight wrapper with FLY toggle"
```

---

### Task 8: Runner — `run.py` (entry point + live verification)

**Files:**
- Create: `py/tello_watch/run.py`

**Interfaces:**
- Consumes: `Flight` (flight); `PersonDetector` (vision); `select_target` (target); `StateMachine`, `StateConfig` (statemachine); `ControlConfig` (control); `assess`, `SafetyAction` (safety).
- Produces: `main() -> None` — the console-script entry point (`tello-watch`).

Behavior: parse `--no-fly` (default flies) and `--battery-floor` (default 20). Connect, read battery, start video, take off. Each loop: read frame, `detector.detect(frame)`, `select_target(...)`, `StateMachine.update(target, w, h, now=time.time())`, then `assess(...)`; on `LAND` → land and exit, else `flight.send(cmd)`. Draw the target box + state text on the frame and `cv2.imshow`. Keyboard: `ESC` → kill (land + exit); `w/a/s/d/q/e/r/f` → the manual override nudges from the legacy `py/main.py` keyboard loop (preserved as the live override layer). Always land + `end()` in a `finally`.

- [ ] **Step 1: Write the runner**

```python
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
```

- [ ] **Step 2: Verify the full suite still passes**

Run: `uv run pytest -q`
Expected: PASS — all tests from Tasks 2–7 green.

- [ ] **Step 3: Live verify with motors OFF (props will not spin)**

With the Tello powered on and the Mac joined to its Wi-Fi:
Run: `uv run tello-watch --no-fly`
Expected: prints battery; a `tello-watch` window shows the live feed; standing in view draws a green box; the shortest person is boxed when multiple people are present; state text reads `SCAN` then `TRACK`. ESC closes cleanly. **No motor movement.**

- [ ] **Step 4: Live verify with motors ON (clear area, prop guards, hand on ESC)**

In an open space, props guarded, ready to hit ESC:
Run: `uv run tello-watch`
Expected: takes off, yaw-scans, locks onto the shortest person and yaws to keep them centered; walking left/right makes it rotate to follow; leaving frame >2.5s → it hovers then resumes scanning; ESC lands immediately. Confirm it lands on low battery.

- [ ] **Step 5: Commit**

```bash
git add py/tello_watch/run.py
git commit -m "Add tello-watch runner entry point"
```

---

## Notes on the legacy code

`py/main.py` (manual keyboard flight) and `py/requirements.txt` are superseded by this package and `pyproject.toml`. Leave them in place for now as a reference; a follow-up can remove `requirements.txt` once the uv flow is confirmed on the user's Mac. The manual override keys (`w/a/s/d/q/e/r/f`) are preserved in `run.py`.
