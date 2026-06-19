# Tello Toddler Watcher — Design

**Date:** 2026-06-19\
**Status:** Approved, pending implementation plan\
**Platform:** Ryze Tech Tello, driven from a MacBook over Wi-Fi (Python / djitellopy / OpenCV)

## Goal

A harness, run on the MacBook, that instructs the Tello to take off, scan the room
to locate a toddler, lock on, and keep him centered in frame as he moves.

Delivered in two versions:

- **Version A (this spec's deliverable): stationary watcher.** The drone hovers in
  roughly one spot and only *yaws* (rotates) to keep the toddler centered. No
  chasing. "Security camera on a stick."
- **Version B (future, additive): active follower.** The drone translates
  (forward/back/strafe) to follow at a distance. Built later by extending one
  module; out of scope here.

## Hard constraints and safety stance

The Tello has **a single 2D front camera (no depth), no obstacle avoidance, exposed
props, and ~13 min of flight per battery.** It cannot map space or avoid collisions.
Version A is designed so the drone never needs to approach the child — it rotates in
place and watches. Prop guards on the physical drone are strongly recommended.

Full house-following ("follow him around the house") is explicitly **not achievable
on this platform** and is not a goal of this project. It would require a SLAM-class
platform (e.g. Skydio) and is a separate, much larger effort.

## Target identification

- v1 detects **people** and selects the **shortest bounding box** as the toddler
  (a height heuristic to prefer him over adults in the room).
- The detector and the selector are separate, swappable modules, so "specifically
  him" (face recognition or a worn marker) can replace the heuristic later without
  touching the rest of the system.

## Architecture — small, swappable modules

```
py/
  tello_watch/
    flight.py        # thin wrapper over djitellopy.Tello: connect, takeoff, land, yaw, hover, battery
    vision.py        # video frame -> list of person bounding boxes (swappable detector)
    target.py        # boxes -> chosen target (v1: shortest box = toddler heuristic)
    control.py       # target offset -> flight command (v1: yaw only; B adds translation here)
    safety.py        # battery floor, connection/video watchdog, manual kill, rate caps
    statemachine.py  # SCAN -> TRACK -> REACQUIRE, plus failsafe transitions
    run.py           # wires it together; the entry point run on the Mac
```

**Key seam for Version B:** `control.py` is the only module that decides *how the
drone moves*. Version A emits yaw-only commands; Version B adds forward/back/strafe
in the same module. Detection, target selection, safety, and the state machine are
identical across A and B.

### Module responsibilities

- **`flight.py`** — wraps `djitellopy.Tello`. Exposes connect, takeoff, land,
  hover, rate-capped yaw, and battery/state reads. Honors a global `FLY` toggle so
  logic can be exercised without arming the motors.
- **`vision.py`** — OpenCV DNN module running **MobileNet-SSD**, filtered to the
  `person` class. Input: a BGR frame. Output: a list of `(x, y, w, h, confidence)`
  boxes. Runs on the Mac CPU at ~10–15 fps. No new pip dependencies beyond the
  existing OpenCV; the model `.prototxt` and `.caffemodel` files are checked in.
- **`target.py`** — given a list of boxes, returns the chosen target (v1: the
  shortest box above a confidence threshold), or `None` if no person is present.
- **`control.py`** — given the target's offset from frame center and the frame
  dimensions, returns a flight command. v1: proportional **yaw** to recenter
  horizontally, with a center deadzone to avoid jitter. Commands are rate-capped.
- **`safety.py`** — evaluated every loop, can override any state. Triggers an
  immediate land on: manual kill (ESC / Ctrl-C), battery below floor, connection
  loss, video loss, or unhandled error. Enforces yaw-rate and command-size caps.
- **`statemachine.py`** — drives SCAN -> TRACK -> REACQUIRE and the failsafe
  transitions (see below).
- **`run.py`** — the entry point: connects, takes off, runs the loop, draws the
  live video window with the target box, and routes keyboard input (kill switch +
  the existing manual-flight overrides).

## Control behavior — Version A

Proportional yaw control. Measure the horizontal distance between the target box
center and the frame center; command a small, rate-capped yaw to recenter. A
deadzone around the center keeps the drone still when the toddler is roughly
centered. Yaw only — no translation in Version A.

## State machine

- **SCAN** — slow yaw sweep of the room. On detecting the shortest person, lock and
  transition to TRACK. This is the "orient itself in a room" step.
- **TRACK** — proportional yaw keeps the target centered.
- **REACQUIRE** — target not detected for ~2–3 s -> hover and resume a slow
  re-scan. On re-detection -> TRACK. **This state never self-lands.** Only the
  safety layer initiates a landing.

## Failsafe policy

- **Manual kill switch always live** — ESC and Ctrl-C trigger an immediate land
  from any state. The keyboard manual-flight loop from the existing `main.py`
  remains available as a live override layer.
- **Uncertainty (lost target, timeout) -> hover**, not land. The drone keeps
  watching and re-scanning; it does not come down on its own for losing him.
- **Battery floor: 20%** -> stop and land. (The Tello also auto-lands when
  critically low regardless.)
- **Connection or video loss** -> hover then land.
- **No sudden moves** — yaw rate and command size are hard-capped; small smooth
  increments only.

## Testing

- **`vision.py` / `target.py`** — tested offline against saved video clips / still
  frames. Deterministic, no drone required.
- **`control.py` / `statemachine.py`** — unit-tested with synthetic target inputs.
  Pure logic, no hardware.
- **`flight.py` + full loop** — verified live with the `FLY` toggle, battery
  checks, and the kill switch validated first.

## Out of scope

- Version B active following (translation-based follow).
- Face recognition / specific-person identification.
- Spatial mapping / SLAM / obstacle avoidance (not possible on this platform).
- Following the child between rooms or around the house.
