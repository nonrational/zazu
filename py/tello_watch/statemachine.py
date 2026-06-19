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
