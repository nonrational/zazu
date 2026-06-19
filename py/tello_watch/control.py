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
