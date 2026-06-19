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
