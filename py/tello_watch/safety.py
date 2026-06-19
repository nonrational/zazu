from enum import Enum, auto


class SafetyAction(Enum):
    CONTINUE = auto()
    LAND = auto()


def assess(*, battery, connected, video_ok, kill_requested, battery_floor):
    if kill_requested or not connected or not video_ok or battery < battery_floor:
        return SafetyAction.LAND
    return SafetyAction.CONTINUE
