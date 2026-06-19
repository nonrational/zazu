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
