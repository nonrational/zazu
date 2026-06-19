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
