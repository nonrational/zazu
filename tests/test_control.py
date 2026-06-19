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
