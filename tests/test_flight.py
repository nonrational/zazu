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
