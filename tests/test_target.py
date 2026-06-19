from tello_watch.vision import Box
from tello_watch.target import select_target


def test_select_target_returns_shortest_box():
    tall = Box(x=0, y=0, w=40, h=120, conf=0.9)
    short = Box(x=100, y=50, w=30, h=60, conf=0.8)
    assert select_target([tall, short]) is short


def test_select_target_none_when_empty():
    assert select_target([]) is None
