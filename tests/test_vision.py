import numpy as np
from tello_watch.vision import parse_detections, Box, PERSON_CLASS_ID


def test_box_center():
    b = Box(x=10, y=20, w=40, h=60, conf=0.9)
    assert b.cx == 30.0
    assert b.cy == 50.0


def test_parse_detections_keeps_person_above_threshold():
    # raw shape (1, 1, N, 7): [img_id, class_id, conf, x1, y1, x2, y2] normalized
    raw = np.array([[[
        [0, PERSON_CLASS_ID, 0.9, 0.25, 0.5, 0.75, 1.0],  # person, kept
        [0, 7,               0.99, 0.0, 0.0, 1.0, 1.0],     # car, dropped (wrong class)
        [0, PERSON_CLASS_ID, 0.2, 0.0, 0.0, 0.1, 0.1],      # person, dropped (low conf)
    ]]], dtype=np.float32)

    boxes = parse_detections(raw, frame_w=200, frame_h=100, conf_threshold=0.5)

    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x, b.y, b.w, b.h) == (50, 50, 100, 50)
    assert abs(b.conf - 0.9) < 1e-6
