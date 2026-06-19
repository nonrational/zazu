from dataclasses import dataclass

import cv2
import numpy as np

PERSON_CLASS_ID = 15  # MobileNet-SSD VOC class index for "person"


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def parse_detections(raw: np.ndarray, frame_w: int, frame_h: int, conf_threshold: float = 0.5) -> list[Box]:
    boxes = []
    detections = raw[0, 0]  # shape (N, 7)
    for det in detections:
        class_id = int(det[1])
        conf = float(det[2])
        if class_id != PERSON_CLASS_ID or conf < conf_threshold:
            continue
        x1 = int(det[3] * frame_w)
        y1 = int(det[4] * frame_h)
        x2 = int(det[5] * frame_w)
        y2 = int(det[6] * frame_h)
        boxes.append(Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1, conf=conf))
    return boxes


class PersonDetector:
    def __init__(self, prototxt, model, conf_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        raw = self.net.forward()
        return parse_detections(raw, w, h, self.conf_threshold)
