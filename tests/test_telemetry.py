import json

from tello_watch.telemetry import build_record
from tello_watch.vision import Box
from tello_watch.control import Command
from tello_watch.statemachine import State


def test_build_record_shape_and_serializable():
    rec = build_record(
        t=12.8442, frame_idx=193, loop_dt_ms=71, detect_ms=58,
        state=State.TRACK, detections=[Box(120, 40, 30, 80, 0.8231)],
        target=Box(120, 40, 30, 80, 0.8231), offset_norm=0.42344,
        cmd=Command(0, 0, 0, 19), battery=74, video_ok=True, connected=True,
        kill=False, action="CONTINUE", land_reason=None, drone={"yaw": -3, "tof": 120},
    )
    json.loads(json.dumps(rec))  # must be serializable
    assert rec["t"] == 12.844
    assert rec["state"] == "TRACK"
    assert rec["n_detections"] == 1
    assert rec["detections"] == [[120, 40, 30, 80, 0.823]]
    assert rec["target"] == [120, 40, 30, 80, 0.823]
    assert rec["offset_norm"] == 0.4234
    assert rec["cmd"] == {"lr": 0, "fb": 0, "ud": 0, "yaw": 19}
    assert rec["drone"] == {"yaw": -3, "tof": 120}


def test_build_record_handles_none_target_state_cmd():
    rec = build_record(
        t=1.0, frame_idx=1, loop_dt_ms=70, detect_ms=0,
        state=None, detections=[], target=None, offset_norm=None,
        cmd=None, battery=50, video_ok=False, connected=True,
        kill=False, action="CONTINUE", land_reason=None, drone={},
    )
    assert rec["state"] is None
    assert rec["target"] is None
    assert rec["cmd"] is None
    assert rec["offset_norm"] is None
    assert rec["n_detections"] == 0
