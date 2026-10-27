"""Object memory: 3-D localisation, association and filtering of detections; the oracle detector; box masks."""
import numpy as np
import pytest

from kznav.perception.base import Detection
from kznav.perception.camera import Intrinsics
from kznav.perception.objects import DetectionParams, ObjectMemory, box_mask
from kznav.perception.oracle import OracleSemanticDetector

K = Intrinsics.from_hfov(69.0, 160, 120)
P = DetectionParams(score_threshold=0.35, min_mask_pixels=5, merge_radius_m=0.6, max_points=200, max_depth_m=4.5)


def cam(x=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, 0.15, z]
    return T


def patch():
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    mask = np.zeros((120, 160), bool)
    mask[50:70, 70:90] = True
    return depth, mask


def test_localises_a_fronto_parallel_patch():
    depth, mask = patch()
    mem = ObjectMemory(P, K)
    c = mem.add(Detection((70, 50, 90, 70), 0.9, 0, "chair"), mask, depth, cam(), step=3)
    assert c is not None
    assert c.position[2] == pytest.approx(-2.0, abs=1e-6)          # 2 m straight ahead (-Z)
    assert abs(c.position[0]) < 0.05
    assert c.xy[1] == pytest.approx(2.0, abs=1e-6)
    assert c.nearest_distance((0.0, 0.0)) == pytest.approx(2.0, abs=0.05)


def test_merges_nearby_and_splits_far_detections():
    depth, mask = patch()
    mem = ObjectMemory(P, K)
    det = Detection((70, 50, 90, 70), 0.9, 0, "chair")
    mem.add(det, mask, depth, cam(), step=0)
    mem.add(det, mask, depth, cam(x=0.2), step=1)                  # same object seen 0.2 m to the side
    assert len(mem.candidates) == 1 and mem.candidates[0].n_detections == 2
    mem.add(det, mask, depth, cam(x=3.0), step=2)
    assert len(mem.candidates) == 2


def test_low_score_and_empty_masks_ignored():
    depth, _ = patch()
    mem = ObjectMemory(P, K)
    assert mem.add(Detection((0, 0, 10, 10), 0.1, 0, "x"), np.ones((120, 160), bool), depth, cam(), 0) is None
    assert mem.add(Detection((0, 0, 10, 10), 0.9, 0, "x"), np.zeros((120, 160), bool), depth, cam(), 0) is None


def test_oracle_detector_returns_goal_instances_only():
    sem = np.zeros((120, 160), np.int32)
    sem[10:30, 10:30] = 7
    sem[50:60, 50:60] = 3
    det = OracleSemanticDetector({"min_pixels": 20})
    out = det.detect(np.zeros((120, 160, 3), np.uint8), ["chair"], {"semantic": sem, "goal_ids": {0: [7]}})
    assert len(out) == 1 and out[0].box_xyxy == (10.0, 10.0, 30.0, 30.0) and out[0].mask.sum() == 400
    with pytest.raises(ValueError):
        det.detect(np.zeros((120, 160, 3), np.uint8), ["chair"], None)


def test_box_mask():
    m = box_mask(Detection((2.2, 3.0, 5.0, 6.5), 1.0, 0, "x"), (10, 10))
    assert m.sum() == 4 * 3 and m[3, 2] and not m[7, 2]      # rows 3..6, cols 2..4
