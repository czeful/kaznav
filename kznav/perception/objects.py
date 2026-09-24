"""Object candidates: 3D localisation of detections and their association across frames.

A detection's mask (or box) is back-projected with the depth image; the candidate's position is
the median of those world points and its footprint is kept as a small set of map points, so the
policy can approach "the nearest observed part" of the object. Detections within
``merge_radius_m`` of an existing candidate update it, whichever query fired: a candidate is one
physical object, and ``scores`` keeps the best detector score per query (the match vector of the
hypothesis-belief update). ``last_box`` is the most recent detection box (the verifier's crop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .base import Detection
from .camera import Intrinsics, backproject, to_world


@dataclass
class ObjectCandidate:
    cid: int
    query_index: int
    position: np.ndarray                 # habitat world (median of observed points)
    points_xy: np.ndarray                # (N, 2) map-frame points, subsampled
    best_score: float
    n_detections: int
    first_step: int
    last_step: int
    unreachable: bool = False
    verification: List[float] = field(default_factory=list)   # verifier probabilities (calibrated)
    scores: Dict[int, float] = field(default_factory=dict)     # query index -> best detector score
    last_box: Optional[tuple] = None                           # (x1, y1, x2, y2) of the latest detection
    rejected: bool = False                                     # the verifier said "not the target"
    unseen_in_range: int = 0                                   # policy steps in stop range without a detection
    lost: bool = False                                         # never re-observed from close by: given up

    def match_vector(self, n_queries: int) -> np.ndarray:
        """Per-query association strength in [0, 1], relative to the best-matching query."""
        m = np.array([self.scores.get(k, 0.0) for k in range(n_queries)], dtype=float)
        return m / m.max() if m.max() > 0 else m

    def priority(self, weights: Optional[Sequence[float]] = None) -> float:
        if weights is None:
            return self.best_score
        return float(sum(w * self.scores.get(k, 0.0) for k, w in enumerate(weights)))

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.position[0], -self.position[2]])

    def nearest_distance(self, xy: Sequence[float]) -> float:
        return float(np.min(np.linalg.norm(self.points_xy - np.asarray(xy)[None, :], axis=1)))


@dataclass(frozen=True)
class DetectionParams:
    score_threshold: float
    min_mask_pixels: int
    merge_radius_m: float
    max_points: int
    max_depth_m: float

    @classmethod
    def from_config(cls, d: Mapping[str, Any], max_depth_m: float) -> "DetectionParams":
        return cls(float(d["score_threshold"]), int(d["min_mask_pixels"]), float(d["merge_radius_m"]),
                   int(d["max_points_per_detection"]), float(max_depth_m))


class ObjectMemory:
    def __init__(self, params: DetectionParams, K: Intrinsics, rng_seed: int = 0):
        self.p = params
        self.K = K
        self.candidates: List[ObjectCandidate] = []
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Forget all candidates; with ``seed``, restart the point-subsampling RNG (per-episode determinism)."""
        self.candidates = []
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def localize(self, mask: np.ndarray, depth: np.ndarray, T_world_cam: np.ndarray) -> Optional[np.ndarray]:
        """World points of a mask with valid depth (subsampled), or None."""
        v, u = np.nonzero(mask & (depth > 0) & (depth <= self.p.max_depth_m))
        if len(v) < self.p.min_mask_pixels:
            return None
        if len(v) > self.p.max_points:
            idx = self._rng.choice(len(v), self.p.max_points, replace=False)
            v, u = v[idx], u[idx]
        return to_world(backproject(depth, self.K, v, u), T_world_cam)

    def add(self, det: Detection, mask: np.ndarray, depth: np.ndarray, T_world_cam: np.ndarray,
            step: int) -> Optional[ObjectCandidate]:
        if det.score < self.p.score_threshold:
            return None
        pts = self.localize(mask, depth, T_world_cam)
        if pts is None:
            return None
        pos = np.median(pts, axis=0)
        xy = np.stack([pts[:, 0], -pts[:, 2]], axis=1)
        for c in self.candidates:
            if np.hypot(*(c.xy - np.array([pos[0], -pos[2]]))) <= self.p.merge_radius_m:
                n = c.n_detections
                c.position = (c.position * n + pos) / (n + 1)
                keep = c.points_xy if len(c.points_xy) < self.p.max_points else \
                    c.points_xy[self._rng.choice(len(c.points_xy), self.p.max_points // 2, replace=False)]
                c.points_xy = np.concatenate([keep, xy[: self.p.max_points // 2]])
                c.best_score = max(c.best_score, det.score)
                c.scores[det.query_index] = max(c.scores.get(det.query_index, 0.0), det.score)
                c.n_detections += 1
                if step > c.last_step or c.last_box is None or _area(det.box_xyxy) > _area(c.last_box):
                    c.last_box = tuple(float(v) for v in det.box_xyxy)   # largest box of the latest frame
                c.last_step = step
                return c
        c = ObjectCandidate(len(self.candidates), det.query_index, pos, xy, det.score, 1, step, step,
                            scores={det.query_index: det.score}, last_box=tuple(float(v) for v in det.box_xyxy))
        self.candidates.append(c)
        return c

    def best(self, exclude_unreachable: bool = True, weights: Optional[Sequence[float]] = None) -> Optional[ObjectCandidate]:
        """Most promising live candidate: highest (hypothesis-weighted) detector score; rejected ones never."""
        live = [c for c in self.candidates
                if not c.rejected and not c.lost and not (exclude_unreachable and c.unreachable)]
        if not live:
            return None
        return max(live, key=lambda c: (c.priority(weights), c.n_detections))


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_mask(det: Detection, shape: Sequence[int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = det.box_xyxy
    m = np.zeros((h, w), dtype=bool)
    m[max(0, int(y1)) : min(h, int(np.ceil(y2))), max(0, int(x1)) : min(w, int(np.ceil(x2)))] = True
    return m
