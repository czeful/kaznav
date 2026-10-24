"""Object candidates: 3D localisation of detections and their association across frames.

A detection's mask (or box) is back-projected with the depth image; the candidate's position is
the median of those world points and its footprint is kept as a small set of map points, so the
policy can approach "the nearest observed part" of the object. Without a real segmenter the mask is
the whole box, and from a camera 0.15 m above the floor a box reaches down to the floor just in front
of the robot: those floor points became the candidate's "nearest part" and the robot stopped at once
(D-054). So the points are filtered: points less than ``floor_margin_m`` above the floor are dropped,
and of the rest only the front layer (depth within ``depth_band_m`` of the nearest 5%) is kept - the
object, not the wall behind it. Detections within
``merge_radius_m`` of an existing candidate update it, whichever query fired: a candidate is one
physical object, and ``scores`` keeps the best detector score per query (the match vector of the
hypothesis-belief update). ``last_box`` is the most recent detection box (the verifier's crop).

Target-centric fusion (D-060, after ApexNav, Zhang et al. 2025), when ``fusion`` is on: every candidate keeps a
confidence ``conf`` and an observed volume ``volume`` (number of its localised points). Once per frame
(``end_frame``) a candidate detected in the frame is updated with its best score of the frame, weighted by the
points of that detection: conf <- (volume * conf + v * score) / (volume + v); a candidate whose stored points
should be visible in the frame (in the image, in depth range, not occluded) but was not detected gets the same
update with score 0 and v = its visible points (negative evidence). Candidates with conf >= ``reliable_conf``
are "reliable" and the only ones the policy approaches; the rest are "suspected", considered only when
exploration has nothing left. Without fusion, ranking is by the best detector score, as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .base import Detection
from .camera import Intrinsics, backproject, to_world

MAX_FUSION_POINTS = 200          # world points kept per candidate for the visibility test


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
    conf: float = 0.0                                          # fused confidence (D-060)
    volume: float = 0.0                                        # observed points behind ``conf``
    points_world: Optional[np.ndarray] = None                  # (N, 3) subsample, for the visibility test
    misses: int = 0                                            # frames it should have been seen but was not
    points_h: Optional[np.ndarray] = None                      # (N,) height above the floor of each ``points_xy``
    confirmed: bool = False                                    # the verifier accepted it from afar (D-070)
    second_look_steps: int = 0                                 # steps spent moving for a second look (D-071)
    no_new_view: bool = False                                  # no reachable cell for a second look
    crop: Optional[np.ndarray] = None                          # image crop of its best detection (D-072)
    crop_score: float = -1.0                                   # detector score of that detection

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
    floor_margin_m: float = 0.0      # drop points less than this above the floor (0 = keep all)
    depth_band_m: float = 0.0        # keep points within this of the nearest 5% of depths (0 = keep all)
    fusion: bool = False             # D-060 target-centric fusion
    reliable_conf: float = 0.0       # fused confidence a candidate needs to be approached
    min_visible_points: int = 20     # stored points in view needed for negative evidence
    occlusion_tol_m: float = 0.3     # a stored point is occluded when the observed depth is this much nearer

    @classmethod
    def from_config(cls, d: Mapping[str, Any], max_depth_m: float) -> "DetectionParams":
        f = d.get("point_filter") or {}
        fu = d.get("fusion") or {}
        return cls(float(d["score_threshold"]), int(d["min_mask_pixels"]), float(d["merge_radius_m"]),
                   int(d["max_points_per_detection"]), float(max_depth_m),
                   float(f.get("floor_margin_m", 0.0)), float(f.get("depth_band_m", 0.0)),
                   bool(fu.get("enabled", False)), float(fu.get("reliable_confidence", 0.0)),
                   int(fu.get("min_visible_points", 20)), float(fu.get("occlusion_tolerance_m", 0.3)))


class ObjectMemory:
    def __init__(self, params: DetectionParams, K: Intrinsics, rng_seed: int = 0):
        self.p = params
        self.K = K
        self.candidates: List[ObjectCandidate] = []
        self._rng = np.random.default_rng(rng_seed)
        self._frame: Dict[int, tuple] = {}        # cid -> (best score, points) of the current frame

    @property
    def fusion(self) -> bool:
        return self.p.fusion

    def reset(self, seed: Optional[int] = None) -> None:
        """Forget all candidates; with ``seed``, restart the point-subsampling RNG (per-episode determinism)."""
        self.candidates = []
        self._frame = {}
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def localize(self, mask: np.ndarray, depth: np.ndarray, T_world_cam: np.ndarray,
                 floor_y: Optional[float] = None, precise: bool = False) -> Optional[np.ndarray]:
        """World points of a mask with valid depth (floor and background filtered, D-054; subsampled), or None.
        ``precise``: the mask is the object's contour (a real segmenter), so the front-layer depth band is skipped."""
        v, u = np.nonzero(mask & (depth > 0) & (depth <= self.p.max_depth_m))
        if len(v) < self.p.min_mask_pixels:
            return None
        if floor_y is not None and self.p.floor_margin_m > 0:
            pts = to_world(backproject(depth, self.K, v, u), T_world_cam)
            above = pts[:, 1] - floor_y >= self.p.floor_margin_m
            v, u = v[above], u[above]
            if len(v) < self.p.min_mask_pixels:
                return None
        if self.p.depth_band_m > 0 and not precise:
            d = depth[v, u]
            front = d <= np.quantile(d, 0.05) + self.p.depth_band_m
            v, u = v[front], u[front]
            if len(v) < self.p.min_mask_pixels:
                return None
        if len(v) > self.p.max_points:
            idx = self._rng.choice(len(v), self.p.max_points, replace=False)
            v, u = v[idx], u[idx]
        return to_world(backproject(depth, self.K, v, u), T_world_cam)

    def add(self, det: Detection, mask: np.ndarray, depth: np.ndarray, T_world_cam: np.ndarray,
            step: int, floor_y: Optional[float] = None, precise: bool = False) -> Optional[ObjectCandidate]:
        if det.score < self.p.score_threshold:
            return None
        pts = self.localize(mask, depth, T_world_cam, floor_y, precise)
        if pts is None:
            return None
        pos = np.median(pts, axis=0)
        xy = np.stack([pts[:, 0], -pts[:, 2]], axis=1)
        h = pts[:, 1] - (floor_y if floor_y is not None else 0.0)
        c = self._merge(det, pts, pos, xy, h, step)
        if self.p.fusion:
            best, n = self._frame.get(c.cid, (0.0, 0))
            # weight on the scale of the stored sample, so a detection and a miss of the same view weigh the same
            self._frame[c.cid] = (max(best, det.score), max(n, min(len(pts), MAX_FUSION_POINTS)))
            c.points_world = self._keep(c.points_world, pts)
        return c

    def _keep(self, old: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        both = new if old is None else np.concatenate([old, new])
        if len(both) <= MAX_FUSION_POINTS:
            return both
        return both[self._rng.choice(len(both), MAX_FUSION_POINTS, replace=False)]

    def _merge(self, det: Detection, pts: np.ndarray, pos: np.ndarray, xy: np.ndarray, h: np.ndarray,
               step: int) -> ObjectCandidate:
        for c in self.candidates:
            if np.hypot(*(c.xy - np.array([pos[0], -pos[2]]))) <= self.p.merge_radius_m:
                n = c.n_detections
                c.position = (c.position * n + pos) / (n + 1)
                if len(c.points_xy) < self.p.max_points:
                    keep, keep_h = c.points_xy, c.points_h
                else:
                    idx = self._rng.choice(len(c.points_xy), self.p.max_points // 2, replace=False)
                    keep = c.points_xy[idx]
                    keep_h = c.points_h[idx] if c.points_h is not None else None
                c.points_xy = np.concatenate([keep, xy[: self.p.max_points // 2]])
                c.points_h = np.concatenate([keep_h, h[: self.p.max_points // 2]]) \
                    if keep_h is not None and len(keep_h) == len(keep) else None
                c.best_score = max(c.best_score, det.score)
                c.scores[det.query_index] = max(c.scores.get(det.query_index, 0.0), det.score)
                c.n_detections += 1
                if step > c.last_step or c.last_box is None or _area(det.box_xyxy) > _area(c.last_box):
                    c.last_box = tuple(float(v) for v in det.box_xyxy)   # largest box of the latest frame
                c.last_step = step
                return c
        c = ObjectCandidate(len(self.candidates), det.query_index, pos, xy, det.score, 1, step, step,
                            scores={det.query_index: det.score}, last_box=tuple(float(v) for v in det.box_xyxy),
                            points_h=np.asarray(h, dtype=float))
        self.candidates.append(c)
        return c

    def end_frame(self, depth: np.ndarray, T_world_cam: np.ndarray) -> None:
        """Fuse this frame's evidence into every candidate (D-060); a no-op without fusion."""
        if not self.p.fusion:
            return
        for c in self.candidates:
            if c.cid in self._frame:
                score, n = self._frame[c.cid]
            else:
                n = self.visible_points(c, depth, T_world_cam)
                if n < self.p.min_visible_points:
                    continue
                score = 0.0
                c.misses += 1
            c.conf = (c.volume * c.conf + n * score) / (c.volume + n)
            c.volume += n
        self._frame = {}

    def visible_points(self, c: ObjectCandidate, depth: np.ndarray, T_world_cam: np.ndarray) -> int:
        """How many of the candidate's stored points the camera sees now: in the image, within depth range and
        not behind a nearer surface."""
        if c.points_world is None or not len(c.points_world):
            return 0
        R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
        p_cam = (c.points_world - t) @ R                    # world -> camera (R orthonormal)
        z = -p_cam[:, 2]
        front = (z > 0.05) & (z <= self.p.max_depth_m)
        if not front.any():
            return 0
        u, v = self.K.project(p_cam[front])
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (ui >= 0) & (ui < self.K.width) & (vi >= 0) & (vi < self.K.height)
        if not inside.any():
            return 0
        seen = depth[vi[inside], ui[inside]]
        visible = (seen > 0) & (seen >= z[front][inside] - self.p.occlusion_tol_m)
        return int(visible.sum())

    def is_reliable(self, c: ObjectCandidate) -> bool:
        return not self.p.fusion or c.conf >= self.p.reliable_conf

    def best(self, exclude_unreachable: bool = True, weights: Optional[Sequence[float]] = None,
             include_suspected: bool = False) -> Optional[ObjectCandidate]:
        """Most promising live candidate; rejected ones never. Without fusion: highest (hypothesis-weighted)
        detector score. With fusion (D-060): reliable candidates only (all, with ``include_suspected``), ranked by
        fused confidence times the hypothesis-weighted match."""
        live = [c for c in self.candidates
                if not c.rejected and not c.lost and not (exclude_unreachable and c.unreachable)]
        if not self.p.fusion:
            return max(live, key=lambda c: (c.priority(weights), c.n_detections)) if live else None
        if not include_suspected:
            live = [c for c in live if self.is_reliable(c)]
        if not live:
            return None

        def rank(c: ObjectCandidate) -> float:
            if weights is None:
                return c.conf
            m = c.match_vector(len(weights))
            return c.conf * float(np.dot(np.asarray(weights, dtype=float), m))

        return max(live, key=lambda c: (rank(c), c.n_detections))


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_mask(det: Detection, shape: Sequence[int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = det.box_xyxy
    m = np.zeros((h, w), dtype=bool)
    m[max(0, int(y1)) : min(h, int(np.ceil(y2))), max(0, int(x1)) : min(w, int(np.ceil(x2)))] = True
    return m
