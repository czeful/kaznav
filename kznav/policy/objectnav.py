"""ObjectNav decision policy: look around, explore frontiers, approach a detected candidate, stop.

Embodiment-agnostic: every decision step it updates the robot's own occupancy map from the
observation, runs the detector on the text queries, localises detections into object candidates
and returns a ``Decision`` (a path in the map frame, a spin, or stop). Turning a decision into
wheel commands or discrete actions is the executor's job (kznav.run.episode).

Queries are the English category name (baseline) or the language model's visual hypotheses.
Without a verifier the robot stops when it is within ``stop_distance_m`` of a candidate detected
in the last ``recent_steps`` steps. With a verifier (kznav.policy.verification) the stop needs the
VLM's confirmation on a crop of the candidate in the current frame; a candidate it rejects is
never approached again, and the verifier's hypothesis weights re-weight the value map and the
candidate ranking. Frontier choice: nearest by path cost, or VLFM-style value map.
The policy never sees the navmesh or ground-truth semantics (except through an explicitly
configured oracle detector, which is an ablation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..perception.base import Detection, Detector, Segmenter, ValueEncoder
from ..perception.camera import Intrinsics
from ..perception.frontiers import Frontier, find_frontiers
from ..perception.objects import DetectionParams, ObjectCandidate, ObjectMemory, box_mask
from ..perception.occupancy import MapParams, OccupancyMap
from ..perception.value_map import ValueMap
from .planner import GridPlanner, PlannerParams, PlanResult, simplify_path


@dataclass
class Decision:
    kind: str                                   # spin | explore | approach | face | stop | no_frontiers | no_plan
    path_xy: List[Tuple[float, float]] = field(default_factory=list)
    target_xy: Optional[Tuple[float, float]] = None
    reason: str = ""
    candidate_id: Optional[int] = None
    n_frontiers: int = 0


class ObjectNavPolicy:
    def __init__(self, cfg: Mapping[str, Any], detector: Detector, segmenter: Segmenter, queries: Sequence[str],
                 intrinsics: Intrinsics, seed: int = 0, value_encoder: Optional[ValueEncoder] = None,
                 verifier: Any = None):
        emb, pol, per = cfg["embodiment"], cfg["policy"], cfg["perception"]
        self.value_encoder = value_encoder
        self.verifier = verifier                            # kznav.policy.verification.Verifier or None
        self.last_verify: Any = None
        self._rgb: Optional[np.ndarray] = None
        fs = pol["frontier_selection"]
        self.selection = str(fs["mode"]) if value_encoder is not None else "nearest"
        self.value_radius_cells = int(round(float(fs["value_radius_m"]) / float(per["map"]["resolution_m"])))
        self.distance_weight = float(fs["distance_weight_per_m"])
        self.hysteresis = float(fs["hysteresis"])
        self.hfov = float(emb["sensors"]["rgbd"]["hfov_deg"])
        self.query_weights: Optional[List[float]] = None   # stage 5: hypothesis weights from the belief
        self.value_map: Optional[ValueMap] = None
        self.last_value: Optional[float] = None
        self._target: Optional[Tuple[float, float]] = None
        self.cfg = cfg
        self.K = intrinsics
        self.detector = detector
        self.segmenter = segmenter
        self.queries = list(queries)
        self.map_params = MapParams.from_config(per["map"], emb)
        self.memory = ObjectMemory(DetectionParams.from_config(per["detection"], float(per["map"]["max_range_m"])),
                                   intrinsics, seed)
        self.radius = float(emb["body"]["radius_m"])
        self.planner = GridPlanner(PlannerParams.from_config(pol["planner"]), self.radius)
        self.min_frontier = int(per["frontiers"]["min_size_cells"])
        ap = pol["approach"]
        self.stop_distance = float(ap["stop_distance_m"])
        self.recent_steps = int(ap["recent_steps"])
        self.approach_radius = float(ap["approach_radius_m"])
        self.close_radius = float(ap["close_radius_m"])
        self.max_unseen_in_range = int(ap["max_unseen_in_range_steps"])
        ex = pol["exploration"]
        self.max_no_plan = int(ex["max_consecutive_no_plan"])
        self.tight_margin = float(pol["planner"]["tight_margin_m"])
        self.spin_steps_total = spin_steps(cfg)
        self.map: Optional[OccupancyMap] = None
        self.step_idx = 0
        self.spin_left = 0
        self.no_plan = 0
        self.last_detections: List[Detection] = []
        self.last_frontiers: List[Frontier] = []
        self.last_plan: Optional[PlanResult] = None

    def reset(self, pose: Tuple[float, float, float], seed: Optional[int] = None) -> None:
        """Start an episode. ``seed`` makes the episode independent of which episodes ran before it."""
        self.map = OccupancyMap(self.map_params, self.K, (pose[0], pose[1]))
        self.value_map = ValueMap(self.map.log_odds.shape, self.map.res, self.map.x0, self.map.y0, self.hfov) \
            if self.value_encoder is not None else None
        self._target = None
        self.last_value = None
        self.last_verify = None
        self._rgb = None
        self.memory.reset(seed)
        self.step_idx = 0
        self.spin_left = self.spin_steps_total
        self.no_plan = 0

    # ------------------------------------------------------------------ main step
    def step(self, obs: Any, aux: Optional[Mapping[str, Any]] = None) -> Decision:
        assert self.map is not None, "call reset() first"
        pose = obs.pose
        seen = self.map.update(obs.depth, obs.camera_transform, float(obs.position[1]))
        self.map.mark_free_disk(pose[:2], self.radius)
        if self.value_map is not None and self.queries:
            sc = self.value_encoder.score(obs.rgb, self.queries)
            w = np.asarray(self.query_weights if self.query_weights is not None else [1.0 / len(sc)] * len(sc))
            self.last_value = float(np.dot(w, sc))
            cam = obs.camera_transform
            fwd = cam[:3, :3] @ np.array([0.0, 0.0, -1.0])
            self.value_map.update(seen, (float(cam[0, 3]), float(-cam[2, 3])), math.atan2(-fwd[2], fwd[0]),
                                  self.last_value)
        self._perceive(obs, aux)
        self._rgb = obs.rgb
        d = self._decide(pose)
        self.step_idx += 1
        return d

    def _perceive(self, obs: Any, aux: Optional[Mapping[str, Any]]) -> None:
        dets = self.detector.detect(obs.rgb, self.queries, aux)
        self.last_detections = dets
        for det in dets:
            mask = det.mask if det.mask is not None else self.segmenter.segment(obs.rgb, det.box_xyxy)
            if mask is None:
                mask = box_mask(det, obs.rgb.shape)
            self.memory.add(det, mask, obs.depth, obs.camera_transform, self.step_idx)

    def _decide(self, pose: Tuple[float, float, float]) -> Decision:
        if self.spin_left > 0:
            self.spin_left -= 1
            return Decision("spin", reason="initial look-around")
        cand = self.memory.best(weights=self.query_weights)
        close_view = False
        while cand is not None:
            dist = cand.nearest_distance(pose[:2])
            if dist > self.stop_distance:
                break
            recent = self.step_idx - cand.last_step <= self.recent_steps
            if self.verifier is None and recent:
                return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                reason=f"candidate {cand.cid} at {dist:.2f} m (score {cand.best_score:.2f})")
            if cand.last_step != self.step_idx:
                # In range but not in this frame: turn to it; if it does not show up again, it was a
                # transient detection or is out of the camera's view from here -> give it up.
                cand.unseen_in_range += 1
                if cand.unseen_in_range <= self.max_unseen_in_range:
                    return Decision("face", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                    reason=f"candidate {cand.cid} in range, not in view: turning to it")
                cand.lost = True
                cand = self.memory.best(weights=self.query_weights)
                continue
            cand.unseen_in_range = 0
            ev = self.verifier.check(cand, self._rgb, self.step_idx, len(self.queries))
            self.last_verify = ev
            if self.verifier.belief is not None:
                self.query_weights = self.verifier.weights
            if ev.decision == "stop":
                return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                reason=f"verified candidate {cand.cid}: P={ev.p_candidate:.2f} at {dist:.2f} m")
            if ev.decision == "wait":
                close_view = True            # approach closer for a bigger view
                break
            cand = self.memory.best(weights=self.query_weights)       # rejected: next candidate
        # Plan with the safety margin first; if nothing useful is reachable, retry tight (bare radius).
        for margin, tag in ((None, ""), (self.tight_margin, " [tight]")):
            plan = self.planner.plan(self.map, pose[:2], extra_cells=self._candidate_cells(), margin_m=margin)
            self.last_plan = plan
            if cand is not None:
                path = self._path_to_candidate(plan, cand, self.close_radius if close_view else self.approach_radius)
                if path:
                    self.no_plan = 0
                    return Decision("approach", path, tuple(cand.xy),
                                    f"candidate {cand.cid}, {cand.nearest_distance(pose[:2]):.2f} m away{tag}", cand.cid)
            fr = find_frontiers(self.map.free, self.map.unknown, plan.blocked, plan.start, self.min_frontier) \
                if plan.ok else []
            fr = [f for f in fr if math.isfinite(plan.cost_to(f.cell))]
            self.last_frontiers = fr
            if fr:
                self.no_plan = 0
                best = self._choose_frontier(fr, plan)
                path = self._cells_to_xy(simplify_path(plan.path_cells(best.cell), plan.blocked))
                x, y = self.map.to_xy(np.array([best.cell[0]]), np.array([best.cell[1]]))
                return Decision("explore", path, (float(x[0]), float(y[0])), f"frontier of {best.size} cells{tag}",
                                n_frontiers=len(fr))
        self.no_plan += 1
        if self.no_plan >= self.max_no_plan:
            return Decision("no_frontiers", reason="no reachable frontier or candidate, even with tight planning")
        return Decision("no_plan", reason="nothing reachable this step; looking around")

    def _choose_frontier(self, fr: List[Frontier], plan: PlanResult) -> Frontier:
        """Nearest frontier, or (value mode) the most promising one: value - w * path cost, with hysteresis."""
        if self.selection == "nearest" or self.value_map is None:
            return min(fr, key=lambda f: plan.cost_to(f.cell))

        def score(f: Frontier) -> float:
            return self.value_map.value_at(f.cell, self.value_radius_cells) - self.distance_weight * plan.cost_to(f.cell)

        best = max(fr, key=score)
        if self._target is not None:
            # keep heading to the previous target region unless another frontier is clearly better
            prev = min(fr, key=lambda f: (f.centroid[0] - self._target[0]) ** 2 + (f.centroid[1] - self._target[1]) ** 2)
            d2 = (prev.centroid[0] - self._target[0]) ** 2 + (prev.centroid[1] - self._target[1]) ** 2
            if d2 * self.map.res ** 2 <= 0.5 ** 2 and score(prev) >= score(best) - self.hysteresis:
                best = prev
        self._target = best.centroid
        return best

    def on_blocked(self, pose: Tuple[float, float, float], ahead_m: float, radius_m: float) -> None:
        """Proprioceptive bump: mark a virtual obstacle just ahead of the robot on its own map."""
        x = pose[0] + ahead_m * math.cos(pose[2])
        y = pose[1] + ahead_m * math.sin(pose[2])
        self.map.mark_obstacle_disk((x, y), radius_m)

    # ------------------------------------------------------------------ helpers
    def _candidate_cells(self) -> List[Tuple[int, int]]:
        out = []
        for c in self.memory.candidates:
            r, col = self.map.to_cell(np.array([c.xy[0]]), np.array([c.xy[1]]))
            out.append((int(r[0]), int(col[0])))
        return out

    def _path_to_candidate(self, plan: PlanResult, cand: ObjectCandidate, radius: float) -> List[Tuple[float, float]]:
        if not plan.ok:
            return []
        m = self.map
        pts = cand.points_xy
        lo = pts.min(axis=0) - radius
        hi = pts.max(axis=0) + radius
        r0, c0 = m.to_cell(np.array([lo[0]]), np.array([lo[1]]))
        r1, c1 = m.to_cell(np.array([hi[0]]), np.array([hi[1]]))
        r0, r1 = max(0, int(r0[0])), min(m.n - 1, int(r1[0]))
        c0, c1 = max(0, int(c0[0])), min(m.n - 1, int(c1[0]))
        if r1 < r0 or c1 < c0:
            return []
        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        xs, ys = m.to_xy(rr, cc)
        d = np.min(np.hypot(xs[..., None] - pts[None, None, :, 0], ys[..., None] - pts[None, None, :, 1]), axis=-1)
        ok = (d <= radius) & ~plan.blocked[r0 : r1 + 1, c0 : c1 + 1]
        cost = np.where(ok, plan.dist[r0 : r1 + 1, c0 : c1 + 1], np.inf)
        if not np.isfinite(cost).any():
            return []
        i = np.unravel_index(int(np.argmin(cost)), cost.shape)
        goal = (int(rr[i]), int(cc[i]))
        return self._cells_to_xy(simplify_path(plan.path_cells(goal), plan.blocked))

    def _cells_to_xy(self, cells: Sequence[Tuple[int, int]]) -> List[Tuple[float, float]]:
        if not cells:
            return []
        xs, ys = self.map.to_xy(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
        return [(float(x), float(y)) for x, y in zip(xs, ys)]


def spin_steps(cfg: Mapping[str, Any]) -> int:
    """Decision steps needed for the initial look-around of this embodiment."""
    emb, ex = cfg["embodiment"], cfg["policy"]["exploration"]
    angle = math.radians(float(ex["initial_spin_deg"]))
    if angle <= 0:
        return 0
    if emb["kind"] == "wheeled":
        per_step = float(ex["spin_ang_vel_radps"]) * float(cfg["eval"]["episode_budget"]["decision_period_s"])
    else:
        per_step = math.radians(float(emb["drive"]["turn_angle_deg"]))
    return int(math.ceil(angle / per_step))
