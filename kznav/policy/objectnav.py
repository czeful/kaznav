"""ObjectNav decision policy: look around, explore frontiers, approach a detected candidate, stop.

Embodiment-agnostic: every decision step it updates the robot's own occupancy map from the
observation, runs the detector on the text queries, localises detections into object candidates
and returns a ``Decision`` (a path in the map frame, a spin, or stop). Turning a decision into
wheel commands or discrete actions is the executor's job (kznav.run.episode).

Stage-4 baseline (no language model): queries are the English category name(s); the robot stops
when it is within ``stop_distance_m`` of a candidate detected in the last ``recent_steps`` steps.
Stage 5 inserts the VLM verification before the stop and the Bayesian hypothesis weights.
Frontier choice is nearest-by-path-cost; the value map (VLFM-style) plugs in here later.
The policy never sees the navmesh or ground-truth semantics (except through an explicitly
configured oracle detector, which is an ablation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..perception.base import Detection, Detector, Segmenter
from ..perception.camera import Intrinsics
from ..perception.frontiers import Frontier, find_frontiers
from ..perception.objects import DetectionParams, ObjectCandidate, ObjectMemory, box_mask
from ..perception.occupancy import MapParams, OccupancyMap
from .planner import GridPlanner, PlannerParams, PlanResult, simplify_path


@dataclass
class Decision:
    kind: str                                   # spin | explore | approach | stop | no_frontiers | no_plan
    path_xy: List[Tuple[float, float]] = field(default_factory=list)
    target_xy: Optional[Tuple[float, float]] = None
    reason: str = ""
    candidate_id: Optional[int] = None
    n_frontiers: int = 0


class ObjectNavPolicy:
    def __init__(self, cfg: Mapping[str, Any], detector: Detector, segmenter: Segmenter, queries: Sequence[str],
                 intrinsics: Intrinsics, seed: int = 0):
        emb, pol, per = cfg["embodiment"], cfg["policy"], cfg["perception"]
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

    def reset(self, pose: Tuple[float, float, float]) -> None:
        self.map = OccupancyMap(self.map_params, self.K, (pose[0], pose[1]))
        self.memory.reset()
        self.step_idx = 0
        self.spin_left = self.spin_steps_total
        self.no_plan = 0

    # ------------------------------------------------------------------ main step
    def step(self, obs: Any, aux: Optional[Mapping[str, Any]] = None) -> Decision:
        assert self.map is not None, "call reset() first"
        pose = obs.pose
        self.map.update(obs.depth, obs.camera_transform, float(obs.position[1]))
        self.map.mark_free_disk(pose[:2], self.radius)
        self._perceive(obs, aux)
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
        cand = self.memory.best()
        if cand is not None:
            dist = cand.nearest_distance(pose[:2])
            if dist <= self.stop_distance and self.step_idx - cand.last_step <= self.recent_steps:
                return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                reason=f"candidate {cand.cid} at {dist:.2f} m (score {cand.best_score:.2f})")
        # Plan with the safety margin first; if nothing useful is reachable, retry tight (bare radius).
        for margin, tag in ((None, ""), (self.tight_margin, " [tight]")):
            plan = self.planner.plan(self.map, pose[:2], extra_cells=self._candidate_cells(), margin_m=margin)
            self.last_plan = plan
            if cand is not None:
                path = self._path_to_candidate(plan, cand)
                if path:
                    self.no_plan = 0
                    return Decision("approach", path, tuple(cand.xy),
                                    f"candidate {cand.cid}, {cand.nearest_distance(pose[:2]):.2f} m away{tag}", cand.cid)
            fr = find_frontiers(self.map.free, self.map.unknown, plan.blocked, plan.start, self.min_frontier)                 if plan.ok else []
            fr = [f for f in fr if math.isfinite(plan.cost_to(f.cell))]
            self.last_frontiers = fr
            if fr:
                self.no_plan = 0
                best = min(fr, key=lambda f: plan.cost_to(f.cell))
                path = self._cells_to_xy(simplify_path(plan.path_cells(best.cell), plan.blocked))
                x, y = self.map.to_xy(np.array([best.cell[0]]), np.array([best.cell[1]]))
                return Decision("explore", path, (float(x[0]), float(y[0])), f"frontier of {best.size} cells{tag}",
                                n_frontiers=len(fr))
        self.no_plan += 1
        if self.no_plan >= self.max_no_plan:
            return Decision("no_frontiers", reason="no reachable frontier or candidate, even with tight planning")
        return Decision("no_plan", reason="nothing reachable this step; looking around")

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

    def _path_to_candidate(self, plan: PlanResult, cand: ObjectCandidate) -> List[Tuple[float, float]]:
        if not plan.ok:
            return []
        m = self.map
        pts = cand.points_xy
        lo = pts.min(axis=0) - self.approach_radius
        hi = pts.max(axis=0) + self.approach_radius
        r0, c0 = m.to_cell(np.array([lo[0]]), np.array([lo[1]]))
        r1, c1 = m.to_cell(np.array([hi[0]]), np.array([hi[1]]))
        r0, r1 = max(0, int(r0[0])), min(m.n - 1, int(r1[0]))
        c0, c1 = max(0, int(c0[0])), min(m.n - 1, int(c1[0]))
        if r1 < r0 or c1 < c0:
            return []
        rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        xs, ys = m.to_xy(rr, cc)
        d = np.min(np.hypot(xs[..., None] - pts[None, None, :, 0], ys[..., None] - pts[None, None, :, 1]), axis=-1)
        ok = (d <= self.approach_radius) & ~plan.blocked[r0 : r1 + 1, c0 : c1 + 1]
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
