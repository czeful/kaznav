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
from .verification import crop_box


@dataclass
class Decision:
    """The policy's decision for one step: what it does (spin, explore, approach, face, stop, ...) and why."""
    kind: str                                   # spin | explore | approach | face | stop | no_frontiers | no_plan
    path_xy: List[Tuple[float, float]] = field(default_factory=list)
    target_xy: Optional[Tuple[float, float]] = None
    reason: str = ""
    candidate_id: Optional[int] = None
    n_frontiers: int = 0


class ObjectNavPolicy:
    """Zero-shot ObjectNav policy: mapping, detection, value-guided exploration, approach, verification, stop.

    Args:
        cfg: The full run config (embodiment, policy and perception blocks are read).
        detector: Open-vocabulary detector.
        segmenter: Segmenter for detection masks.
        queries: The goal's English hypotheses (detector queries).
        intrinsics: Camera intrinsics.
        seed: Seed of the per-episode random generators.
        value_encoder: Encoder of the semantic value map; None selects the nearest frontier instead.
        verifier: Verify-before-stop component, or None (stop on detection alone).
    """
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
        self.room_weight = float(fs.get("room_weight", 0.0))      # D-072: share of the room prior in the value
        self.rooms: List[str] = []                                 # the language model's rooms for the target
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
        # D-065: with a verifier, suspected candidates are approached too (the VLM rejects false ones on the spot)
        self.approach_suspected = bool(pol["verification"].get("approach_suspected", False))
        # D-070 "last metre"; read with defaults, so configs without the keys keep their behaviour (and hash)
        self.stop_mode = str(ap.get("stop_mode", "distance"))
        if self.stop_mode not in ("distance", "visible"):
            raise ValueError(f"policy.approach.stop_mode must be distance or visible, not {self.stop_mode!r}")
        self.visible_max = float(ap.get("visible_stop_max_m", 0.9))
        self.visible_close = float(ap.get("visible_close_m", 0.6))
        self.visible_margin = math.radians(float(ap.get("visible_margin_deg", 1.0)))
        self.visible_fraction = float(ap.get("visible_min_fraction", 0.3))
        self.ring_slack = float(ap.get("visible_target_slack_m", 0.15))
        vv = pol["verification"]
        self.commit = bool(vv.get("commit_approach", False))
        self.verify_range = float(vv.get("verify_range_m", 0.0))
        self.whole_margin = float(vv.get("whole_box_margin_px", 4.0))
        self.second_look_deg = float(vv.get("second_look_deg", 0.0))          # D-071; 0 = off
        self.second_look_max = int(vv.get("second_look_max_steps", 40))
        self.recheck_suspected = bool(vv.get("recheck_suspected_crops", False))   # D-072
        rgbd = emb["sensors"]["rgbd"]
        mount = rgbd["mount"]
        self.cam_h = float(mount["height_m"])
        self.cam_fwd = float(mount.get("forward_m", 0.0))
        self.cam_pitch = math.radians(float(mount.get("pitch_deg", 0.0)))
        res_h, res_w = (float(v) for v in rgbd["resolution_hw"])
        self.half_vfov = math.atan(math.tan(math.radians(self.hfov) / 2) * res_h / res_w)
        self.committed: Optional[int] = None                # candidate the robot is driving to (commit_approach)
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
        self.exhausted = False                             # no frontier left: suspected candidates count (D-060)

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
        self.exhausted = False
        self.committed = None
        self.step_idx = 0
        self.spin_left = self.spin_steps_total
        self.no_plan = 0

    # ------------------------------------------------------------------ main step
    def step(self, obs: Any, aux: Optional[Mapping[str, Any]] = None) -> Decision:
        """Update the maps from one observation and decide what to do next.

        Args:
            obs: The observation (RGB, depth, pose, camera transform).
            aux: Ground-truth side channels for oracle detectors only.

        Returns:
            The decision, with the path to follow when the robot moves.
        """
        assert self.map is not None, "call reset() first"
        pose = obs.pose
        seen = self.map.update(obs.depth, obs.camera_transform, float(obs.position[1]))
        self.map.mark_free_disk(pose[:2], self.radius)
        if self.value_map is not None and self.queries:
            n = len(self.queries)
            rooms = list(self.rooms) if self.room_weight > 0 else []
            sc = self.value_encoder.score(obs.rgb, list(self.queries) + rooms)
            w = np.asarray(self.query_weights if self.query_weights is not None else [1.0 / n] * n)
            self.last_value = float(np.dot(w, sc[:n]))
            if rooms:
                # D-072: VLFM-style room prior - the frame's similarity to the rooms the model expects the target in
                self.last_value = (1 - self.room_weight) * self.last_value + self.room_weight * float(np.mean(sc[n:]))
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
        try:
            self._localize(dets, obs)
        finally:
            self.memory.end_frame(obs.depth, obs.camera_transform)

    def _localize(self, dets: List[Detection], obs: Any) -> None:
        for det in dets:
            mask = det.mask if det.mask is not None else self.segmenter.segment(obs.rgb, det.box_xyxy)
            precise = det.mask is None and mask is not None and self.segmenter.precise
            if mask is None:
                mask = box_mask(det, obs.rgb.shape)
            c = self.memory.add(det, mask, obs.depth, obs.camera_transform, self.step_idx,
                                floor_y=float(obs.position[1]), precise=precise)
            if c is not None and self.recheck_suspected and self.verifier is not None and det.score > c.crop_score:
                c.crop = crop_box(obs.rgb, det.box_xyxy, self.verifier.p.crop_padding, self.verifier.p.min_crop_px)[0].copy()
                c.crop_score = det.score

    def _decide(self, pose: Tuple[float, float, float]) -> Decision:
        if self.spin_left > 0:
            self.spin_left -= 1
            return Decision("spin", reason="initial look-around")
        cand = self._best()
        close_view = False
        look: Optional[ObjectCandidate] = None
        while cand is not None:
            dist = cand.nearest_distance(pose[:2])
            if not self._in_stop_range(cand, dist, pose):
                if self._ask_from_afar(cand, dist):
                    # D-070: the whole object is in view from up to verify_range_m; a "yes" confirms it (the robot
                    # then drives into the stop range and stops without asking again), a "no" rejects it
                    ev = self._verify(cand, dist, self._bearing(cand, pose))
                    if ev.decision == "stop":
                        cand.confirmed = True
                    elif ev.decision == "reject":
                        cand = self._best()
                        continue
                break
            recent = self.step_idx - cand.last_step <= self.recent_steps
            if self.verifier is None and recent:
                return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                reason=f"candidate {cand.cid} at {dist:.2f} m (score {cand.best_score:.2f}"
                                       + (f", fused {cand.conf:.2f}, misses {cand.misses})" if self.memory.fusion else ")"))
            if cand.last_step != self.step_idx:
                # In range but not in this frame: turn to it; if it does not show up again, it was a
                # transient detection or is out of the camera's view from here -> give it up.
                cand.unseen_in_range += 1
                if cand.unseen_in_range <= self.max_unseen_in_range:
                    return Decision("face", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                    reason=f"candidate {cand.cid} in range, not in view: turning to it")
                if cand.confirmed:
                    return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                    reason=f"candidate {cand.cid} accepted from afar, in stop range at {dist:.2f} m")
                cand.lost = True
                cand = self._best()
                continue
            cand.unseen_in_range = 0
            if cand.confirmed:
                return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                reason=f"candidate {cand.cid} accepted from afar, in view at {dist:.2f} m")
            bearing = self._bearing(cand, pose)
            if self._second_look_due(cand, bearing):
                # D-071: undecided and no bigger view to come from here - look again from elsewhere around the object
                cand.second_look_steps += 1
                if cand.second_look_steps <= self.second_look_max:
                    look = cand
                    break
                cand.rejected = True
                cand = self._best()
                continue
            ev = self._verify(cand, dist, bearing)
            if ev.decision == "stop":
                return Decision("stop", target_xy=tuple(cand.xy), candidate_id=cand.cid,
                                reason=f"verified candidate {cand.cid}: P={ev.p_candidate:.2f} at {dist:.2f} m")
            if ev.decision == "wait":
                close_view = True            # approach closer for a bigger view
                break
            cand = self._best()       # rejected: next candidate
        # Plan with the safety margin first; if nothing useful is reachable, retry tight (bare radius).
        for margin, tag in ((None, ""), (self.tight_margin, " [tight]")):
            plan = self.planner.plan(self.map, pose[:2], extra_cells=self._candidate_cells(), margin_m=margin)
            self.last_plan = plan
            if look is not None:
                path = self._path_to_candidate(plan, look, self._range_radius(), ring=self.stop_mode == "visible",
                                               away_from=self._bearing(look, pose), min_deg=self.second_look_deg + 10)
                if path:
                    self.no_plan = 0
                    return Decision("approach", path, tuple(look.xy), f"second look at candidate {look.cid}{tag}",
                                    look.cid)
                look.no_new_view = True       # nowhere to look from: the verifier decides as before
                close_view = True
            if cand is not None:
                if self.stop_mode == "visible":
                    # target cells well inside the stop range (the executor arrives within its tolerance); already
                    # at such a cell but not in range (a view blocked from there): the plain approach, as before
                    path = self._path_to_candidate(plan, cand, (self.visible_close if close_view else self.visible_max)
                                                   - self.ring_slack, ring=True)
                    if _length(path) < 2 * self.ring_slack:
                        path = self._path_to_candidate(plan, cand, self.approach_radius)
                else:
                    path = self._path_to_candidate(plan, cand, self.close_radius if close_view else self.approach_radius)
                if path:
                    self.no_plan = 0
                    if self.commit and self.verifier is not None:
                        self.committed = cand.cid
                    return Decision("approach", path, tuple(cand.xy),
                                    f"candidate {cand.cid}, {cand.nearest_distance(pose[:2]):.2f} m away{tag}", cand.cid)
            fr = find_frontiers(self.map.free, self.map.unknown, plan.blocked, plan.start, self.min_frontier) \
                if plan.ok else []
            fr = [f for f in fr if math.isfinite(plan.cost_to(f.cell))]
            self.last_frontiers = fr
            if fr:
                self.exhausted = False
                self.no_plan = 0
                best = self._choose_frontier(fr, plan)
                path = self._cells_to_xy(simplify_path(plan.path_cells(best.cell), plan.blocked))
                x, y = self.map.to_xy(np.array([best.cell[0]]), np.array([best.cell[1]]))
                return Decision("explore", path, (float(x[0]), float(y[0])), f"frontier of {best.size} cells{tag}",
                                n_frontiers=len(fr))
        if self.memory.fusion and not self.exhausted:
            # Exploration has nothing left: suspected candidates (fused confidence below the bar) become eligible.
            self.exhausted = True
            if self.recheck_suspected and self.verifier is not None:
                self._recheck_suspected()
            if self._best() is not None:
                return Decision("no_plan", reason="exploration exhausted: considering suspected candidates")
        self.no_plan += 1
        if self.no_plan >= self.max_no_plan:
            return Decision("no_frontiers", reason="no reachable frontier or candidate, even with tight planning")
        return Decision("no_plan", reason="nothing reachable this step; looking around")

    def _best(self) -> Optional[ObjectCandidate]:
        if self.committed is not None:
            # D-070 commit_approach: the candidate the robot drives to stays its target until the verifier decides,
            # it is lost from close by, or it is unreachable - a missed frame no longer makes the robot leave it
            c = next((c for c in self.memory.candidates if c.cid == self.committed), None)
            if c is not None and not (c.rejected or c.lost or c.unreachable):
                return c
            self.committed = None
        suspected = self.exhausted or (self.approach_suspected and self.verifier is not None)
        return self.memory.best(weights=self.query_weights, include_suspected=suspected)

    def _recheck_suspected(self) -> None:
        """D-072: before driving to suspected candidates, ask about their stored crops; a confident "no" drops one
        (the failed D-065 fix drove to every suspected candidate and wasted the budget)."""
        for c in self.memory.candidates:
            if c.rejected or c.lost or c.crop is None or self.memory.is_reliable(c) or self.verifier.calls(c.cid):
                continue
            self.last_verify = self.verifier.check_crop(c, c.crop, self.step_idx, len(self.queries))
            if self.verifier.belief is not None:
                self.query_weights = self.verifier.weights

    def _verify(self, cand: ObjectCandidate, dist: float, bearing: Optional[float] = None) -> Any:
        ev = self.verifier.check(cand, self._rgb, self.step_idx, len(self.queries), dist_m=dist, bearing=bearing)
        self.last_verify = ev
        if self.verifier.belief is not None:
            self.query_weights = self.verifier.weights
        return ev

    @staticmethod
    def _bearing(cand: ObjectCandidate, pose: Sequence[float]) -> float:
        """Direction from the object to the robot (map frame): where around the object the robot looks from."""
        return math.atan2(pose[1] - cand.xy[1], pose[0] - cand.xy[0])

    def _range_radius(self) -> float:
        return self.visible_max - self.ring_slack if self.stop_mode == "visible" else self.approach_radius

    def _second_look_due(self, cand: ObjectCandidate, bearing: float) -> bool:
        v = self.verifier
        if self.second_look_deg <= 0 or v is None or cand.no_new_view:
            return False
        n = v.calls(cand.cid)
        return 1 <= n < v.p.max_calls and v.undecided(cand.cid) and not v.can_call(cand, bearing)

    def _in_stop_range(self, cand: ObjectCandidate, dist: float, pose: Tuple[float, float, float]) -> bool:
        """distance: within stop_distance_m. visible (D-070): also farther, up to visible_stop_max_m, where enough of
        the object's points are in the camera's vertical field of view (a TV on a stand, out of view from 0.6 m for a
        camera 0.15 m high, is in view from 0.85 m). Within stop_distance_m it is in range as before, so an object
        out of view from every cell of the ring (a TV high on the wall) cannot hold the robot in an endless approach."""
        if dist <= self.stop_distance or self.stop_mode == "distance":
            return dist <= self.stop_distance
        return dist <= self.visible_max and \
            float(self._visible_fraction(cand, np.array(pose[0]), np.array(pose[1]))) >= self.visible_fraction

    def _ask_from_afar(self, cand: ObjectCandidate, dist: float) -> bool:
        """D-070: ask the verifier before the stop range when the whole object is in this frame (its box clear of the
        image border) and a call would be made (first view, or a view grown enough)."""
        if self.verifier is None or self.verify_range <= 0 or cand.confirmed or dist > self.verify_range:
            return False
        if cand.last_step != self.step_idx or cand.last_box is None:
            return False
        x1, y1, x2, y2 = cand.last_box
        m = self.whole_margin
        whole = x1 >= m and y1 >= m and x2 <= self.K.width - m and y2 <= self.K.height - m
        return whole and self.verifier.can_call(cand)

    def _visible_fraction(self, cand: ObjectCandidate, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Share of the candidate's points inside the camera's vertical field of view, for a robot at (xs, ys) facing
        the object (camera ``forward_m`` ahead of the centre, ``height_m`` above the floor, pitched by ``pitch_deg``)."""
        pts, h = cand.points_xy, cand.points_h
        if h is None or len(h) != len(pts) or not len(pts):
            return np.ones(np.shape(xs))
        d = np.hypot(xs[..., None] - pts[:, 0], ys[..., None] - pts[:, 1]) - self.cam_fwd
        elev = np.arctan2(h - self.cam_h, np.maximum(d, 0.05))
        lo = self.cam_pitch - self.half_vfov + self.visible_margin
        hi = self.cam_pitch + self.half_vfov - self.visible_margin
        return ((elev >= lo) & (elev <= hi)).mean(axis=-1)

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

    def _path_to_candidate(self, plan: PlanResult, cand: ObjectCandidate, radius: float,
                           ring: bool = False, away_from: Optional[float] = None,
                           min_deg: float = 0.0) -> List[Tuple[float, float]]:
        """Path to the cheapest free cell within ``radius`` of the candidate's points; with ``ring`` (D-070 visible
        stop mode) only cells from which the object is in the camera's vertical view, falling back to the plain
        approach radius when no such cell is reachable. ``away_from`` / ``min_deg`` (D-071 second look): only cells at
        least ``min_deg`` around the object from that bearing."""
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
        if ring:
            ok &= self._visible_fraction(cand, xs, ys) >= self.visible_fraction
        if away_from is not None:
            turn = np.abs((np.arctan2(ys - cand.xy[1], xs - cand.xy[0]) - away_from + np.pi) % (2 * np.pi) - np.pi)
            ok &= np.degrees(turn) >= min_deg
        cost = np.where(ok, plan.dist[r0 : r1 + 1, c0 : c1 + 1], np.inf)
        if not np.isfinite(cost).any():
            return self._path_to_candidate(plan, cand, self.approach_radius, away_from=away_from,
                                           min_deg=min_deg) if ring else []
        i = np.unravel_index(int(np.argmin(cost)), cost.shape)
        goal = (int(rr[i]), int(cc[i]))
        return self._cells_to_xy(simplify_path(plan.path_cells(goal), plan.blocked))

    def _cells_to_xy(self, cells: Sequence[Tuple[int, int]]) -> List[Tuple[float, float]]:
        if not cells:
            return []
        xs, ys = self.map.to_xy(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
        return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _length(path: Sequence[Tuple[float, float]]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


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
