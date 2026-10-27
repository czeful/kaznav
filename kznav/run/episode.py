"""One ObjectNav episode: policy decisions -> embodiment actions -> metrics row.

The executor turns a ``Decision`` into motion for the embodiment:
* wheeled robot: pure pursuit along the decision's path for one decision period
  (``decision_period_s`` / ``control_dt_s`` control ticks), or a spin in place;
* point agent: one discrete habitat action per decision (turn towards the next waypoint or
  move forward), as in habitat-lab's ObjectNav.

The runner owns everything the policy must not see: the navmesh (geodesic distances for SPL),
goal view points, ground-truth semantics for the funnel columns, stuck and collision monitors.

Localization (D-051): the policy and the controller get the pose of the embodiment's localizer
(kznav.sim.localization) - the true pose by default, or dead-reckoned odometry in the robustness
study. The executor updates it on every tick it produces; the runner hands the policy the
observation re-expressed in the estimated frame and keeps the true pose for every metric.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..eval.failures import AttributionInputs, OutcomeInputs, Termination, attribute_failure, classify_episode
from ..eval.metrics import EpisodeGeometry, SuccessRule, compute_episode_metrics, path_length
from ..eval.records import EpisodeResult
from ..eval.robot import StuckMonitor, StuckParams, count_collision_events
from ..policy.controller import PurePursuit, PurePursuitParams
from ..policy.objectnav import Decision, ObjectNavPolicy
from ..policy.verification import event_dict
from ..sim.episodes import ObjectNavEpisode, footprint_distance
from ..sim.frames import wrap_angle
from ..sim.localization import make_localizer
from ..sim.noise import ODOMETRY_DRIFT, _build


def fall_limits(cfg: Mapping[str, Any]) -> Tuple[float, float]:
    """Tilt (deg) and drop below the start height (m) that end an episode as a fall. None for an embodiment that may
    change floors: habitat-lab's agent walks stairs, and its episodes go on on any floor (D-061)."""
    if not cfg["embodiment"].get("fall_terminates", True):
        return math.inf, math.inf
    fall = cfg["eval"]["fall"]
    return float(fall["max_tilt_deg"]), float(fall["max_drop_m"])


def episode_seed(run_seed: int, episode_id: str) -> int:
    """Deterministic per-episode seed: an episode's randomness depends on (run seed, episode id) only,
    never on which episodes the same process ran before it."""
    h = hashlib.sha256(f"{int(run_seed)}:{episode_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


class Executor:
    """Turns policy decisions into robot motion for one embodiment.

    The robot-vacuum follows the decision's path with the pure-pursuit controller for one decision period and
    backs off when blocked; the standard agent takes discrete forward and turn actions. Both update the
    localisation estimate on every tick.

    Args:
        env: The simulator.
        cfg: The run config (embodiment, controller, recovery and executor blocks are read).
    """
    def __init__(self, env: Any, cfg: Mapping[str, Any]):
        self.env = env
        self.cfg = cfg
        emb = cfg["embodiment"]
        self.wheeled = emb["kind"] == "wheeled"
        drift = _build(ODOMETRY_DRIFT, (emb.get("noise") or {}).get("odometry_drift", {}), np.random.default_rng(0))
        self.localizer = make_localizer(emb, drift)       # re-seeded per episode by run_episode
        self._est: List[Tuple[float, float, float]] = []
        if self.wheeled:
            self.ctrl = PurePursuit(PurePursuitParams.from_config(cfg["policy"]["controller"]))
            period = float(cfg["eval"]["episode_budget"]["decision_period_s"])
            self.ticks_per_decision = max(1, int(round(period / float(emb["physics"]["control_dt_s"]))))
            self.spin_w = float(cfg["policy"]["exploration"]["spin_ang_vel_radps"])
            self.rec = cfg["policy"]["recovery"]
            self.control_dt = float(emb["physics"]["control_dt_s"])
            self.no_progress = 0
            self.last_blocked: Optional[Tuple[float, float, float]] = None
            self.repeats = 0
            self.cliffs = 0
        else:
            self.step_m = float(emb["drive"]["forward_step_m"])
            self.turn = math.radians(float(emb["drive"]["turn_angle_deg"]))
            pe = cfg["policy"].get("point_executor") or {}
            # D-063: the path point to steer at (this far ahead) and the heading error tolerated before turning;
            # defaults = the behaviour before D-063 (one step ahead, half a turn)
            self.lookahead = float(pe.get("lookahead_m", self.step_m))
            self.tolerance = math.radians(float(pe.get("heading_tolerance_deg", math.degrees(self.turn) / 2)))
            self.mark_collisions = bool(pe.get("mark_collisions", False))

    def reset(self) -> None:
        """Clear the recovery state at the start of an episode."""
        self.no_progress = 0
        self.last_blocked = None
        self.repeats = 0
        self.cliffs = 0

    def _drive(self, v: float, w: float, n: int) -> List[Any]:
        """env.drive, with the localizer updated on every tick (``self._est``: the estimates of this decision)."""
        ticks = self.env.drive(v, w, n)
        for t in ticks:
            self._est.append(self.localizer.tick(t))
        return ticks

    def _act(self, action: str) -> Any:
        t = self.env.act(action)
        self._est.append(self.localizer.tick(t))
        return t

    def execute(self, d: Decision, pose: Tuple[float, float, float]) -> Tuple[List[Any], Optional[Tuple[float, float, float]]]:
        """Run one decision from the (estimated) ``pose``. Returns the ticks and, if the robot had to back off, the
        estimated pose where it was blocked."""
        self._est = []
        if self.wheeled:
            if d.kind in ("spin", "no_plan"):          # no plan yet: look around instead of standing still
                return self._drive(0.0, self.spin_w, self.ticks_per_decision), None
            if d.kind == "face" and d.target_xy is not None:
                return self._face(d.target_xy, pose), None
            if not d.path_xy:
                return self._drive(0.0, 0.0, self.ticks_per_decision), None
            self.ctrl.set_path(d.path_xy)
            ticks = []
            start = pose
            for _ in range(self.ticks_per_decision):
                v, w = self.ctrl.command(pose)
                t = self._drive(v, w, 1)[0]
                ticks.append(t)
                pose = self.localizer.pose
                if t.cliff:
                    # cliff sensor: stop, back off, report where (the policy marks it on its map)
                    self.cliffs += 1
                    n = max(1, int(round(float(self.rec["backup_s"]) / self.control_dt)))
                    ticks.extend(self._drive(-float(self.rec["backup_speed_mps"]), 0.0, n))
                    return ticks, pose
            return ticks, self._check_progress(start, ticks)
        # point agent: one discrete action
        if d.kind == "face" and d.target_xy is not None:
            err = wrap_angle(math.atan2(d.target_xy[1] - pose[1], d.target_xy[0] - pose[0]) - pose[2])
            return [self._act("turn_left" if err > 0 else "turn_right")], None
        if d.kind in ("spin", "no_plan") or not d.path_xy:
            return [self._act("turn_left")], None
        target = next((p for p in d.path_xy[1:] if math.hypot(p[0] - pose[0], p[1] - pose[1]) >= self.lookahead),
                      d.path_xy[-1])
        err = wrap_angle(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
        if abs(err) > self.tolerance:
            return [self._act("turn_left" if err > 0 else "turn_right")], None
        t = self._act("move_forward")
        if self.mark_collisions and t.collision:
            # D-063: the navmesh stopped the step (something the map does not show, or a gap too narrow): report
            # where, so the policy marks a virtual obstacle ahead and plans around it instead of pushing again
            end = self._est[-1]
            if math.hypot(end[0] - pose[0], end[1] - pose[1]) < 0.5 * self.step_m:
                return [t], pose
        return [t], None

    def _face(self, target: Tuple[float, float], pose: Tuple[float, float, float]) -> List[Any]:
        """Turn in place towards ``target`` for (at most) one decision period."""
        ticks = []
        for _ in range(self.ticks_per_decision):
            err = wrap_angle(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
            if abs(err) < 0.05:
                break
            w = math.copysign(min(self.spin_w, 2.0 * abs(err)), err)
            t = self._drive(0.0, w, 1)[0]
            ticks.append(t)
            pose = self.localizer.pose
        return ticks or self._drive(0.0, 0.0, 1)

    def _check_progress(self, start: Tuple[float, float, float], ticks: List[Any]) -> Optional[Tuple[float, float, float]]:
        """Proprioceptive blockage check (commanded forward motion, no displacement by the robot's own pose
        estimate) and back-off."""
        r = self.rec
        end = self._est[len(ticks) - 1] if len(self._est) >= len(ticks) else ticks[-1].pose
        moved = math.hypot(end[0] - start[0], end[1] - start[1])
        v_mean = float(np.mean([t.v_cmd for t in ticks]))
        if v_mean >= float(r["min_cmd_lin_mps"]) and moved < float(r["min_progress_m"]):
            self.no_progress += 1
        else:
            self.no_progress = 0
        if self.no_progress < int(r["no_progress_decisions"]):
            return None
        self.no_progress = 0
        blocked = end
        lb = self.last_blocked
        same = lb is not None and math.hypot(blocked[0] - lb[0], blocked[1] - lb[1]) <= float(r["same_spot_radius_m"])
        self.repeats = self.repeats + 1 if same else 1
        self.last_blocked = blocked
        if self.repeats > int(r["max_repeats_same_spot"]):
            return None          # backing off does not help here (e.g. high-centred): let the stuck monitor decide
        n = max(1, int(round(float(r["backup_s"]) / self.control_dt)))
        ticks.extend(self._drive(-float(r["backup_speed_mps"]), 0.0, n))
        return blocked


@dataclass
class EpisodeTrace:
    """Everything recorded during an episode besides the metrics row (for logs / videos)."""

    decisions: List[Dict[str, Any]] = field(default_factory=list)
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    poses: List[Tuple[float, float, float]] = field(default_factory=list)


def _goal_fraction(goal_pixels: np.ndarray, box: Sequence[float]) -> Optional[float]:
    """Share of goal-instance pixels inside an image box (None without a semantic image or an empty box)."""
    if goal_pixels.shape == (1, 1) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    region = goal_pixels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return float(region.mean()) if region.size else None


def _goal_pixels(semantic: Optional[np.ndarray], ids: Sequence[int]) -> np.ndarray:
    if semantic is None:
        return np.zeros((1, 1), dtype=bool)
    return np.isin(np.asarray(semantic), list(ids))


def run_episode(env: Any, policy: ObjectNavPolicy, executor: Executor, ep: ObjectNavEpisode, cfg: Mapping[str, Any],
                *, run_id: str, cell_id: str = "", backend: str = "none", model_id: str = "none",
                command_lang: str = "en", command_text: str = "", command_id: str = "", word_type: str = "universal",
                use_oracle_aux: bool = False, on_frame: Optional[Callable[..., None]] = None,
                logger: Any = None, language: Any = None, calls: Any = None) -> Tuple[EpisodeResult, EpisodeTrace]:
    """``language``: the episode's ``LanguageGoal`` (None: queries are set by the caller, no VLM);
    ``calls``: a ListSink fed by the VLM backend, for the episode's cost columns."""
    t_wall = time.perf_counter()
    n_calls0 = len(calls.records) if calls is not None else 0
    ev = cfg["eval"]
    budget = int(ev["episode_budget"]["max_decision_steps"])
    fun = ev["funnel"]
    goal_ids = ep.goal_semantic_ids()
    wheeled = cfg["embodiment"]["kind"] == "wheeled"
    s = ev["stuck"]
    stuck = StuckMonitor(StuckParams(float(s["window_s"]), float(s["min_displacement_m"]), float(s["min_yaw_change_rad"]),
                                     float(s["min_cmd_lin"]), float(s["min_cmd_ang"])))
    abort_after = float(s["abort_after_s"])
    max_tilt, max_drop = fall_limits(cfg)

    seed = episode_seed(int(cfg["run"]["seed"]), ep.episode_id)
    if hasattr(env, "reseed_noise"):
        env.reseed_noise(seed)                     # depth / actuation noise streams of this episode (D-034)
    obs = env.reset(ep.start_position, ep.start_yaw)
    start_height = float(obs.position[1])
    loc = executor.localizer
    loc.reset(obs.pose, seed=seed ^ 0x5EED)
    pobs = loc.observation(obs)                    # what the policy sees: its own (estimated) frame
    policy.reset(pobs.pose, seed=seed)
    executor.reset()
    recoveries = 0
    trace = EpisodeTrace(positions=[tuple(obs.position)], poses=[obs.pose])
    contact_flags: List[bool] = []
    contact_times: List[float] = []
    in_view = detected = verified_goal = False
    termination = Termination.STEP_LIMIT
    stop_cand = None
    n_steps = 0
    verifier = getattr(policy, "verifier", None)
    n_events = 0
    stop_radius = float(fun["stop_candidate_radius_m"])
    content_thr = float(fun["verify_label_box_overlap"])
    verify_rejected_goal = False
    # Privileged content label per candidate: goal pixels in its latest detection box, from the frame of that
    # detection (D-046). Position labels depend on where the segmenter put the candidate; this does not.
    cand_fraction: Dict[int, Optional[float]] = {}

    def is_goal(cand: Any) -> bool:            # privileged: for analysis columns / logs only
        return min(footprint_distance(cand.position, g.aabb_min, g.aabb_max) for g in ep.goals) <= stop_radius

    def goal_label(cand: Any, fraction: Optional[float]) -> bool:
        return fraction >= content_thr if fraction is not None else is_goal(cand)

    if language is not None and not language.parse_ok:
        budget = 0
        termination = Termination.MODEL_ERROR
    for step in range(budget):
        n_steps = step + 1
        gp = _goal_pixels(obs.semantic, goal_ids)
        in_view |= int(gp.sum()) >= int(fun["min_goal_pixels"])
        aux = {"semantic": obs.semantic, "goal_ids": {0: goal_ids}} if use_oracle_aux else None
        frame_step = policy.step_idx
        d = policy.step(pobs, aux)
        for c in policy.memory.candidates:
            if c.last_step == frame_step and c.last_box is not None:
                cand_fraction[c.cid] = _goal_fraction(gp, c.last_box)
        if verifier is not None:
            for vev in verifier.events[n_events:]:
                cand = next(c for c in policy.memory.candidates if c.cid == vev.cid)
                # Two privileged labels: by position (the candidate is near a goal instance) and by content
                # (goal pixels inside the detection box) - the label the verifier is calibrated against (D-042).
                box_f, crop_f = _goal_fraction(gp, vev.box_xyxy), _goal_fraction(gp, vev.crop_xyxy)
                goal = goal_label(cand, box_f)
                verified_goal |= goal and vev.decision == "stop"
                verify_rejected_goal |= goal and vev.decision == "reject"
                if logger is not None:
                    logger.event("verification", episode_id=ep.episode_id, **event_dict(vev),
                                 is_goal_privileged=is_goal(cand), box_goal_fraction=box_f, crop_goal_fraction=crop_f,
                                 is_goal_by_content=None if box_f is None else box_f >= content_thr)
            n_events = len(verifier.events)
        for det in policy.last_detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det.box_xyxy)
            box = gp[max(0, y1):y2, max(0, x1):x2] if gp.shape != (1, 1) else np.zeros((0,))
            if box.size and box.mean() >= float(fun["detection_goal_overlap"]):
                detected = True
        rec = {"type": "decision", "step": step, "kind": d.kind, "reason": d.reason, "pose": list(obs.pose),
               **({"est_pose": list(pobs.pose)} if loc.mode != "ground_truth" else {}),
               "candidate": d.candidate_id, "n_frontiers": d.n_frontiers, "n_detections": len(policy.last_detections),
               "sim_time": env.sim_time}
        trace.decisions.append(rec)
        if logger is not None:
            logger.event("decision", episode_id=ep.episode_id, **{k: v for k, v in rec.items() if k != "type"})
        if on_frame is not None:
            on_frame(obs=obs, decision=d, policy=policy, trace=trace, step=step, episode=ep)
        if d.kind == "stop":
            termination = Termination.STOP_CALLED
            stop_cand = next((c for c in policy.memory.candidates if c.cid == d.candidate_id), None)
            break
        if d.kind == "no_frontiers":
            termination = Termination.NO_FRONTIERS
            break
        ticks, blocked_pose = executor.execute(d, pobs.pose)
        if blocked_pose is not None:
            recoveries += 1
            rc = cfg["policy"]["recovery"]
            r_obs = float(rc["virtual_obstacle_radius_m"])
            policy.on_blocked(blocked_pose, ahead_m=float(cfg["embodiment"]["body"]["radius_m"]) + r_obs, radius_m=r_obs)
            if logger is not None:
                logger.event("recovery", episode_id=ep.episode_id, step=step, blocked_pose=list(blocked_pose))
        aborted = fell = False
        for t in ticks:
            if t.tilt_deg > max_tilt or t.position[1] < start_height - max_drop:
                fell = True          # the fall itself is not added to the path (it is not driving)
                break
            trace.positions.append(t.position)
            trace.poses.append(t.pose)
            contact_flags.append(t.collision)
            contact_times.append(t.t)
            if wheeled:
                stuck.update(t.t, *t.pose, t.v_cmd, t.w_cmd)
                if stuck.current_stuck_s >= abort_after:
                    aborted = True
                    break
        if fell:
            termination = Termination.FELL
            if logger is not None:
                logger.event("fell", episode_id=ep.episode_id, step=step, position=list(ticks[-1].position),
                             tilt_deg=float(ticks[-1].tilt_deg))
            break
        obs = env.observe()
        pobs = loc.observation(obs)
        if aborted:
            termination = Termination.STUCK_ABORT
            if logger is not None and getattr(env, "robot", None) is not None:
                r = env.robot
                logger.event("stuck_diagnostics", episode_id=ep.episode_id, step=step,
                             position=[float(v) for v in r.reference_position()], tilt_deg=r.tilt_deg(),
                             wheel_velocities=list(r.wheel_velocities()), contacts=r.contact_details(),
                             last_decision=d.kind, last_path=[list(p) for p in d.path_xy])
            break

    # ---------------------------------------------------------------- metrics (privileged)
    final = np.asarray(trace.positions[-1], dtype=np.float64)
    vps = ep.view_points()
    d_final = env.geodesic_distance(final, vps) if vps else math.inf
    rule = SuccessRule(float(cfg["dataset"]["success_distance_m"]), bool(ev["success"]["require_visible_at_stop"]))
    geom = EpisodeGeometry(start_geodesic=ep.start_geodesic_m, path_length=path_length(trace.positions),
                           final_geodesic=d_final, called_stop=termination is Termination.STOP_CALLED)
    m = compute_episode_metrics(geom, rule)
    stop_is_goal: Optional[bool] = None
    stop_fraction: Optional[float] = None
    stop_by_position: Optional[bool] = None
    stop_source = ""
    if termination is Termination.STOP_CALLED and stop_cand is not None:
        stop_fraction = cand_fraction.get(stop_cand.cid)
        stop_by_position = is_goal(stop_cand)
        stop_is_goal = goal_label(stop_cand, stop_fraction)
        stop_source = "content" if stop_fraction is not None else "position"
    outcome = classify_episode(OutcomeInputs(termination, m.success, stop_is_goal))
    gap = float(ev["collisions"]["merge_gap_s"])
    recs = [r for r in (calls.records[n_calls0:] if calls is not None else []) if not r.context.warmup]
    if language is not None and language.parse_record is not None and not any(r is language.parse_record for r in recs):
        recs = [language.parse_record] + recs      # parse cost is charged to every episode (see GoalCache)
    lat = [float(r.latency_total_s) for r in recs]
    events = verifier.events if verifier is not None else []
    parse_ok = language.parse_ok if language is not None else True
    hyp_match = language.hypothesis_match if language is not None else None
    loc_err = loc.error(tuple(trace.poses[-1]))
    stage = attribute_failure(AttributionInputs(
        success=m.success, termination=termination, stop_candidate_is_goal=stop_is_goal, parse_ok=parse_ok,
        hypothesis_match=hyp_match, goal_ever_in_view=bool(in_view), goal_ever_detected=bool(detected),
        goal_ever_verify_rejected=verify_rejected_goal))
    res = EpisodeResult(
        run_id=run_id, cell_id=cell_id, dataset=ep.dataset, split=ep.split, scene_id=ep.scene_id,
        episode_id=ep.episode_id, goal_category=ep.goal_category, word_type=word_type, command_lang=command_lang,
        command_id=command_id, command_text=command_text or ep.goal_category, backend=backend, model_id=model_id,
        embodiment=str(cfg["embodiment"]["name"]), seed=int(cfg["run"]["seed"]),
        success=m.success, spl=m.spl, soft_spl=m.soft_spl, dts=m.dts, distance_to_goal=m.distance_to_goal,
        start_geodesic=m.start_geodesic, path_length=m.path_length, num_decision_steps=n_steps,
        sim_time_s=float(env.sim_time), wall_time_s=time.perf_counter() - t_wall,
        termination=termination.value, outcome=outcome.value, stop_candidate_is_goal=stop_is_goal,
        collisions=count_collision_events(contact_flags, contact_times, gap) if contact_flags else 0,
        stuck_time_s=float(stuck.stuck_time_s), goal_ever_in_view=bool(in_view), goal_ever_detected=bool(detected),
        goal_ever_verified_positive=bool(verified_goal),
        parse_ok=parse_ok,
        parse_attempts=language.parse_attempts if language is not None else 0, n_hypotheses=len(policy.queries),
        target_match=language.target_match if language is not None else None,
        hypothesis_match=hyp_match,
        vlm_calls=len(recs), vlm_calls_verify=sum(1 for r in recs if r.method == "verify_candidate"),
        vlm_latency_total_s=float(sum(lat)), vlm_latency_median_s=float(np.median(lat)) if lat else float("nan"),
        vlm_http_retries=sum(int(r.http_retries) for r in recs), vlm_json_retries=sum(int(r.json_retries) for r in recs),
        verify_logprobs_missing=sum(1 for e in events if e.called and e.p_raw is None),
        recoveries=recoveries,
        goal_mode=language.goal_mode if language is not None else "",
        pipeline_lang=language.effective_lang if language is not None else "",
        translation=language.translation if language is not None else "",
        translation_latency_s=float(language.translation_record.latency_total_s)
        if language is not None and language.translation_record is not None else 0.0,
        stop_box_goal_fraction=stop_fraction, stop_is_goal_by_position=stop_by_position, stop_label_source=stop_source,
        goal_ever_verify_rejected=verify_rejected_goal, failure_stage=stage.value,
        vlm_call_errors=sum(1 for r in recs if not r.ok),
        localization=loc.mode, loc_error_final_m=float(loc_err[0]), loc_heading_error_final_deg=math.degrees(loc_err[1]),
    )
    return res, trace
