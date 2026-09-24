"""One ObjectNav episode: policy decisions -> embodiment actions -> metrics row.

The executor turns a ``Decision`` into motion for the embodiment:
* wheeled robot: pure pursuit along the decision's path for one decision period
  (``decision_period_s`` / ``control_dt_s`` control ticks), or a spin in place;
* point agent: one discrete habitat action per decision (turn towards the next waypoint or
  move forward), as in habitat-lab's ObjectNav.

The runner owns everything the policy must not see: the navmesh (geodesic distances for SPL),
goal view points, ground-truth semantics for the funnel columns, stuck and collision monitors.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..eval.failures import OutcomeInputs, Termination, classify_episode
from ..eval.metrics import EpisodeGeometry, SuccessRule, compute_episode_metrics, path_length
from ..eval.records import EpisodeResult
from ..eval.robot import StuckMonitor, StuckParams, count_collision_events
from ..policy.controller import PurePursuit, PurePursuitParams
from ..policy.objectnav import Decision, ObjectNavPolicy
from ..sim.episodes import ObjectNavEpisode, footprint_distance
from ..sim.frames import wrap_angle


class Executor:
    def __init__(self, env: Any, cfg: Mapping[str, Any]):
        self.env = env
        self.cfg = cfg
        emb = cfg["embodiment"]
        self.wheeled = emb["kind"] == "wheeled"
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
        else:
            self.step_m = float(emb["drive"]["forward_step_m"])
            self.turn = math.radians(float(emb["drive"]["turn_angle_deg"]))

    def reset(self) -> None:
        self.no_progress = 0
        self.last_blocked = None
        self.repeats = 0

    def execute(self, d: Decision, pose: Tuple[float, float, float]) -> Tuple[List[Any], Optional[Tuple[float, float, float]]]:
        """Run one decision. Returns the ticks and, if the robot had to back off, the pose where it was blocked."""
        env = self.env
        if self.wheeled:
            if d.kind in ("spin", "no_plan"):          # no plan yet: look around instead of standing still
                return env.drive(0.0, self.spin_w, self.ticks_per_decision), None
            if not d.path_xy:
                return env.drive(0.0, 0.0, self.ticks_per_decision), None
            self.ctrl.set_path(d.path_xy)
            ticks = []
            start = pose
            for _ in range(self.ticks_per_decision):
                v, w = self.ctrl.command(pose)
                t = env.drive(v, w, 1)[0]
                ticks.append(t)
                pose = t.pose
            return ticks, self._check_progress(start, ticks)
        # point agent: one discrete action
        if d.kind in ("spin", "no_plan") or not d.path_xy:
            return [env.act("turn_left")], None
        target = next((p for p in d.path_xy[1:] if math.hypot(p[0] - pose[0], p[1] - pose[1]) >= self.step_m),
                      d.path_xy[-1])
        err = wrap_angle(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
        if abs(err) > self.turn / 2:
            return [env.act("turn_left" if err > 0 else "turn_right")], None
        return [env.act("move_forward")], None

    def _check_progress(self, start: Tuple[float, float, float], ticks: List[Any]) -> Optional[Tuple[float, float, float]]:
        """Proprioceptive blockage check (commanded forward motion, no displacement) and back-off."""
        r = self.rec
        moved = math.hypot(ticks[-1].pose[0] - start[0], ticks[-1].pose[1] - start[1])
        v_mean = float(np.mean([t.v_cmd for t in ticks]))
        if v_mean >= float(r["min_cmd_lin_mps"]) and moved < float(r["min_progress_m"]):
            self.no_progress += 1
        else:
            self.no_progress = 0
        if self.no_progress < int(r["no_progress_decisions"]):
            return None
        self.no_progress = 0
        blocked = ticks[-1].pose
        lb = self.last_blocked
        same = lb is not None and math.hypot(blocked[0] - lb[0], blocked[1] - lb[1]) <= float(r["same_spot_radius_m"])
        self.repeats = self.repeats + 1 if same else 1
        self.last_blocked = blocked
        if self.repeats > int(r["max_repeats_same_spot"]):
            return None          # backing off does not help here (e.g. high-centred): let the stuck monitor decide
        n = max(1, int(round(float(r["backup_s"]) / self.control_dt)))
        ticks.extend(self.env.drive(-float(r["backup_speed_mps"]), 0.0, n))
        return blocked


@dataclass
class EpisodeTrace:
    """Everything recorded during an episode besides the metrics row (for logs / videos)."""

    decisions: List[Dict[str, Any]] = field(default_factory=list)
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    poses: List[Tuple[float, float, float]] = field(default_factory=list)


def _goal_pixels(semantic: Optional[np.ndarray], ids: Sequence[int]) -> np.ndarray:
    if semantic is None:
        return np.zeros((1, 1), dtype=bool)
    return np.isin(np.asarray(semantic), list(ids))


def run_episode(env: Any, policy: ObjectNavPolicy, executor: Executor, ep: ObjectNavEpisode, cfg: Mapping[str, Any],
                *, run_id: str, cell_id: str = "", backend: str = "none", model_id: str = "none",
                command_lang: str = "en", command_text: str = "", command_id: str = "", word_type: str = "universal",
                use_oracle_aux: bool = False, on_frame: Optional[Callable[..., None]] = None,
                logger: Any = None) -> Tuple[EpisodeResult, EpisodeTrace]:
    t_wall = time.perf_counter()
    ev = cfg["eval"]
    budget = int(ev["episode_budget"]["max_decision_steps"])
    fun = ev["funnel"]
    goal_ids = ep.goal_semantic_ids()
    wheeled = cfg["embodiment"]["kind"] == "wheeled"
    s = ev["stuck"]
    stuck = StuckMonitor(StuckParams(float(s["window_s"]), float(s["min_displacement_m"]), float(s["min_yaw_change_rad"]),
                                     float(s["min_cmd_lin"]), float(s["min_cmd_ang"])))
    abort_after = float(s["abort_after_s"])

    obs = env.reset(ep.start_position, ep.start_yaw)
    policy.reset(obs.pose)
    executor.reset()
    recoveries = 0
    trace = EpisodeTrace(positions=[tuple(obs.position)], poses=[obs.pose])
    contact_flags: List[bool] = []
    contact_times: List[float] = []
    in_view = detected = False
    termination = Termination.STEP_LIMIT
    stop_cand = None
    n_steps = 0
    for step in range(budget):
        n_steps = step + 1
        gp = _goal_pixels(obs.semantic, goal_ids)
        in_view |= int(gp.sum()) >= int(fun["min_goal_pixels"])
        aux = {"semantic": obs.semantic, "goal_ids": {0: goal_ids}} if use_oracle_aux else None
        d = policy.step(obs, aux)
        for det in policy.last_detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det.box_xyxy)
            box = gp[max(0, y1):y2, max(0, x1):x2] if gp.shape != (1, 1) else np.zeros((0,))
            if box.size and box.mean() >= float(fun["detection_goal_overlap"]):
                detected = True
        rec = {"type": "decision", "step": step, "kind": d.kind, "reason": d.reason, "pose": list(obs.pose),
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
        ticks, blocked_pose = executor.execute(d, obs.pose)
        if blocked_pose is not None:
            recoveries += 1
            rc = cfg["policy"]["recovery"]
            r_obs = float(rc["virtual_obstacle_radius_m"])
            policy.on_blocked(blocked_pose, ahead_m=float(cfg["embodiment"]["body"]["radius_m"]) + r_obs, radius_m=r_obs)
            if logger is not None:
                logger.event("recovery", episode_id=ep.episode_id, step=step, blocked_pose=list(blocked_pose))
        aborted = False
        for t in ticks:
            trace.positions.append(t.position)
            trace.poses.append(t.pose)
            contact_flags.append(t.collision)
            contact_times.append(t.t)
            if wheeled:
                stuck.update(t.t, *t.pose, t.v_cmd, t.w_cmd)
                if stuck.current_stuck_s >= abort_after:
                    aborted = True
                    break
        obs = env.observe()
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
    if termination is Termination.STOP_CALLED and stop_cand is not None:
        stop_is_goal = min(footprint_distance(stop_cand.position, g.aabb_min, g.aabb_max) for g in ep.goals) \
            <= float(fun["stop_candidate_radius_m"])
    outcome = classify_episode(OutcomeInputs(termination, m.success, stop_is_goal))
    gap = float(ev["collisions"]["merge_gap_s"])
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
        goal_ever_verified_positive=False, parse_ok=True, parse_attempts=0, n_hypotheses=len(policy.queries),
        target_match=None, hypothesis_match=None, vlm_calls=0, vlm_calls_verify=0, vlm_latency_total_s=0.0,
        vlm_latency_median_s=float("nan"), vlm_http_retries=0, vlm_json_retries=0, verify_logprobs_missing=0,
        recoveries=recoveries,
    )
    return res, trace
