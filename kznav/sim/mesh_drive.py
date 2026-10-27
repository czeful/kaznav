"""D-003 check: drive the physical robot along its own navmesh paths on scanned meshes.

For each scene: recompute the navmesh for the wheeled embodiment, sample start/goal pairs,
follow the shortest path with pure pursuit (replanning periodically) and record whether the
robot arrives, collides or gets stuck. A path that is feasible on the navmesh but not for the
physics model is either a mesh artefact (holes, bumps below the navmesh resolution) or an
inconsistency between navmesh parameters and the robot — both have to be known before the
wheeled-robot numbers can be interpreted.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..eval.robot import StuckMonitor, StuckParams, count_collision_events
from ..eval.metrics import path_length
from ..policy.controller import PurePursuit, PurePursuitParams
from .env import HabitatEnv, SceneSpec
from .frames import habitat_to_map


@dataclass
class DriveEpisode:
    """Result of one navmesh-path drive: outcome, time, path length, collisions and where it got stuck."""
    scene: str
    index: int
    start: List[float]
    goal: List[float]
    geodesic_m: float
    arrived: bool
    outcome: str                    # arrived | stuck | timeout
    time_s: float
    path_m: float
    final_dist_m: float
    collision_events: int
    stuck_time_s: float
    max_tilt_deg: float
    stuck_at: Optional[List[float]] = None
    stuck_clearance_m: Optional[float] = None   # navmesh distance to the closest obstacle where it got stuck
    stuck_links: List[str] = field(default_factory=list)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


def _stuck_params(cfg: Mapping[str, Any]) -> StuckParams:
    s = cfg["eval"]["stuck"]
    return StuckParams(float(s["window_s"]), float(s["min_displacement_m"]), float(s["min_yaw_change_rad"]),
                       float(s["min_cmd_lin"]), float(s["min_cmd_ang"]))


def sample_pairs(env: HabitatEnv, n: int, lo: float, hi: float, rng: np.random.Generator, plan_pf,
                 max_tries: int = 5000):
    """Start/goal pairs on the planning navmesh; the distance reported is the evaluation-navmesh geodesic."""
    pairs = []
    plan_pf.seed(int(rng.integers(0, 2**31 - 1)))
    for _ in range(max_tries):
        if len(pairs) >= n:
            break
        a, b = env.random_navigable_point(plan_pf), env.random_navigable_point(plan_pf)
        if not lo <= env.geodesic_distance(a, [b], plan_pf) <= hi:
            continue
        pairs.append((a, b, env.geodesic_distance(a, [b])))
    return pairs


def drive_episode(env: HabitatEnv, cfg: Mapping[str, Any], vcfg: Mapping[str, Any], start: np.ndarray,
                  goal: np.ndarray, geodesic: float, scene: str, index: int, plan_pf) -> DriveEpisode:
    """Drive the robot from ``start`` to ``goal`` along the planning navmesh with pure pursuit.

    Args:
        env: The simulator (articulated wheeled robot).
        cfg: The run config (controller, stuck detection and collision merging are read).
        vcfg: The check's config (spawn lift, time budget, replanning period, goal radius, stuck abort).
        start: Start point on the navmesh.
        goal: Goal point on the navmesh.
        geodesic: Evaluation-navmesh geodesic distance from start to goal.
        scene: Scene name recorded in the result.
        index: Index of the pair in the scene.
        plan_pf: Planning navmesh (radius inflated by a margin).

    Returns:
        The drive's outcome (``arrived``, ``stuck`` or ``timeout``) and its measurements.
    """
    ctrl = PurePursuit(PurePursuitParams.from_config(cfg["policy"]["controller"]))
    path3 = env.shortest_path(start, goal, plan_pf)
    first = np.asarray(habitat_to_map(path3[1] if len(path3) > 1 else goal)) - np.asarray(habitat_to_map(start))
    yaw0 = math.atan2(first[1], first[0])
    env.reset(start + np.array([0.0, float(vcfg["spawn_lift_m"]), 0.0]), yaw0)
    stuck = StuckMonitor(_stuck_params(cfg))
    budget = geodesic / float(vcfg["timeout_speed_mps"]) + float(vcfg["timeout_slack_s"])
    dt = env.robot.control_dt
    replan_ticks = max(1, int(round(float(vcfg["replan_every_s"]) / dt)))
    ticks = []
    outcome = "timeout"
    stuck_at = None
    stuck_clear = None
    stuck_links: List[str] = []
    k = 0
    pose = env.robot.pose2d()
    while env.sim_time < budget:
        if k % replan_ticks == 0:
            here = env.snap(env.robot.reference_position(), plan_pf)
            pts = env.shortest_path(here, goal, plan_pf) or [here, goal]
            ctrl.set_path([habitat_to_map(p) for p in pts])
        k += 1
        if math.hypot(*(np.asarray(habitat_to_map(goal)) - pose[:2])) <= float(vcfg["goal_radius_m"]):
            outcome = "arrived"
            break
        v, w = ctrl.command(pose)
        tk = env.drive(v, w, 1)[0]
        ticks.append(tk)
        pose = tk.pose
        stuck.update(tk.t, *tk.pose, tk.v_cmd, tk.w_cmd)
        if stuck.current_stuck_s >= float(vcfg["stuck_abort_s"]):
            outcome = "stuck"
            p = np.asarray(tk.position)
            stuck_at = [float(v) for v in p]
            stuck_clear = float(env.sim.pathfinder.distance_to_closest_obstacle(env.snap(p), 2.0))
            stuck_links = sorted({l for t in ticks[-50:] for l in t.collision_links})
            break
    gap = float(cfg["eval"]["collisions"]["merge_gap_s"])
    xy = [t.pose[:2] for t in ticks]
    return DriveEpisode(
        scene=scene, index=index, start=[float(v) for v in start], goal=[float(v) for v in goal],
        geodesic_m=round(geodesic, 3), arrived=outcome == "arrived", outcome=outcome, time_s=round(env.sim_time, 2),
        path_m=round(path_length([t.position for t in ticks]) if len(ticks) > 1 else 0.0, 3),
        final_dist_m=round(float(np.linalg.norm(np.asarray(habitat_to_map(goal)) - np.asarray(pose[:2]))), 3),
        collision_events=count_collision_events([t.collision for t in ticks], [t.t for t in ticks], gap),
        stuck_time_s=round(stuck.stuck_time_s, 2), max_tilt_deg=round(max((t.tilt_deg for t in ticks), default=0.0), 2),
        stuck_at=stuck_at, stuck_clearance_m=stuck_clear, stuck_links=stuck_links,
        trajectory=[(round(x, 3), round(y, 3)) for x, y in xy[:: max(1, len(xy) // 400)]],
    )


def run_scene(cfg: Mapping[str, Any], vcfg: Mapping[str, Any], scene_path: str, workdir: Path,
              rng: np.random.Generator) -> Tuple[List[DriveEpisode], Dict[str, Any]]:
    """Run the navmesh-drive check on one scene.

    Args:
        cfg: The run config.
        vcfg: The check's config (pair count and distances, planning margin, seed).
        scene_path: Scene to load.
        workdir: Directory for the simulator's generated files.
        rng: Random generator for the start/goal pairs.

    Returns:
        The drives and a scene summary (navigable areas, pair count, wall time, top-down map).
    """
    env = HabitatEnv(cfg, SceneSpec(scene_id=scene_path), workdir, seed=int(vcfg["seed"]))
    try:
        plan_pf = env.make_pathfinder(float(vcfg["planning_margin_m"]))
        info = {"scene": scene_path, "navigable_area_m2": round(env.navigable_area(), 2),
                "planning_navigable_area_m2": round(float(plan_pf.navigable_area), 2),
                "navmesh": env.navmesh_origin, "stage": env.stage_attributes}
        pairs = sample_pairs(env, int(vcfg["episodes_per_scene"]), float(vcfg["min_geodesic_m"]),
                             float(vcfg["max_geodesic_m"]), rng, plan_pf)
        info["pairs_sampled"] = len(pairs)
        eps = []
        t0 = time.perf_counter()
        for i, (a, b, d) in enumerate(pairs):
            eps.append(drive_episode(env, cfg, vcfg, a, b, d, Path(scene_path).stem, i, plan_pf))
        info["wall_s"] = round(time.perf_counter() - t0, 1)
        info["topdown"] = _topdown(env)
        return eps, info
    finally:
        env.close()


def _topdown(env: HabitatEnv, mpp: float = 0.05) -> Dict[str, Any]:
    pf = env.sim.pathfinder
    lo, hi = pf.get_bounds()
    height = float(env.snap(env.random_navigable_point())[1])
    grid = np.asarray(pf.get_topdown_view(mpp, height), dtype=bool)
    return {"grid": grid, "lower": [float(v) for v in lo], "mpp": mpp}


def summarize(eps: List[DriveEpisode]) -> Dict[str, Any]:
    """Aggregate drives: arrivals, stuck and timeout counts, collisions and stuck episodes per 100 m."""
    n = len(eps)
    if n == 0:
        return {"n": 0}
    dist = sum(e.path_m for e in eps)
    return {
        "n": n,
        "arrived": sum(e.arrived for e in eps),
        "stuck": sum(e.outcome == "stuck" for e in eps),
        "timeout": sum(e.outcome == "timeout" for e in eps),
        "arrival_rate": round(sum(e.arrived for e in eps) / n, 3),
        "driven_m": round(dist, 1),
        "collision_events_per_100m": round(100 * sum(e.collision_events for e in eps) / max(dist, 1e-9), 2),
        "stuck_episodes_per_100m": round(100 * sum(e.outcome == "stuck" for e in eps) / max(dist, 1e-9), 2),
        "median_stuck_clearance_m": (float(np.median([e.stuck_clearance_m for e in eps if e.stuck_clearance_m is not None]))
                                     if any(e.stuck_clearance_m is not None for e in eps) else None),
    }


def episode_dicts(eps: List[DriveEpisode]) -> List[Dict[str, Any]]:
    """Drives as plain dicts (for JSON)."""
    return [asdict(e) for e in eps]
