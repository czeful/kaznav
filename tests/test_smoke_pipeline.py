"""GPU-free smoke test of the whole ObjectNav loop (policy -> executor -> runner -> metrics) on MockEnv.

MockEnv is an analytic box world with the HabitatEnv interface; its numbers are plumbing checks only.
"""
import math

import numpy as np
import pytest

from kznav.eval.records import column_names
from kznav.perception.base import create_detector, create_segmenter
from kznav.perception.camera import backproject, to_world
from kznav.policy.objectnav import ObjectNavPolicy, spin_steps
from kznav.run.config import compose_config
from kznav.run.episode import Executor, run_episode
from kznav.sim.episodes import GoalInstance, ObjectNavEpisode, compute_view_points
from kznav.sim.mock_env import MockBox, MockEnv, room_with_boxes

GOAL = MockBox((1.6, 0.0, -2.3), (2.1, 0.9, -1.8), semantic_id=7, color=(200, 40, 40))
TABLE = MockBox((-1.0, 0.0, -0.6), (0.4, 0.75, 0.6), semantic_id=3, color=(120, 90, 60))


def small_cfg(embodiment, detector="oracle_semantic", steps=250):
    return compose_config([f"embodiment={embodiment}", f"detector={detector}",
                           "embodiment.sensors.rgbd.resolution_hw=[120,160]", "perception.map.pixel_stride=2",
                           f"eval.episode_budget.max_decision_steps={steps}", "perception.detection.min_mask_pixels=10"])


def episode_for(env, cfg):
    g = GoalInstance("chair_7", 7, "chair", [1.85, 0.45, -2.05], list(GOAL.lo), list(GOAL.hi))
    g.view_points = compute_view_points(env, g, success_radius_m=1.0, grid_step_m=0.1, min_pixels=20, max_snap_m=0.05)
    start = [-2.0, 0.0, 2.0]
    l = env.geodesic_distance(start, g.view_points)
    return ObjectNavEpisode("mock_000", "mock_room", "mock", "smoke", start, 0.3, "chair", [g], l,
                            cfg["embodiment"]["name"])


@pytest.mark.parametrize("embodiment", ["wheeled_vacuum", "habitat_agent"])
def test_oracle_episode_succeeds(embodiment):
    cfg = small_cfg(embodiment)
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    assert ep.goals[0].view_points, "view points must exist around the goal"
    assert 2.0 < ep.start_geodesic_m < 8.0
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), ["chair"], env.K)
    res, trace = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", use_oracle_aux=True)
    assert res.outcome == "success", (res.outcome, res.termination, res.distance_to_goal)
    assert res.success and 0.0 < res.spl <= 1.0
    # on success SoftSPL = (1 - d_T / l) * efficiency, i.e. slightly below SPL
    assert res.soft_spl == pytest.approx(res.spl * (1 - res.distance_to_goal / res.start_geodesic))
    assert res.goal_ever_in_view and res.goal_ever_detected
    assert res.path_length >= res.start_geodesic - 0.1
    assert set(res.to_row()) == set(column_names())
    assert trace.decisions[0]["kind"] == "spin"
    assert sum(d["kind"] == "spin" for d in trace.decisions) >= spin_steps(cfg)


def test_without_detector_nothing_is_found():
    cfg = small_cfg("wheeled_vacuum", detector="none", steps=120)
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), ["chair"], env.K)
    res, _ = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke")
    assert not res.success and res.spl == 0.0
    assert res.outcome in ("not_found", "step_limit")
    assert res.goal_ever_in_view and not res.goal_ever_detected     # the funnel shows where it broke


def test_mock_render_matches_camera_model():
    cfg = small_cfg("wheeled_vacuum")
    env = MockEnv(cfg, room_with_boxes(6.0))
    obs = env.reset([0.0, 0.0, 0.0], math.pi / 2)              # facing -Z, wall face at z = -3.0
    h, w = obs.depth.shape
    assert obs.depth[h // 2, w // 2] == pytest.approx(3.0 - 0.14, abs=1e-5)
    v, u = np.array([h - 1]), np.array([w // 2])                # floor pixel below the optical axis
    p = to_world(backproject(obs.depth, env.K, v, u), obs.camera_transform)
    assert p[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_view_point_grid_must_cover_success_distance():
    cfg = small_cfg("wheeled_vacuum")
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL]))
    g = GoalInstance("chair_7", 7, "chair", [1.85, 0.45, -2.05], list(GOAL.lo), list(GOAL.hi))
    with pytest.raises(ValueError, match="grid step"):
        compute_view_points(env, g, success_radius_m=1.0, grid_step_m=0.2, min_pixels=20, max_snap_m=0.05)
