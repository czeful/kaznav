"""Noise models and odometry localization (D-051)."""
import math

import numpy as np
import pytest

from kznav.perception.base import create_segmenter
from kznav.perception.oracle import OracleSemanticDetector
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.run.episode import Executor, run_episode
from kznav.sim.env import Observation, Tick
from kznav.sim.frames import map_to_habitat
from kznav.sim.localization import (GroundTruthLocalizer, OdometryLocalizer, compose, frame_correction, make_localizer,
                                    relative)
from kznav.sim.mock_env import MockEnv, room_with_boxes
from kznav.sim.noise import (GaussianDepthNoise, GaussianIncrementDrift, GaussianTwistNoise, NoOdometryDrift,
                             build_noise_models, redwood_model_path, redwood_simulate, reseed_noise_models)

from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg

RNG = lambda s=0: np.random.default_rng(s)          # noqa: E731


# ---------------------------------------------------------------- depth
@pytest.mark.skipif(redwood_model_path() is None, reason="needs habitat-sim's Redwood model file")
def test_redwood_matches_habitat_sim_statistically():
    from habitat_sim.sensors.noise_models.redwood_depth_noise_model import RedwoodNoiseModelCPUImpl

    model = np.load(redwood_model_path())
    hab = RedwoodNoiseModelCPUImpl(model.copy(), 1.0)
    ours = model.reshape(80, 80, 5).astype(np.float32)
    for z in (1.0, 2.5, 5.0):
        gt = np.full((120, 160), z, dtype=np.float32)
        a = np.concatenate([redwood_simulate(gt, ours, 1.0, RNG(i)).ravel() for i in range(5)])
        b = np.concatenate([hab.simulate(gt).ravel() for _ in range(5)])
        a, b = a[a > 0], b[b > 0]
        assert a.mean() == pytest.approx(b.mean(), rel=0.01)
        assert a.std() == pytest.approx(b.std(), rel=0.15)
    far = redwood_simulate(np.full((20, 20), 12.0, np.float32), ours, 1.0, RNG())
    assert not far.any()                                                  # no return beyond 10 m


def test_redwood_is_seeded_and_scales_with_the_multiplier():
    model = np.ones((80, 80, 5), np.float32)
    gt = np.full((60, 80), 3.0, np.float32)
    assert np.array_equal(redwood_simulate(gt, model, 1.0, RNG(3)), redwood_simulate(gt, model, 1.0, RNG(3)))
    quiet = redwood_simulate(gt, model, 0.2, RNG(1))
    loud = redwood_simulate(gt, model, 2.0, RNG(1))
    assert quiet.std() < loud.std()


def test_gaussian_depth_noise_keeps_holes_and_drops_pixels():
    d = np.full((50, 50), 2.0, np.float32)
    d[0, :] = 0.0
    out = GaussianDepthNoise({"rel_std": 0.05, "dropout": 0.1}, RNG()).apply(d)
    assert not out[0].any() and 0.05 < (out[1:] == 0).mean() < 0.15
    assert out[out > 0].std() == pytest.approx(0.1, rel=0.2)


# ---------------------------------------------------------------- actuation and drift
def test_twist_noise_holds_errors_and_reseeds():
    n = GaussianTwistNoise({"v_rel_std": 0.1, "w_rel_std": 0.1, "w_bias_std": 0.05, "hold_s": 1.0}, RNG())
    first = [n.apply(0.2, 0.5, 0.1) for _ in range(10)]
    assert len(set(first)) == 1                                           # held for 1 s
    assert n.apply(0.2, 0.5, 0.1) != first[0]                             # resampled
    assert n.apply(0.0, 0.0, 0.1) == (0.0, 0.0)                           # standing still stays still
    n.reseed(7)
    a = [n.apply(0.2, 0.5, 0.1) for _ in range(25)]
    n.reseed(7)
    assert a == [n.apply(0.2, 0.5, 0.1) for _ in range(25)]


def test_increment_drift_statistics():
    d = GaussianIncrementDrift({"trans_rel_std": 0.02, "yaw_deg_per_m_std": 2.0, "rot_rel_std": 0.0,
                                "lat_std_per_m": 0.0}, RNG())
    out = np.array([d.apply(0.1, 0.0, 0.0) for _ in range(20000)])
    assert out[:, 0].std() == pytest.approx(0.002, rel=0.05) and abs(out[:, 0].mean() - 0.1) < 1e-4
    assert out[:, 2].std() == pytest.approx(math.radians(2.0) * 0.1, rel=0.05)


def test_models_reseed_per_episode():
    cfg = {"depth": {"model": "gaussian", "rel_std": 0.1}, "actuation": {"model": "gaussian_twist"},
           "odometry_drift": {"model": "gaussian_increment"}}
    models = build_noise_models(cfg, seed=0)
    reseed_noise_models(models, 11)
    a = (models[0].apply(np.ones((4, 4), np.float32)), models[1].apply(0.2, 0.1, 0.05), models[2].apply(0.1, 0, 0))
    models[1].apply(0.3, 0.3, 5.0)                                        # consume some randomness
    reseed_noise_models(models, 11)
    b = (models[0].apply(np.ones((4, 4), np.float32)), models[1].apply(0.2, 0.1, 0.05), models[2].apply(0.1, 0, 0))
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


# ---------------------------------------------------------------- frames and localizers
def test_relative_and_compose_are_inverse():
    a, b = (1.0, 2.0, 0.7), (1.5, 1.2, -2.9)
    assert compose(a, relative(a, b)) == pytest.approx(b)


def obs_at(pose, height=0.0):
    pos = map_to_habitat(pose[0], pose[1], height)
    T = np.eye(4)
    yaw0 = pose[2] - math.pi / 2                     # camera looks along -Z at yaw pi/2
    c, s = math.cos(yaw0), math.sin(yaw0)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[:3, 3] = pos + [0, 0.1, 0]
    return Observation(rgb=np.zeros((2, 2, 3), np.uint8), depth=np.zeros((2, 2), np.float32), semantic=None,
                       pose=tuple(pose), position=pos, camera_transform=T, sim_time=0.0)


def test_frame_correction_moves_the_true_pose_onto_the_estimate():
    true, est = (2.0, -1.0, 0.3), (2.4, -0.7, 0.9)
    C = frame_correction(true, est)
    o = obs_at(true, height=0.05)
    moved = C @ np.append(o.position, 1.0)
    assert moved[:3] == pytest.approx(map_to_habitat(est[0], est[1], 0.05))
    fwd = (C @ o.camera_transform)[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert math.atan2(-fwd[2], fwd[0]) == pytest.approx(est[2])          # the camera now faces the estimated yaw


def tick(t, pose, twist=None):
    return Tick(t=t, pose=pose, position=tuple(map_to_habitat(pose[0], pose[1])), v_cmd=0.0, w_cmd=0.0,
                collision=False, odom_twist=twist)


def test_odometry_without_drift_tracks_the_truth_and_with_drift_does_not():
    rng = RNG(5)
    poses = [(0.0, 0.0, 0.0)]
    for _ in range(200):
        poses.append(compose(poses[-1], (0.02 + 0.01 * rng.random(), 0.0, 0.1 * rng.standard_normal())))
    perfect = OdometryLocalizer(NoOdometryDrift({}, RNG()), "true_increment")
    perfect.reset(poses[0])
    for i, p in enumerate(poses[1:]):
        perfect.tick(tick(0.1 * (i + 1), p))
    assert perfect.error(poses[-1])[0] < 1e-9
    noisy = OdometryLocalizer(GaussianIncrementDrift({"yaw_deg_per_m_std": 5.0}, RNG()), "true_increment")
    noisy.reset(poses[0], seed=1)
    for i, p in enumerate(poses[1:]):
        noisy.tick(tick(0.1 * (i + 1), p))
    assert noisy.error(poses[-1])[0] > 0.01
    o = noisy.observation(obs_at(poses[-1]))
    assert o.pose == noisy.pose and o.position[:1] == pytest.approx([noisy.pose[0]])


def test_wheel_encoder_odometry_integrates_the_twist():
    loc = OdometryLocalizer(NoOdometryDrift({}, RNG()), "wheel_encoders")
    loc.reset((0.0, 0.0, 0.0))
    for i in range(10):                                                   # 1 s of 0.2 m/s straight
        loc.tick(tick(0.1 * (i + 1), (0.0, 0.0, 0.0), twist=(0.2, 0.0)))  # true pose did not move: wheels slip
    assert loc.pose[0] == pytest.approx(0.2) and loc.error((0.0, 0.0, 0.0))[0] == pytest.approx(0.2)


def test_make_localizer_reads_the_embodiment():
    assert isinstance(make_localizer({"localization": "ground_truth"}, None), GroundTruthLocalizer)
    loc = make_localizer({"localization": "odometry", "kind": "wheeled", "sensors": {"odometry": {}}},
                         NoOdometryDrift({}, RNG()))
    assert loc.source == "wheel_encoders"
    with pytest.raises(ValueError):
        make_localizer({"localization": "slam"}, None)


# ---------------------------------------------------------------- end to end on the mock simulator
def run(cfg):
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    pol = ObjectNavPolicy(cfg, OracleSemanticDetector({"min_pixels": 20}), create_segmenter(cfg["segmenter"]),
                          ["chair"], env.K)
    res, trace = run_episode(env, pol, Executor(env, cfg), ep, cfg, run_id="t", use_oracle_aux=True)
    return res, trace


@pytest.mark.parametrize("embodiment", ["habitat_agent", "wheeled_vacuum"])
def test_odometry_without_drift_reproduces_ground_truth_and_drift_is_deterministic(embodiment):
    gt_cfg = small_cfg(embodiment)
    gt, gt_trace = run(gt_cfg)
    emb = {**gt_cfg["embodiment"], "localization": "odometry",
           "sensors": {**gt_cfg["embodiment"]["sensors"], "odometry": {"source": "true_increment"}}}
    odo, odo_trace = run({**gt_cfg, "embodiment": emb})
    assert odo.localization == "odometry" and gt.localization == "ground_truth"
    assert (odo.success, odo.num_decision_steps) == (gt.success, gt.num_decision_steps)
    assert np.allclose(odo_trace.positions, gt_trace.positions) and odo.loc_error_final_m < 1e-6
    drift = {**emb, "noise": {**emb["noise"], "odometry_drift": {"model": "gaussian_increment",
                                                                 "yaw_deg_per_m_std": 3.0}}}
    a, _ = run({**gt_cfg, "embodiment": drift})
    b, _ = run({**gt_cfg, "embodiment": drift})
    same = lambda x, y: x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))  # noqa: E731
    ra, rb = a.to_row(), b.to_row()
    assert a.loc_error_final_m > 0 and all(same(ra[k], rb[k]) for k in ra if k != "wall_time_s")
