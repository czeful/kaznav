"""Integration tests against Habitat (skipped where habitat_sim is not installed).

Kept short (a few simulated seconds each); the full physics suite is tools/validate_robot.py.
Each test creates its own simulator and closes it: Habitat supports one live GL context per process.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

habitat_sim = pytest.importorskip("habitat_sim")

from kznav.run.config import compose_config  # noqa: E402
from kznav.sim.arena import Box, floor, populate  # noqa: E402
from kznav.sim.env import HabitatEnv, SceneSpec  # noqa: E402

pytestmark = pytest.mark.habitat
FWD = math.pi / 2
TEST_SCENES = Path(os.environ.get("KZNAV_DATA_ROOT", Path.home() / "kznav_data")) / "scene_datasets" / "habitat-test-scenes"


@pytest.fixture(scope="module")
def wcfg():
    return compose_config(["embodiment=wheeled_vacuum"])


def arena(cfg, tmp_path, boxes=()):
    return HabitatEnv(cfg, SceneSpec("NONE", populate=populate([floor(), *boxes])), tmp_path, seed=0)


def test_robot_rests_on_floor_and_drives_straight(wcfg, tmp_path):
    env = arena(wcfg, tmp_path)
    try:
        obs = env.reset((0.0, 0.0, 0.0), FWD)
        assert abs(obs.position[1]) < 0.003
        assert obs.rgb.shape == (480, 640, 3) and obs.depth.shape == (480, 640)
        ticks = env.drive(0.3, 0.0, 30)
        speed = (ticks[-1].pose[1] - ticks[-11].pose[1]) / 1.0
        assert speed == pytest.approx(0.3, rel=0.05)
        assert abs(ticks[-1].pose[0]) < 0.01
        assert not any(t.collision for t in ticks)
    finally:
        env.close()


def test_wall_contact_is_a_collision(wcfg, tmp_path):
    env = arena(wcfg, tmp_path, [Box((0.0, 0.3, -0.75), (2.0, 0.3, 0.05), name="wall")])
    try:
        env.reset((0.0, 0.0, 0.0), FWD)
        ticks = env.drive(0.3, 0.0, 40)
        assert any(t.collision for t in ticks)
        assert "base_link" in {l for t in ticks for l in t.collision_links}
        assert ticks[-1].pose[1] + 0.17 <= 0.70 + 0.01     # front edge stops at the wall face
    finally:
        env.close()


def test_stage_margin_override(wcfg, tmp_path):
    glb = TEST_SCENES / "van-gogh-room.glb"
    if not glb.exists():
        pytest.skip("habitat test scenes not downloaded")
    env = HabitatEnv(wcfg, SceneSpec(str(glb)), tmp_path, seed=0)
    try:
        assert env.stage_attributes["margin"] == 0.0
        assert env.sim.get_stage_initialization_template().margin == 0.0
        p = env.snap(env.random_navigable_point())
        obs = env.reset(p + np.array([0.0, 0.005, 0.0]), FWD)
        hit = env.sim.cast_ray(habitat_sim.geo.Ray(np.array([p[0], p[1] + 0.5, p[2]], dtype=np.float32),
                                                   np.array([0.0, -1.0, 0.0], dtype=np.float32)), 2.0)
        floor_y = [h.point[1] for h in hit.hits if h.object_id == -1][0]   # skip the robot itself
        assert abs(obs.position[1] - floor_y) < 0.01, "robot must rest on the rendered floor, not 4 cm above it"
    finally:
        env.close()


def test_point_agent_discrete_actions(tmp_path):
    glb = TEST_SCENES / "van-gogh-room.glb"
    if not glb.exists():
        pytest.skip("habitat test scenes not downloaded")
    cfg = compose_config(["embodiment=habitat_agent"])
    env = HabitatEnv(cfg, SceneSpec(str(glb)), tmp_path, seed=0)
    try:
        p = env.snap(env.random_navigable_point())
        obs = env.reset(p, FWD)
        assert obs.camera_transform[1, 3] == pytest.approx(p[1] + 0.88, abs=1e-3)
        t = env.act("turn_left")
        assert t.pose[2] == pytest.approx(FWD + math.radians(30), abs=1e-3)
        t = env.act("move_forward")
        moved = math.hypot(t.pose[0] - obs.pose[0], t.pose[1] - obs.pose[1])
        assert moved <= 0.25 + 1e-4
    finally:
        env.close()
