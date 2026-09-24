"""Simulator-independent parts of the sim layer: URDF generation, drive kinematics, frames, controller."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from kznav.policy.controller import PurePursuit, PurePursuitParams
from kznav.sim.diff_drive import DriveLimits, VelocityLimiter, body_twist, integrate_unicycle, wheel_speeds
from kznav.sim.frames import habitat_to_map, map_to_habitat, quat_from_yaw, wrap_angle, yaw_from_quat
from kznav.sim.urdf import RobotGeometry, build_urdf


@pytest.fixture(scope="module")
def emb(repo_root):
    return yaml.safe_load((repo_root / "configs/embodiment/wheeled_vacuum.yaml").read_text(encoding="utf-8"))


def _xyz(el):
    return [float(v) for v in el.find("origin").get("xyz").split()]


class TestURDF:
    def test_structure(self, emb):
        root = ET.fromstring(build_urdf(emb))
        links = {l.get("name") for l in root.findall("link")}
        assert links == {"base_link", "left_wheel", "right_wheel", "front_caster", "rear_caster"}
        joints = {j.get("name"): j for j in root.findall("joint")}
        assert joints["left_wheel_joint"].get("type") == "continuous"
        assert joints["left_wheel_joint"].find("axis").get("xyz") == "-1 0 0"

    def test_base_origin_is_the_centre_of_mass(self, emb):
        """Habitat/Bullet put the root frame at the base inertial frame (stage 3 finding)."""
        root = ET.fromstring(build_urdf(emb))
        base = [l for l in root.findall("link") if l.get("name") == "base_link"][0]
        assert _xyz(base.find("inertial")) == [0.0, 0.0, 0.0]

    def test_wheels_and_casters_touch_the_floor(self, emb):
        g = RobotGeometry.from_config(emb)
        root = ET.fromstring(build_urdf(emb))
        joints = {j.get("name"): j for j in root.findall("joint")}
        com = g.com
        for side, sign in (("left", -1), ("right", 1)):
            x, y, z = _xyz(joints[f"{side}_wheel_joint"])
            assert x == pytest.approx(sign * g.wheel_base / 2)
            assert y + com[1] == pytest.approx(g.wheel_radius, abs=1e-6)       # axle at wheel-radius height
            assert z + com[2] == pytest.approx(0.0, abs=1e-6)
        fx, fy, fz = _xyz(joints["front_caster_joint"])
        assert fy + com[1] - g.caster_radius == pytest.approx(0.0, abs=1e-6)     # front caster on the floor
        rx, ry, rz = _xyz(joints["rear_caster_joint"])
        assert ry + com[1] - g.caster_radius == pytest.approx(g.rear_lift, abs=1e-6)
        assert fz + com[2] < 0 < rz + com[2]                                     # front is -Z

    def test_reference_point_maps_to_floor(self, emb):
        g = RobotGeometry.from_config(emb)
        ref = np.array(g.reference_in_base) + np.array(g.com)
        assert ref == pytest.approx([0.0, 0.0, 0.0])

    def test_top_height(self, emb):
        g = RobotGeometry.from_config(emb)
        assert g.top_height == pytest.approx(0.15 + 0.015)
        assert g.top_height <= emb["navmesh"]["agent_height_m"]

    def test_geometry_checks(self, emb):
        bad = yaml.safe_load(yaml.safe_dump(emb))
        bad["drive"]["wheel_base_m"] = 0.40
        with pytest.raises(ValueError, match="footprint"):
            RobotGeometry.from_config(bad)
        bad = yaml.safe_load(yaml.safe_dump(emb))
        bad["body"]["mass_kg"] = 0.2
        with pytest.raises(ValueError, match="mass"):
            RobotGeometry.from_config(bad)

    def test_wheel_shape_option(self, emb):
        e = yaml.safe_load(yaml.safe_dump(emb))
        for shape, tag in (("sphere", "sphere"), ("cylinder", "cylinder")):
            e["drive"]["wheel_collision_shape"] = shape
            root = ET.fromstring(build_urdf(e))
            wheel = [l for l in root.findall("link") if l.get("name") == "left_wheel"][0]
            assert wheel.find("collision/geometry")[0].tag == tag
        e["drive"]["wheel_collision_shape"] = "cube"
        with pytest.raises(ValueError):
            build_urdf(e)


LIMITS = DriveLimits(max_lin_vel=0.3, max_ang_vel=1.5, max_lin_acc=0.5, max_ang_acc=3.0)


class TestDiffDrive:
    def test_wheel_speed_round_trip(self):
        for v, w in ((0.3, 0.0), (0.0, 1.5), (0.2, -0.7)):
            wl, wr = wheel_speeds(v, w, 0.23, 0.035)
            assert body_twist(wl, wr, 0.23, 0.035) == pytest.approx((v, w))

    def test_turning_left_spins_right_wheel_faster(self):
        wl, wr = wheel_speeds(0.0, 1.0, 0.23, 0.035)
        assert wr > 0 > wl

    def test_limiter_respects_limits(self):
        lim = VelocityLimiter(LIMITS)
        vs = [lim.step(10.0, 10.0, 0.1) for _ in range(20)]
        dv = np.diff([0.0] + [v for v, _ in vs])
        dw = np.diff([0.0] + [w for _, w in vs])
        assert max(dv) <= 0.5 * 0.1 + 1e-12 and max(dw) <= 3.0 * 0.1 + 1e-12
        assert vs[-1] == pytest.approx((0.3, 1.5))

    def test_limiter_reaches_speed_in_expected_time(self):
        lim = VelocityLimiter(LIMITS)
        v = [lim.step(0.3, 0.0, 0.1)[0] for _ in range(10)]
        assert v[5] == pytest.approx(0.3)      # 0.3 / 0.5 = 0.6 s
        assert v[4] == pytest.approx(0.25)

    def test_unicycle_straight_and_circle(self):
        assert integrate_unicycle(0, 0, 0, 1.0, 0.0, 2.0) == pytest.approx((2.0, 0.0, 0.0))
        x, y, yaw = 0.0, 0.0, 0.0
        for _ in range(100):
            x, y, yaw = integrate_unicycle(x, y, yaw, 0.2, 0.5, 2 * math.pi / 0.5 / 100)
        assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert wrap_angle(yaw) == pytest.approx(0.0, abs=1e-9)


class TestFrames:
    def test_default_heading(self):
        assert yaw_from_quat((1.0, 0.0, 0.0, 0.0)) == pytest.approx(math.pi / 2)

    @pytest.mark.parametrize("yaw", [0.0, 0.7, math.pi / 2, -2.5, 3.0])
    def test_quat_yaw_round_trip(self, yaw):
        assert wrap_angle(yaw_from_quat(quat_from_yaw(yaw)) - yaw) == pytest.approx(0.0, abs=1e-12)

    def test_positions(self):
        p = map_to_habitat(1.5, -2.0, 0.3)
        assert list(p) == [1.5, 0.3, 2.0]
        assert habitat_to_map(p) == (1.5, -2.0)


def simulate(ctrl, pose, steps=2000, dt=0.1):
    lim = VelocityLimiter(LIMITS)
    for _ in range(steps):
        if ctrl.done(pose):
            return pose, True
        v, w = lim.step(*ctrl.command(pose), dt)
        pose = integrate_unicycle(*pose, v, w, dt)
    return pose, False


PARAMS = PurePursuitParams(lookahead_m=0.2, goal_tolerance_m=0.1, max_lin_vel=0.3, max_ang_vel=1.0,
                           rotate_in_place_rad=0.8, slowdown_radius_m=0.5, min_lin_vel=0.05)


class TestPurePursuit:
    def test_follows_l_path(self):
        c = PurePursuit(PARAMS)
        path = [(0, 0), (2, 0), (2, 2)]
        c.set_path(path)
        pose, ok = simulate(c, (0.0, 0.0, 0.0))
        assert ok and math.hypot(pose[0] - 2, pose[1] - 2) <= 0.1

    def test_rotates_in_place_when_target_is_behind(self):
        c = PurePursuit(PARAMS)
        c.set_path([(0, 0), (-2, 0)])
        v, w = c.command((0.0, 0.0, 0.0))
        assert v == 0.0 and abs(w) == PARAMS.max_ang_vel

    def test_stops_at_goal(self):
        c = PurePursuit(PARAMS)
        c.set_path([(0, 0), (1, 0)])
        assert c.command((0.95, 0.0, 0.0)) == (0.0, 0.0)

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError):
            PurePursuit(PARAMS).set_path([])
