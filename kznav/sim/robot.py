"""Embodiments inside Habitat: the wheeled robot (Bullet articulated object) and the standard point agent.

``WheeledRobot`` turns (v, w) commands into wheel-motor velocity targets through the
speed/acceleration limiter and the (optional) actuation-noise model; the motion itself is
whatever Bullet produces from wheel torques, friction and contacts — slipping, bumping and
getting stuck are not scripted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .diff_drive import DriveLimits, VelocityLimiter, body_twist, wheel_speeds
from .frames import habitat_to_map, quat_from_yaw, yaw_from_quat
from .noise import ActuationNoise
from .urdf import LINK_ROLES, RobotGeometry, build_urdf


@dataclass(frozen=True)
class ContactSummary:
    collision: bool            # any non-support contact above the force threshold
    n_collision_points: int
    max_collision_force_n: float
    collision_links: Tuple[str, ...]
    support_force_n: float     # total normal force carried by wheels/casters


def _mn():
    import magnum as mn  # lazy: keep kznav importable without Habitat

    return mn


class WheeledRobot:
    def __init__(self, sim: Any, emb: Mapping[str, Any], workdir: Path, actuation_noise: ActuationNoise):
        import habitat_sim

        self.sim = sim
        self.emb = emb
        self.geom = RobotGeometry.from_config(emb)
        self.limiter = VelocityLimiter(DriveLimits.from_config(emb["drive"]["limits"]))
        self.noise = actuation_noise
        phys = emb["physics"]
        if phys["mode"] != "articulated":
            raise ValueError(f"physics.mode {phys['mode']!r} is not implemented (only 'articulated', D-003)")
        self.physics_dt = float(phys["physics_dt_s"])
        self.control_dt = float(phys["control_dt_s"])
        cts = phys["contacts"]
        self.min_force = float(cts["min_normal_force_n"])
        self.support_min_y = float(cts["support_normal_min_y"])

        workdir.mkdir(parents=True, exist_ok=True)
        urdf_path = workdir / "robot.urdf"
        urdf_path.write_text(build_urdf(emb), encoding="utf-8")
        self.ao = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(urdf_path), fixed_base=False)
        self.link_ids: Dict[str, int] = {self.ao.get_link_name(i): i for i in self.ao.get_link_ids()}
        self.link_names: Dict[int, str] = {i: n for n, i in self.link_ids.items()}
        self.link_names[-1] = "base_link"
        # Habitat creates default position motors for every DoF; they would fight the wheels.
        for mid in list(self.ao.existing_joint_motor_ids.keys()):
            self.ao.remove_joint_motor(mid)
        drive = emb["drive"]
        max_impulse = float(drive["max_wheel_torque_nm"]) * self.physics_dt
        gain = float(drive["motor_velocity_gain"])
        self.motor_ids = {
            side: self.ao.create_joint_motor(
                self.link_ids[f"{side}_wheel"], habitat_sim.physics.JointMotorSettings(0.0, 0.0, 0.0, gain, max_impulse)
            )
            for side in ("left", "right")
        }
        self.dof = {side: self.ao.get_link_dof_offset(self.link_ids[f"{side}_wheel"]) for side in ("left", "right")}
        for name, lid in self.link_ids.items():
            role = LINK_ROLES[name]
            mu = {"wheel": drive["wheel_friction"], "caster": drive["casters"]["friction"]}.get(role)
            if mu is not None:
                self.ao.set_link_friction(lid, float(mu))
        # Chassis friction comes from the URDF <contact> tag: set_link_friction(-1, ...) (the base
        # link) segfaults in habitat-sim 0.2.4.
        self.v_applied = 0.0
        self.w_applied = 0.0

    # ------------------------------------------------------------------ state
    @property
    def object_id(self) -> int:
        return int(self.ao.object_id)

    def _rotation_wxyz(self) -> Tuple[float, float, float, float]:
        q = self.ao.rotation
        return float(q.scalar), float(q.vector[0]), float(q.vector[1]), float(q.vector[2])

    def reference_position(self) -> np.ndarray:
        """World position of the reference point (floor under the chassis centre)."""
        mn = _mn()
        p = self.ao.transformation.transform_point(mn.Vector3(*self.geom.reference_in_base))
        return np.array([p[0], p[1], p[2]], dtype=np.float64)

    def pose2d(self) -> Tuple[float, float, float]:
        x, y = habitat_to_map(self.reference_position())
        return x, y, yaw_from_quat(self._rotation_wxyz())

    def tilt_deg(self) -> float:
        """Angle between the body's up axis and world up (bumps, ramps, high-centring)."""
        mn = _mn()
        up = self.ao.transformation.transform_vector(mn.Vector3(0.0, 1.0, 0.0))
        return math.degrees(math.acos(max(-1.0, min(1.0, float(up[1])))))

    def wheel_velocities(self) -> Tuple[float, float]:
        jv = self.ao.joint_velocities
        return float(jv[self.dof["left"]]), float(jv[self.dof["right"]])

    def odometry_twist(self) -> Tuple[float, float]:
        wl, wr = self.wheel_velocities()
        return body_twist(wl, wr, self.geom.wheel_base, self.geom.wheel_radius)

    def camera_transform(self, mount: Mapping[str, Any]):
        """World transform of the RGB-D optical centre (looks along -Z, pitched up by pitch_deg)."""
        mn = _mn()
        offset = self.geom.to_base((0.0, float(mount["height_m"]), -float(mount["forward_m"])))
        local = mn.Matrix4.translation(mn.Vector3(*offset)) @ mn.Matrix4.rotation_x(mn.Deg(float(mount["pitch_deg"])))
        return self.ao.transformation @ local

    # ------------------------------------------------------------------ control
    def place(self, reference_position: Sequence[float], yaw: float) -> None:
        """Put the reference point at ``reference_position`` (floor height) with map heading ``yaw``, at rest."""
        mn = _mn()
        w, x, y, z = quat_from_yaw(yaw)
        rot = mn.Quaternion(mn.Vector3(x, y, z), w)
        base = mn.Vector3(*reference_position) - rot.transform_vector(mn.Vector3(*self.geom.reference_in_base))
        self.ao.transformation = mn.Matrix4.from_(rot.to_matrix(), base)
        self.ao.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.ao.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.ao.joint_velocities = [0.0] * len(self.ao.joint_velocities)
        self.limiter.reset()
        self._set_wheel_targets(0.0, 0.0)
        self.v_applied = self.w_applied = 0.0

    def _set_wheel_targets(self, w_left: float, w_right: float) -> None:
        for side, target in (("left", w_left), ("right", w_right)):
            mid = self.motor_ids[side]
            s = self.ao.get_joint_motor_settings(mid)
            s.velocity_target = float(target)
            self.ao.update_joint_motor(mid, s)

    def command(self, v_cmd: float, w_cmd: float) -> Tuple[float, float]:
        """Apply one control tick's command; returns the (limited, noise-free) twist sent to the drive."""
        v, w = self.limiter.step(v_cmd, w_cmd, self.control_dt)
        v_act, w_act = self.noise.apply(v, w, self.control_dt)
        self._set_wheel_targets(*wheel_speeds(v_act, w_act, self.geom.wheel_base, self.geom.wheel_radius))
        self.v_applied, self.w_applied = v, w
        return v, w

    # ------------------------------------------------------------------ contacts
    def contacts(self) -> ContactSummary:
        me = self.object_id
        n_coll = 0
        max_f = 0.0
        links: List[str] = []
        support = 0.0
        for c in self.sim.get_physics_contact_points():
            if c.normal_force < self.min_force:
                continue
            if c.object_id_a == me and c.object_id_b != me:
                link, normal_into_robot = c.link_id_a, np.asarray(c.contact_normal_on_b_in_ws)
            elif c.object_id_b == me and c.object_id_a != me:
                link, normal_into_robot = c.link_id_b, -np.asarray(c.contact_normal_on_b_in_ws)
            else:
                continue
            name = self.link_names.get(int(link), "base_link")
            role = LINK_ROLES.get(name, "chassis")
            if role in ("wheel", "caster") and normal_into_robot[1] >= self.support_min_y:
                support += float(c.normal_force)
                continue
            n_coll += 1
            max_f = max(max_f, float(c.normal_force))
            if name not in links:
                links.append(name)
        return ContactSummary(n_coll > 0, n_coll, max_f, tuple(links), support)
