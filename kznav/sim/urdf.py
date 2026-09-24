"""Generate the robot URDF from the embodiment config (no geometry lives in hand-edited files).

Frames
------
*Robot frame*: origin on the floor under the chassis centre, +Y up, forward -Z, right +X
(Habitat's convention; URDF rotations are applied in this frame as written).

*base_link frame*: origin at the centre of mass of the chassis. Habitat/Bullet place an
articulated object's root frame at the base link's INERTIAL frame, so a URDF whose base
inertial origin is not at the link origin shifts every collision shape by that offset
(found in stage 3: the robot spawned nose-down inside the floor). We therefore put the
base_link origin at the CoM and express all geometry relative to it.

``RobotGeometry.com`` is the CoM in the robot frame; the robot's reference point (used for
distances and path length) is the robot-frame origin, i.e. ``-com`` in the base frame.

Links:
  base_link      chassis cylinder + camera housing box
  left_wheel     continuous joint about -X (positive joint velocity drives forward)
  right_wheel    idem
  front_caster   fixed low-friction sphere touching the floor
  rear_caster    fixed low-friction sphere ``rear_lift_m`` above the floor (anti-tip)

URDF cylinders are aligned with the link's local Z, so the chassis collision is rotated
by 90° about X (vertical) and the wheels by 90° about Y (lateral axis).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

HALF_PI = math.pi / 2


@dataclass(frozen=True)
class RobotGeometry:
    """Derived geometry, all in metres / kilograms, in the base_link frame unless noted."""

    body_radius: float
    body_height: float
    clearance: float
    wheel_radius: float
    wheel_width: float
    wheel_base: float
    wheel_mass: float
    caster_radius: float
    caster_mass: float
    front_offset: float
    rear_offset: float
    rear_lift: float
    chassis_mass: float
    com_forward: float
    housing_size: Tuple[float, float, float]
    cam_height: float
    cam_forward: float

    @classmethod
    def from_config(cls, emb: Mapping[str, Any]) -> "RobotGeometry":
        body, drive, cas = emb["body"], emb["drive"], emb["drive"]["casters"]
        mount = emb["sensors"]["rgbd"]["mount"]
        wheels = 2 * float(drive["wheel_mass_kg"]) + 2 * float(cas["mass_kg"])
        chassis = float(body["mass_kg"]) - wheels
        if chassis <= 0:
            raise ValueError("body.mass_kg must exceed the wheel and caster masses")
        g = cls(
            body_radius=float(body["radius_m"]), body_height=float(body["height_m"]),
            clearance=float(body["ground_clearance_m"]),
            wheel_radius=float(drive["wheel_radius_m"]), wheel_width=float(drive["wheel_width_m"]),
            wheel_base=float(drive["wheel_base_m"]), wheel_mass=float(drive["wheel_mass_kg"]),
            caster_radius=float(cas["radius_m"]), caster_mass=float(cas["mass_kg"]),
            front_offset=float(cas["front_offset_m"]), rear_offset=float(cas["rear_offset_m"]),
            rear_lift=float(cas["rear_lift_m"]), chassis_mass=chassis,
            com_forward=float(body["com_forward_offset_m"]),
            housing_size=tuple(float(v) for v in emb["camera_housing"]["size_m"]),  # type: ignore[arg-type]
            cam_height=float(mount["height_m"]), cam_forward=float(mount["forward_m"]),
        )
        g.check()
        return g

    def check(self) -> None:
        if self.clearance >= self.body_height:
            raise ValueError("ground clearance must be below the body height")
        if self.wheel_base / 2 + self.wheel_width / 2 > self.body_radius:
            raise ValueError("wheels stick out of the chassis footprint")
        for off in (self.front_offset, self.rear_offset):
            if off + self.caster_radius > self.body_radius:
                raise ValueError("caster sticks out of the chassis footprint")
        if self.cam_forward > self.body_radius:
            raise ValueError("camera mounted outside the chassis footprint")

    @property
    def chassis_length(self) -> float:
        return self.body_height - self.clearance

    @property
    def com(self) -> Tuple[float, float, float]:
        """Centre of mass of the chassis in the robot frame (= base_link origin)."""
        return (0.0, self.clearance + self.chassis_length / 2, -self.com_forward)

    def to_base(self, p_robot: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Robot-frame point -> base_link frame."""
        c = self.com
        return (p_robot[0] - c[0], p_robot[1] - c[1], p_robot[2] - c[2])

    @property
    def reference_in_base(self) -> Tuple[float, float, float]:
        """The robot reference point (floor under the chassis centre) in the base_link frame."""
        return self.to_base((0.0, 0.0, 0.0))

    @property
    def top_height(self) -> float:
        """Highest point of the robot above the floor (chassis or camera housing)."""
        return max(self.body_height, self.cam_height + self.housing_size[1] / 2)


def _fmt(*v: float) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _origin(parent: ET.Element, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", xyz=_fmt(*xyz), rpy=_fmt(*rpy))


def _inertial(link: ET.Element, mass: float, ixx: float, iyy: float, izz: float, xyz=(0.0, 0.0, 0.0)) -> None:
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, xyz)
    ET.SubElement(inertial, "mass", value=_fmt(mass))
    ET.SubElement(inertial, "inertia", ixx=_fmt(ixx), ixy="0", ixz="0", iyy=_fmt(iyy), iyz="0", izz=_fmt(izz))


def _collision(link: ET.Element, geom: str, attrs: Mapping[str, str], xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    col = ET.SubElement(link, "collision")
    _origin(col, xyz, rpy)
    g = ET.SubElement(col, "geometry")
    ET.SubElement(g, geom, **attrs)


def _contact(link: ET.Element, friction: float) -> None:
    c = ET.SubElement(link, "contact")
    ET.SubElement(c, "lateral_friction", value=_fmt(friction))


def build_urdf(emb: Mapping[str, Any], name: str = "kznav_robot") -> str:
    g = RobotGeometry.from_config(emb)
    drive = emb["drive"]
    robot = ET.Element("robot", name=name)

    # --- chassis (base_link origin = CoM) ------------------------------------------
    base = ET.SubElement(robot, "link", name="base_link")
    r, h, m = g.body_radius, g.chassis_length, g.chassis_mass
    i_side = m * (3 * r * r + h * h) / 12.0
    _inertial(base, m, i_side, m * r * r / 2.0, i_side)
    _contact(base, float(emb["physics"]["chassis_friction"]))
    _collision(base, "cylinder", {"radius": _fmt(r), "length": _fmt(h)},
               g.to_base((0.0, g.clearance + h / 2, 0.0)), (HALF_PI, 0.0, 0.0))
    hx, hy, hz = g.housing_size
    _collision(base, "box", {"size": _fmt(hx, hy, hz)}, g.to_base((0.0, g.cam_height, -g.cam_forward)))

    # --- driven wheels -------------------------------------------------------------
    wr, ww, wm = g.wheel_radius, g.wheel_width, g.wheel_mass
    i_axis = wm * wr * wr / 2.0
    i_diam = wm * (3 * wr * wr + ww * ww) / 12.0
    for side, x in (("left", -g.wheel_base / 2), ("right", g.wheel_base / 2)):
        link = ET.SubElement(robot, "link", name=f"{side}_wheel")
        _inertial(link, wm, i_axis, i_diam, i_diam)
        _contact(link, float(drive["wheel_friction"]))
        shape = str(drive.get("wheel_collision_shape", "cylinder"))
        if shape == "cylinder":
            _collision(link, "cylinder", {"radius": _fmt(wr), "length": _fmt(ww)}, rpy=(0.0, HALF_PI, 0.0))
        elif shape == "sphere":
            # Smoother contact on triangle meshes (no cylinder rim catching internal triangle edges).
            _collision(link, "sphere", {"radius": _fmt(wr)})
        else:
            raise ValueError(f"wheel_collision_shape must be cylinder|sphere, got {shape!r}")
        joint = ET.SubElement(robot, "joint", name=f"{side}_wheel_joint", type="continuous")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=f"{side}_wheel")
        _origin(joint, g.to_base((x, wr, 0.0)))
        ET.SubElement(joint, "axis", xyz="-1 0 0")  # positive joint velocity = forward (-Z)
        ET.SubElement(joint, "dynamics", damping="0", friction="0")

    # --- casters -------------------------------------------------------------------
    cr, cm = g.caster_radius, g.caster_mass
    i_c = 2.0 * cm * cr * cr / 5.0
    for which, z, lift in (("front", -g.front_offset, 0.0), ("rear", g.rear_offset, g.rear_lift)):
        link = ET.SubElement(robot, "link", name=f"{which}_caster")
        _inertial(link, cm, i_c, i_c, i_c)
        _contact(link, float(drive["casters"]["friction"]))
        _collision(link, "sphere", {"radius": _fmt(cr)})
        joint = ET.SubElement(robot, "joint", name=f"{which}_caster_joint", type="fixed")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=f"{which}_caster")
        _origin(joint, g.to_base((0.0, cr + lift, z)))

    ET.indent(robot)
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")


LINK_ROLES = {
    "base_link": "chassis",
    "left_wheel": "wheel",
    "right_wheel": "wheel",
    "front_caster": "caster",
    "rear_caster": "caster",
}
