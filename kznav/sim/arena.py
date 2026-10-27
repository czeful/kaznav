"""Primitive-box arenas for controlled physics validation (flat floor, walls, steps, slabs, loose boxes).

Two Habitat quirks handled here (both found while building stage 3):
* the ``cubeSolid`` primitive spans [-1, 1]^3, so ``scale`` = half extents;
* a STATIC object's transform must be set BEFORE switching it to STATIC, otherwise its
  collision shape stays where it was created (the origin).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Box:
    """One ``cubeSolid`` box of an arena: centre and half extents in metres (Habitat world frame), static or dynamic."""
    center: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]
    static: bool = True
    mass_kg: float = 1.0
    friction: float = 1.0
    name: str = "box"


def add_box(sim: Any, box: Box):
    """Add a box to the simulator and return the rigid object.

    The transform is set before the motion type, so a static box's collision shape is where the box is (see
    the module docstring).
    """
    import magnum as mn
    from habitat_sim.physics import MotionType

    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    tpl = otm.get_template_by_handle(otm.get_template_handles("cubeSolid")[0])
    tpl.scale = mn.Vector3(*box.half_extents)
    tpl.friction_coefficient = float(box.friction)
    tpl.mass = float(box.mass_kg)
    handle = f"kznav_{box.name}_{rom.get_num_objects()}"
    otm.register_template(tpl, handle)
    obj = rom.add_object_by_template_handle(handle)
    obj.translation = mn.Vector3(*box.center)
    obj.motion_type = MotionType.STATIC if box.static else MotionType.DYNAMIC
    return obj


def floor(size: float = 20.0, thickness: float = 0.1, friction: float = 1.0) -> Box:
    """Top surface at y = 0."""
    return Box((0.0, -thickness / 2, 0.0), (size / 2, thickness / 2, size / 2), friction=friction, name="floor")


def populate(boxes: Sequence[Box]):
    """A ``SceneSpec.populate`` callback that adds ``boxes`` (use with scene_id="NONE")."""

    def _populate(sim: Any) -> None:
        for b in boxes:
            add_box(sim, b)

    return _populate
