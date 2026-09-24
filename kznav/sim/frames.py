"""Conversions between Habitat's 3D frame and the 2D map frame used by perception and planning.

Habitat: +Y up, an unrotated agent looks along -Z, +X is to its right.
Map (2D): x_map = x_h, y_map = -z_h (so "forward" of an unrotated agent is +y_map),
yaw measured counter-clockwise from +x_map. An unrotated agent therefore has yaw = +pi/2.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def habitat_to_map(p: Sequence[float]) -> Tuple[float, float]:
    return float(p[0]), float(-p[2])


def map_to_habitat(x: float, y: float, height: float = 0.0) -> np.ndarray:
    return np.array([x, height, -y], dtype=np.float64)


def quat_rotate(q_wxyz: Sequence[float], v: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(c) for c in q_wxyz)
    qv = np.array([x, y, z])
    vv = np.asarray(v, dtype=float)
    t = 2.0 * np.cross(qv, vv)
    return vv + w * t + np.cross(qv, t)


def yaw_from_quat(q_wxyz: Sequence[float]) -> float:
    """Map-frame yaw of the body's forward (-Z) axis; independent of roll/pitch for small tilts."""
    fwd = quat_rotate(q_wxyz, (0.0, 0.0, -1.0))
    return math.atan2(-fwd[2], fwd[0])


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """(w, x, y, z) rotation about +Y giving map-frame heading ``yaw``."""
    theta = yaw - math.pi / 2  # rotation about +Y from the default heading (+pi/2)
    return math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
