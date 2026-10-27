"""Pinhole camera model matching Habitat's CameraSensor.

Habitat depth is *planar* (z-) depth along the optical axis; the camera looks along its local
-Z with +Y up and +X right; square pixels, so fy = fx. Pixel centres are at (u + 0.5, v + 0.5)
— the convention that reproduced the mounted camera height to 0.1 mm in the stage-3 check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels: focal lengths, principal point and image size."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, hfov_deg: float, width: int, height: int) -> "Intrinsics":
        """Intrinsics of a camera with square pixels from its horizontal field of view and image size."""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx, fx, width / 2.0, height / 2.0, int(width), int(height))

    def project(self, p_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Camera-frame points (N, 3) -> pixel coordinates (u, v); points must have z < 0."""
        z = -p_cam[:, 2]
        u = self.fx * p_cam[:, 0] / z + self.cx - 0.5
        v = -self.fy * p_cam[:, 1] / z + self.cy - 0.5
        return u, v


def pixel_grid(K: Intrinsics, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the sampled pixels (centre of each stride x stride block)."""
    off = stride // 2
    v, u = np.mgrid[off : K.height : stride, off : K.width : stride]
    return v.ravel(), u.ravel()


def backproject(depth: np.ndarray, K: Intrinsics, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Camera-frame 3D points (N, 3) for pixels (v, u) with planar depth ``depth[v, u]``."""
    z = depth[v, u].astype(np.float64)
    x = (u + 0.5 - K.cx) / K.fx * z
    y = -(v + 0.5 - K.cy) / K.fy * z
    return np.stack([x, y, -z], axis=1)


def to_world(p_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Transform camera-frame points (N, 3) to the world frame with the 4x4 camera pose ``T_world_cam``."""
    return p_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
