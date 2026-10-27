"""Path-following controller for a differential-drive robot (pure pursuit with rotate-in-place).

Input: a polyline path in the 2D map frame and the current pose (x, y, yaw). Output: a (v, w)
command; speed and acceleration limits are applied downstream by the drive's limiter, so the
controller only has to respect the speed caps it is configured with.

Rules
* goal reached when within ``goal_tolerance_m`` of the last point -> (0, 0)
* heading error to the look-ahead point > ``rotate_in_place_rad`` -> turn on the spot
* otherwise pure pursuit: curvature k = 2 sin(alpha) / L_d, v slowed on sharp curvature and
  near the goal, w = k * v (capped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PurePursuitParams:
    """Controller parameters (``policy.controller``): look-ahead, goal tolerance, speed caps, rotate-in-place angle."""
    lookahead_m: float
    goal_tolerance_m: float
    max_lin_vel: float
    max_ang_vel: float
    rotate_in_place_rad: float
    slowdown_radius_m: float
    min_lin_vel: float

    @classmethod
    def from_config(cls, c: Mapping[str, Any]) -> "PurePursuitParams":
        """Build from the ``policy.controller`` config block."""
        return cls(**{k: float(c[k]) for k in cls.__dataclass_fields__})


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PurePursuit:
    """Pure-pursuit path follower with rotate-in-place for a differential-drive robot (see the module docstring)."""
    def __init__(self, params: PurePursuitParams):
        self.p = params
        self.path = np.zeros((0, 2))
        self._seg = 0

    def set_path(self, points_xy: Sequence[Sequence[float]]) -> None:
        """Follow a new polyline path (map frame) from its first segment.

        Raises:
            ValueError: If the path is empty.
        """
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            raise ValueError("empty path")
        self.path = pts
        self._seg = 0

    def done(self, pose: Tuple[float, float, float]) -> bool:
        """Whether the pose is within the goal tolerance of the path's last point."""
        return bool(np.hypot(*(self.path[-1] - pose[:2])) <= self.p.goal_tolerance_m)

    def _lookahead_point(self, xy: np.ndarray) -> np.ndarray:
        """First point along the path (from the current segment on) at distance >= L_d from xy."""
        path = self.path
        if len(path) == 1:
            return path[0]
        # advance the segment index to the closest segment (monotone, so the robot never backtracks)
        best, best_d = self._seg, math.inf
        for i in range(self._seg, len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            t = float(np.clip(np.dot(xy - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
            d = float(np.hypot(*(a + t * ab - xy)))
            if d < best_d:
                best, best_d = i, d
        self._seg = best
        L = self.p.lookahead_m
        for i in range(self._seg, len(path) - 1):
            a, b = path[i], path[i + 1]
            # intersection of the circle |p - xy| = L with segment a + t (b - a), largest t in [0, 1]
            d = b - a
            f = a - xy
            A, B, C = np.dot(d, d), 2 * np.dot(f, d), np.dot(f, f) - L * L
            disc = B * B - 4 * A * C
            if A > 1e-12 and disc >= 0:
                t = (-B + math.sqrt(disc)) / (2 * A)
                if 0.0 <= t <= 1.0:
                    return a + t * d
        return path[-1]

    def command(self, pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """Velocity command ``(v, w)`` for the current pose ``(x, y, yaw)``; ``(0, 0)`` at the goal."""
        x, y, yaw = pose
        xy = np.array([x, y])
        if self.done(pose):
            return 0.0, 0.0
        target = self._lookahead_point(xy)
        dx, dy = target - xy
        alpha = _wrap(math.atan2(dy, dx) - yaw)
        p = self.p
        if abs(alpha) > p.rotate_in_place_rad:
            return 0.0, math.copysign(p.max_ang_vel, alpha)
        dist_goal = float(np.hypot(*(self.path[-1] - xy)))
        L = max(float(np.hypot(dx, dy)), 1e-6)
        k = 2.0 * math.sin(alpha) / L
        v = p.max_lin_vel * min(1.0, dist_goal / p.slowdown_radius_m)
        if abs(k) > 1e-9:
            v = min(v, p.max_ang_vel / abs(k))
        v = max(v, p.min_lin_vel)
        w = max(-p.max_ang_vel, min(p.max_ang_vel, k * v))
        return v, w
