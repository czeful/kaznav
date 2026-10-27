"""Differential-drive kinematics and command limiting (pure numpy, no simulator).

Conventions: v > 0 drives forward, w > 0 turns left (counter-clockwise seen from above).
Wheel angular velocities are positive when they drive the robot forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class DriveLimits:
    """Speed and acceleration limits of the drive (m/s, rad/s, m/s^2, rad/s^2)."""
    max_lin_vel: float
    max_ang_vel: float
    max_lin_acc: float
    max_ang_acc: float

    @classmethod
    def from_config(cls, limits: Mapping[str, Any]) -> "DriveLimits":
        """Build from the ``embodiment.drive.limits`` config block."""
        return cls(
            max_lin_vel=float(limits["max_lin_vel_mps"]),
            max_ang_vel=float(limits["max_ang_vel_radps"]),
            max_lin_acc=float(limits["max_lin_acc_mps2"]),
            max_ang_acc=float(limits["max_ang_acc_radps2"]),
        )


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class VelocityLimiter:
    """Clamp commanded (v, w) to the speed limits, then rate-limit to the acceleration limits."""

    def __init__(self, limits: DriveLimits):
        self.limits = limits
        self.v = 0.0
        self.w = 0.0

    def reset(self) -> None:
        """Bring the limited twist back to rest (start of an episode)."""
        self.v = self.w = 0.0

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> Tuple[float, float]:
        """Advance one control tick and return the limited twist ``(v, w)``.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        L = self.limits
        v_t = _clamp(v_cmd, -L.max_lin_vel, L.max_lin_vel)
        w_t = _clamp(w_cmd, -L.max_ang_vel, L.max_ang_vel)
        self.v += _clamp(v_t - self.v, -L.max_lin_acc * dt, L.max_lin_acc * dt)
        self.w += _clamp(w_t - self.w, -L.max_ang_acc * dt, L.max_ang_acc * dt)
        return self.v, self.w


def wheel_speeds(v: float, w: float, wheel_base: float, wheel_radius: float) -> Tuple[float, float]:
    """(left, right) wheel angular velocities in rad/s for a body twist (v, w)."""
    return (v - w * wheel_base / 2) / wheel_radius, (v + w * wheel_base / 2) / wheel_radius


def body_twist(w_left: float, w_right: float, wheel_base: float, wheel_radius: float) -> Tuple[float, float]:
    """Inverse of ``wheel_speeds``: (v, w) from wheel angular velocities."""
    return wheel_radius * (w_left + w_right) / 2, wheel_radius * (w_right - w_left) / wheel_base


def integrate_unicycle(x: float, y: float, yaw: float, v: float, w: float, dt: float) -> Tuple[float, float, float]:
    """Exact constant-twist integration in the 2D map frame (x right, y forward-at-yaw-90°, yaw CCW)."""
    if abs(w) < 1e-9:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    r = v / w
    yaw2 = yaw + w * dt
    return x + r * (math.sin(yaw2) - math.sin(yaw)), y - r * (math.cos(yaw2) - math.cos(yaw)), yaw2
