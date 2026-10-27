"""Kinematic wheeled base on its own navmesh (``embodiment.physics.mode: kinematic``, D-053).

The wheeled robot's body, camera and navmesh, without Bullet contacts: each control tick the commanded twist goes
through the same speed / acceleration limiter (and actuation noise model) as the articulated robot, is integrated as
a unicycle, and the move is constrained to the robot's navmesh with ``pathfinder.try_step`` - the robot slides
along walls instead of pushing into them, and it cannot leave its navmesh (no drives off steps, no high-centring on
scan noise, no falls). HM3D's scanned meshes are not physically accurate (holes, bumpy floors, soft stair edges),
so with Bullet the main table would measure the mesh as much as the method (D-052). The articulated robot stays for
the robustness study and the transfer to the real robot.

Odometry reports the realised twist (what wheel encoders measure; when the robot slides along a wall the wheels
still turn, so odometry over-estimates progress - as on a real robot).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

import numpy as np

from .diff_drive import DriveLimits, VelocityLimiter, integrate_unicycle
from .frames import habitat_to_map, map_to_habitat
from .noise import ActuationNoise

SLIDE_TOLERANCE_M = 1e-3          # try_step result farther than this from the target: the move was blocked (collision)


class KinematicBase:
    """The wheeled robot's base moved kinematically on its navmesh (see the module docstring).

    Args:
        pathfinder: The robot's navmesh.
        emb: The embodiment config (drive limits and control period are read).
        actuation_noise: Actuation-noise model applied to every command.
    """
    def __init__(self, pathfinder: Any, emb: Mapping[str, Any], actuation_noise: ActuationNoise):
        self.pathfinder = pathfinder
        self.limiter = VelocityLimiter(DriveLimits.from_config(emb["drive"]["limits"]))
        self.noise = actuation_noise
        self.control_dt = float(emb["physics"]["control_dt_s"])
        self.x = self.y = self.yaw = self.h = 0.0
        self.v_applied = self.w_applied = 0.0

    def place(self, reference_position: Any, yaw: float) -> None:
        """Put the reference point on the navmesh at ``reference_position`` (snapped) with map heading ``yaw``."""
        p = np.asarray(self.pathfinder.snap_point(np.asarray(reference_position, dtype=np.float32)), dtype=np.float64)
        if not np.all(np.isfinite(p)):
            p = np.asarray(reference_position, dtype=np.float64)
        self.x, self.y = habitat_to_map(p)
        self.h = float(p[1])
        self.yaw = float(yaw)
        self.limiter.reset()
        self.v_applied = self.w_applied = 0.0

    def step(self, v_cmd: float, w_cmd: float) -> Tuple[float, float, bool]:
        """One control tick. Returns the limited (noise-free) twist and whether the move was blocked."""
        v, w = self.limiter.step(v_cmd, w_cmd, self.control_dt)
        v_act, w_act = self.noise.apply(v, w, self.control_dt)
        nx, ny, nyaw = integrate_unicycle(self.x, self.y, self.yaw, v_act, w_act, self.control_dt)
        start = map_to_habitat(self.x, self.y, self.h).astype(np.float32)
        target = map_to_habitat(nx, ny, self.h).astype(np.float32)
        end = np.asarray(self.pathfinder.try_step(start, target), dtype=np.float64)
        # Blocked: the command stays (a stalled motor keeps its setpoint), so the executor's no-progress check sees
        # forward command without displacement and backs off, as with the articulated robot.
        blocked = math.hypot(end[0] - target[0], end[2] - target[2]) > SLIDE_TOLERANCE_M
        self.x, self.y = habitat_to_map(end)
        self.h = float(end[1])
        self.yaw = math.atan2(math.sin(nyaw), math.cos(nyaw))
        self.v_applied, self.w_applied = v_act, w_act
        return v, w, blocked

    def reference_position(self) -> np.ndarray:
        """Habitat world position of the reference point (floor under the chassis centre)."""
        return map_to_habitat(self.x, self.y, self.h)

    def pose2d(self) -> Tuple[float, float, float]:
        """Map-frame pose ``(x, y, yaw)``."""
        return self.x, self.y, self.yaw

    def odometry_twist(self) -> Tuple[float, float]:
        """Twist realised in the last tick, as wheel encoders would report it."""
        return self.v_applied, self.w_applied
