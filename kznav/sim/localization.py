"""What pose the policy and the controller are given (D-051).

``ground_truth`` (default, standard ObjectNav GPS+compass): the simulator's pose.
``odometry`` (robustness study; what a real robot-vacuum without SLAM has): the pose is dead-reckoned from the
known start pose. Each control tick's increment comes from

* ``wheel_encoders`` - the wheel joint velocities (``Tick.odom_twist``), so Bullet's wheel slip shows up as
  odometry error exactly as on a real robot; or
* ``true_increment`` - the true motion between ticks (the point agent, or perfect encoders),

and then passes through the odometry drift model (kznav.sim.noise). The policy receives the observation
re-expressed in its estimated frame: the planar rigid transform that maps the true pose onto the estimate is
applied to the camera transform and the reference position, so the map, candidates and plans all live in the
robot's own drifting frame. Everything the runner measures (metrics, funnel, stop labels) keeps the true pose.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .diff_drive import integrate_unicycle
from .frames import wrap_angle

Pose = Tuple[float, float, float]


def relative(a: Pose, b: Pose) -> Pose:
    """Pose b in the frame of pose a (dx forward, dy left, dyaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    ex, ey = b[0] - a[0], b[1] - a[1]
    return c * ex + s * ey, -s * ex + c * ey, wrap_angle(b[2] - a[2])


def compose(a: Pose, d: Pose) -> Pose:
    """Pose a moved by the robot-frame increment d."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * d[0] - s * d[1], a[1] + s * d[0] + c * d[1], wrap_angle(a[2] + d[2])


def frame_correction(true_pose: Pose, est_pose: Pose) -> np.ndarray:
    """4x4 Habitat-frame transform C with C * (true pose) = estimated pose: translate the true reference point to the
    origin, rotate about +Y by the heading error, translate to the estimate (map x = hab x, map y = -hab z)."""
    dth = est_pose[2] - true_pose[2]
    c, s = math.cos(dth), math.sin(dth)
    to_origin = np.eye(4)
    to_origin[0, 3], to_origin[2, 3] = -true_pose[0], true_pose[1]
    rot = np.eye(4)
    rot[0, 0], rot[0, 2], rot[2, 0], rot[2, 2] = c, s, -s, c
    to_est = np.eye(4)
    to_est[0, 3], to_est[2, 3] = est_pose[0], -est_pose[1]
    return to_est @ rot @ to_origin


class GroundTruthLocalizer:
    mode = "ground_truth"

    def __init__(self) -> None:
        self.pose: Optional[Pose] = None

    def reset(self, true_pose: Pose, seed: Optional[int] = None) -> None:
        self.pose = tuple(true_pose)

    def tick(self, t: Any) -> Pose:
        self.pose = tuple(t.pose)
        return self.pose

    def observation(self, obs: Any) -> Any:
        return obs

    def error(self, true_pose: Pose) -> Tuple[float, float]:
        return 0.0, 0.0


class OdometryLocalizer:
    mode = "odometry"

    def __init__(self, drift: Any, source: str = "wheel_encoders"):
        if source not in ("wheel_encoders", "true_increment"):
            raise ValueError("odometry source must be wheel_encoders | true_increment")
        self.drift, self.source = drift, source
        self.pose: Optional[Pose] = None
        self._last_true: Optional[Pose] = None
        self._last_t = 0.0

    def reset(self, true_pose: Pose, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.drift.reseed(seed)
        self.pose = tuple(true_pose)              # the start pose is known (as on a robot that starts at its dock)
        self._last_true = tuple(true_pose)
        self._last_t = 0.0

    def tick(self, t: Any) -> Pose:
        dt = max(0.0, float(t.t) - self._last_t)
        if self.source == "wheel_encoders" and getattr(t, "odom_twist", None) is not None and dt > 0:
            v, w = t.odom_twist
            x, y, yaw = integrate_unicycle(0.0, 0.0, 0.0, float(v), float(w), dt)
            inc = (x, y, yaw)
        else:
            inc = relative(self._last_true, tuple(t.pose))
        self.pose = compose(self.pose, self.drift.apply(*inc))
        self._last_true, self._last_t = tuple(t.pose), float(t.t)
        return self.pose

    def observation(self, obs: Any) -> Any:
        """The observation in the estimated frame (true pose of ``obs`` -> current estimate)."""
        C = frame_correction(obs.pose, self.pose)
        pos = C @ np.append(np.asarray(obs.position, dtype=np.float64), 1.0)
        return replace(obs, pose=tuple(self.pose), position=pos[:3], camera_transform=C @ np.asarray(obs.camera_transform))

    def error(self, true_pose: Pose) -> Tuple[float, float]:
        """(position error m, |heading error| rad) of the current estimate."""
        return (math.hypot(self.pose[0] - true_pose[0], self.pose[1] - true_pose[1]),
                abs(wrap_angle(self.pose[2] - true_pose[2])))


def make_localizer(emb: Mapping[str, Any], drift: Any):
    mode = str(emb.get("localization", "ground_truth"))
    if mode == "ground_truth":
        return GroundTruthLocalizer()
    if mode == "odometry":
        odo = (emb.get("sensors") or {}).get("odometry") or {}
        source = str(odo.get("source", "wheel_encoders" if emb.get("kind") == "wheeled" else "true_increment"))
        return OdometryLocalizer(drift, source)
    raise ValueError(f"embodiment.localization must be ground_truth | odometry, got {mode!r}")
