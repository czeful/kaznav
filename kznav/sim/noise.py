"""Sensor and actuation noise models (robustness study, D-051; all "none" for baseline runs).

Selected in ``configs/embodiment/*.yaml: noise.{depth,actuation,odometry_drift}.model``:

* depth ``redwood``: the Redwood / CLAMS depth-camera model as in habitat-sim's ``RedwoodDepthNoiseModel``
  (pixel jitter, 2x2 downsampling, per-pixel depth distortion from the measured model, disparity-domain noise,
  no return beyond 10 m) - re-implemented in numpy so that it draws from our seeded generator: habitat-sim's
  numba version uses the global RNG, which would break per-episode determinism (D-034). ``noise_multiplier``
  scales it as in habitat-sim. ``gaussian``: multiplicative N(0, rel_std) plus random dropout - for sweeps.
* actuation ``gaussian_twist`` (wheeled robot): the realised twist is v(1 + e_v), w(1 + e_w) + b_w, with
  e_v, e_w, b_w resampled every ``hold_s`` seconds (motor control error, uneven floor). Wheel slip itself is
  Bullet's, not a model.
* odometry drift ``gaussian_increment``: per increment in the robot frame, dx(1 + e_t), dy + N(0, lat_std |dx|),
  dyaw(1 + e_r) + N(0, yaw_per_m_std |dx|) - wheel-radius / wheel-base calibration error and heading drift.
  It only matters with ``localization: odometry`` (kznav.sim.localization).

Every model draws from its own stream and is re-seeded per episode (``reseed``), so an episode's noise depends on
(run seed, episode id) only.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from ..registry import Registry


class _Seeded:
    def __init__(self, spec: Mapping[str, Any], rng: np.random.Generator):
        self.spec, self.rng = dict(spec), rng

    def reseed(self, seed: int) -> None:
        """Start a new episode's stream (state such as held actuation errors is reset too)."""
        self.rng = np.random.default_rng(seed)
        self.reset_state()

    def reset_state(self) -> None:
        pass


class DepthNoise(_Seeded, ABC):
    @abstractmethod
    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Noisy copy of a metric depth image (H x W, metres; 0 = no return)."""


class ActuationNoise(_Seeded, ABC):
    @abstractmethod
    def apply(self, v: float, w: float, dt: float) -> Tuple[float, float]:
        """Velocities actually realised by the drive for commanded (v, w) over dt."""


class OdometryDrift(_Seeded, ABC):
    @abstractmethod
    def apply(self, dx: float, dy: float, dyaw: float) -> Tuple[float, float, float]:
        """Odometry increment as reported, given the measured increment in the robot frame."""


DEPTH_NOISE: Registry[DepthNoise] = Registry("depth noise", builtin_modules=["kznav.sim.noise"])
ACTUATION_NOISE: Registry[ActuationNoise] = Registry("actuation noise", builtin_modules=["kznav.sim.noise"])
ODOMETRY_DRIFT: Registry[OdometryDrift] = Registry("odometry drift", builtin_modules=["kznav.sim.noise"])


@DEPTH_NOISE.register("none")
class NoDepthNoise(DepthNoise):
    def apply(self, depth: np.ndarray) -> np.ndarray:
        return depth


@ACTUATION_NOISE.register("none")
class NoActuationNoise(ActuationNoise):
    def apply(self, v: float, w: float, dt: float) -> Tuple[float, float]:
        return v, w


@ODOMETRY_DRIFT.register("none")
class NoOdometryDrift(OdometryDrift):
    def apply(self, dx: float, dy: float, dyaw: float) -> Tuple[float, float, float]:
        return dx, dy, dyaw


# ---------------------------------------------------------------------------- depth
def redwood_model_path() -> Optional[Path]:
    """habitat-sim's measured distortion model (80 x 80 x 5), if habitat-sim is installed."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("habitat_sim.sensors.noise_models")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    p = Path(list(spec.submodule_search_locations)[0]) / "data" / "redwood-depth-dist-model.npy"
    return p if p.exists() else None


def redwood_simulate(gt: np.ndarray, model: np.ndarray, noise_multiplier: float, rng: np.random.Generator) -> np.ndarray:
    """Vectorised habitat-sim ``_simulate`` (redwood_depth_noise_model.py) with an explicit generator."""
    gt = np.asarray(gt, dtype=np.float32)
    h, w = gt.shape
    r = rng.standard_normal((h, w, 3)).astype(np.float32)
    jj, ii = np.mgrid[0:h, 0:w]
    m = float(noise_multiplier)
    y = (np.clip(jj + r[..., 0] * 0.25 * m, 0.0, h - 1.0) + 0.5).astype(np.int64)
    x = (np.clip(ii + r[..., 1] * 0.25 * m, 0.0, w - 1.0) + 0.5).astype(np.int64)
    d = gt[y - y % 2, x - x % 2]
    xm = (x / max(1.0, w - 1.0) * 639.0 + 0.5).astype(np.int64) // 8
    ym = (y / max(1.0, h - 1.0) * 479.0 + 0.5).astype(np.int64) // 6
    i2 = ((d + 1.0) / 2.0).astype(np.int64)
    i1 = i2 - 1
    a = (d - (i1 * 2.0 + 1.0)) / 2.0
    f = (1.0 - a) * model[ym, xm, np.clip(i1, 0, 4)] + a * model[ym, xm, np.minimum(i2, 4)]
    with np.errstate(divide="ignore", invalid="ignore"):
        und = np.where(f < 1e-5, 0.0, d / f)
        denom = np.round((35.130 / und + r[..., 2] * 0.027778 * m) * 8.0)
        out = np.where(denom <= 1e-5, 0.0, 35.130 * 8.0 / denom)
    out[(d >= 10.0) | (und == 0.0) | ~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


@DEPTH_NOISE.register("redwood")
class RedwoodDepthNoise(DepthNoise):
    def __init__(self, spec: Mapping[str, Any], rng: np.random.Generator):
        super().__init__(spec, rng)
        path = spec.get("model_file") or redwood_model_path()
        if path is None:
            raise FileNotFoundError("redwood depth noise needs habitat-sim's redwood-depth-dist-model.npy "
                                    "(or noise.depth.model_file)")
        self.model = np.load(path).reshape(80, 80, 5).astype(np.float32)
        self.multiplier = float(spec.get("noise_multiplier", 1.0))

    def apply(self, depth: np.ndarray) -> np.ndarray:
        return redwood_simulate(depth, self.model, self.multiplier, self.rng)


@DEPTH_NOISE.register("gaussian")
class GaussianDepthNoise(DepthNoise):
    def apply(self, depth: np.ndarray) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float32)
        out = d * (1.0 + float(self.spec.get("rel_std", 0.01)) * self.rng.standard_normal(d.shape).astype(np.float32))
        drop = float(self.spec.get("dropout", 0.0))
        if drop > 0:
            out[self.rng.random(d.shape) < drop] = 0.0
        out[d <= 0] = 0.0
        return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------- actuation and odometry
@ACTUATION_NOISE.register("gaussian_twist")
class GaussianTwistNoise(ActuationNoise):
    def __init__(self, spec: Mapping[str, Any], rng: np.random.Generator):
        super().__init__(spec, rng)
        self.reset_state()

    def reset_state(self) -> None:
        self._left = 0.0
        self._ev = self._ew = self._bw = 0.0

    def apply(self, v: float, w: float, dt: float) -> Tuple[float, float]:
        if self._left <= 1e-9:                     # tolerance: 10 x 0.1 s leaves ~1e-16 of a 1 s hold
            s = self.spec
            self._ev = float(s.get("v_rel_std", 0.05)) * float(self.rng.standard_normal())
            self._ew = float(s.get("w_rel_std", 0.05)) * float(self.rng.standard_normal())
            self._bw = float(s.get("w_bias_std", 0.01)) * float(self.rng.standard_normal())
            self._left = float(s.get("hold_s", 0.5))
        self._left -= dt
        moving = abs(v) > 1e-9 or abs(w) > 1e-9
        return v * (1.0 + self._ev), (w * (1.0 + self._ew) + self._bw) if moving else w


@ODOMETRY_DRIFT.register("gaussian_increment")
class GaussianIncrementDrift(OdometryDrift):
    def apply(self, dx: float, dy: float, dyaw: float) -> Tuple[float, float, float]:
        s, g = self.spec, self.rng
        dist = math.hypot(dx, dy)
        dx2 = dx * (1.0 + float(s.get("trans_rel_std", 0.02)) * float(g.standard_normal()))
        dy2 = dy + float(s.get("lat_std_per_m", 0.01)) * dist * float(g.standard_normal())
        dyaw2 = dyaw * (1.0 + float(s.get("rot_rel_std", 0.02)) * float(g.standard_normal())) \
            + math.radians(float(s.get("yaw_deg_per_m_std", 1.0))) * dist * float(g.standard_normal())
        return dx2, dy2, dyaw2


def _build(reg: Registry, spec: Mapping[str, Any], rng: np.random.Generator):
    spec = dict(spec)
    spec.setdefault("type", spec.get("model", "none"))
    return reg.resolve(spec)(spec, rng)


def noise_streams(seed: int):
    """Three independent generators (depth, actuation, odometry) from one seed."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed) % (2 ** 63)).spawn(3)]


def build_noise_models(noise_cfg: Mapping[str, Any], seed: int):
    """(depth, actuation, odometry) models from the embodiment's ``noise`` block, each with its own RNG stream."""
    r_depth, r_act, r_odo = noise_streams(seed)
    return (
        _build(DEPTH_NOISE, noise_cfg.get("depth", {}), r_depth),
        _build(ACTUATION_NOISE, noise_cfg.get("actuation", {}), r_act),
        _build(ODOMETRY_DRIFT, noise_cfg.get("odometry_drift", {}), r_odo),
    )


def reseed_noise_models(models, seed: int) -> None:
    """Per-episode streams for (depth, actuation, odometry)."""
    for m, g in zip(models, noise_streams(seed)):
        m.rng = g
        m.reset_state()
