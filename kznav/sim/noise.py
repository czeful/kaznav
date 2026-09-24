"""Sensor and actuation noise interfaces (robustness study; all disabled for baseline runs).

Selected in ``configs/embodiment/*.yaml: noise.{depth,actuation,odometry_drift}.model``.
Only the null models exist now; Redwood depth noise (habitat-sim ``RedwoodDepthNoiseModel``),
LoCoBot actuation noise (habitat-lab) and an odometry drift model plug in here later
without touching the simulator wrapper or the policy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple

import numpy as np

from ..registry import Registry


class DepthNoise(ABC):
    def __init__(self, spec: Mapping[str, Any], rng: np.random.Generator):
        self.spec, self.rng = dict(spec), rng

    @abstractmethod
    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Noisy copy of a metric depth image (H x W, metres)."""


class ActuationNoise(ABC):
    def __init__(self, spec: Mapping[str, Any], rng: np.random.Generator):
        self.spec, self.rng = dict(spec), rng

    @abstractmethod
    def apply(self, v: float, w: float, dt: float) -> Tuple[float, float]:
        """Velocities actually realised by the drive for commanded (v, w) over dt."""


class OdometryDrift(ABC):
    def __init__(self, spec: Mapping[str, Any], rng: np.random.Generator):
        self.spec, self.rng = dict(spec), rng

    @abstractmethod
    def apply(self, dx: float, dy: float, dyaw: float) -> Tuple[float, float, float]:
        """Odometry increment as reported, given the true increment in the robot frame."""


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


def _build(reg: Registry, spec: Mapping[str, Any], rng: np.random.Generator):
    spec = dict(spec)
    spec.setdefault("type", spec.get("model", "none"))
    return reg.resolve(spec)(spec, rng)


def build_noise_models(noise_cfg: Mapping[str, Any], seed: int):
    """(depth, actuation, odometry) models from the embodiment's ``noise`` block, each with its own RNG stream."""
    ss = np.random.SeedSequence(seed)
    r_depth, r_act, r_odo = (np.random.default_rng(s) for s in ss.spawn(3))
    return (
        _build(DEPTH_NOISE, noise_cfg.get("depth", {}), r_depth),
        _build(ACTUATION_NOISE, noise_cfg.get("actuation", {}), r_act),
        _build(ODOMETRY_DRIFT, noise_cfg.get("odometry_drift", {}), r_odo),
    )
