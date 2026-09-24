"""Standard ObjectNav episode metrics: success, SPL, SoftSPL, DTS, path length.

Definitions follow habitat-lab 0.2.4 (``habitat/tasks/nav/nav.py``,
``habitat/tasks/nav/object_nav_task.py``) so that numbers are comparable
with the literature:

* ``l`` — ``start_geodesic``: shortest navigable path from the start pose to
  the nearest goal target (for ObjectNav: the nearest goal *view point*),
  computed once at reset on the navmesh of the evaluated embodiment.
* ``p`` — ``path_length``: distance actually travelled, the sum of Euclidean
  distances between consecutive recorded positions.
* ``d_T`` — ``final_geodesic``: shortest navigable path from the final pose to
  the nearest goal target.
* success  = stop was called  AND  d_T < success_distance
  (strict inequality, as in habitat-lab ``Success``).
* SPL      = success * l / max(l, p)
* SoftSPL  = max(0, 1 - d_T / l) * l / max(l, p)
* DTS      = max(0, d_T - success_distance)   (distance to the success region)

Episodes with a non-positive or non-finite ``l`` are rejected with
``ValueError`` instead of silently producing NaN: such an episode is a dataset
or embodiment problem and must be reported, not averaged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def path_length(positions: Sequence[Sequence[float]]) -> float:
    """Length of the polyline through ``positions`` (N x D), in the same units.

    With continuous control the result depends on the sampling rate; positions
    must therefore be recorded at every control tick (see docs/measurement_protocol.md).
    """
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"positions must be an N x D array, got shape {pts.shape}")
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


@dataclass(frozen=True)
class SuccessRule:
    """How success is judged. Values come from the dataset/eval config."""

    success_distance: float
    #: If true, success additionally requires the goal to be visible in the
    #: final frame (``goal_visible_at_stop``). Standard HM3D ObjectNav encodes
    #: visibility in the view points instead, so this defaults to False.
    require_visible_at_stop: bool = False


@dataclass(frozen=True)
class EpisodeGeometry:
    """Everything the standard metrics need about one finished episode."""

    start_geodesic: float
    path_length: float
    final_geodesic: float
    called_stop: bool
    goal_visible_at_stop: Optional[bool] = None


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    spl: float
    soft_spl: float
    dts: float
    distance_to_goal: float
    path_length: float
    start_geodesic: float


def _check_start_geodesic(l: float) -> None:
    if not math.isfinite(l) or l <= 0.0:
        raise ValueError(f"start_geodesic must be finite and > 0, got {l!r} (invalid episode)")


def _efficiency(l: float, p: float) -> float:
    if p < 0.0 or math.isnan(p):
        raise ValueError(f"path_length must be >= 0, got {p!r}")
    return l / max(l, p)


def is_success(g: EpisodeGeometry, rule: SuccessRule) -> bool:
    if not g.called_stop:
        return False
    if not g.final_geodesic < rule.success_distance:  # also False for NaN / inf
        return False
    if rule.require_visible_at_stop:
        if g.goal_visible_at_stop is None:
            raise ValueError("success rule requires visibility but goal_visible_at_stop is unknown")
        return bool(g.goal_visible_at_stop)
    return True


def spl(success: bool, start_geodesic: float, path_len: float) -> float:
    _check_start_geodesic(start_geodesic)
    return float(success) * _efficiency(start_geodesic, path_len)


def soft_spl(start_geodesic: float, path_len: float, final_geodesic: float) -> float:
    _check_start_geodesic(start_geodesic)
    if math.isnan(final_geodesic):
        raise ValueError("final_geodesic is NaN")
    soft_success = max(0.0, 1.0 - final_geodesic / start_geodesic)  # 0 for inf (unreachable)
    return soft_success * _efficiency(start_geodesic, path_len)


def dts(final_geodesic: float, success_distance: float) -> float:
    """Distance to the success region; ``inf`` if the goal is unreachable from the final pose."""
    if math.isnan(final_geodesic):
        raise ValueError("final_geodesic is NaN")
    return max(0.0, final_geodesic - success_distance)


def compute_episode_metrics(g: EpisodeGeometry, rule: SuccessRule) -> EpisodeMetrics:
    ok = is_success(g, rule)
    return EpisodeMetrics(
        success=ok,
        spl=spl(ok, g.start_geodesic, g.path_length),
        soft_spl=soft_spl(g.start_geodesic, g.path_length, g.final_geodesic),
        dts=dts(g.final_geodesic, rule.success_distance),
        distance_to_goal=g.final_geodesic,
        path_length=g.path_length,
        start_geodesic=g.start_geodesic,
    )
