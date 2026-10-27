"""Embodiment-specific metrics: collision events and stuck time.

Both are defined on per-control-tick time series so that the same code serves
the online monitor (episode abort) and the offline recomputation from logs.

Stuck definition (docs/measurement_protocol.md): a tick at time t is *stuck*
when, over the trailing window [t - W, t], motion was commanded at every tick and
  * translation commanded at every tick (|v_cmd| >= v_min): net displacement < d_min —
    pivoting on the spot while trying to drive does not count as progress;
  * otherwise (rotation only, or mixed): net displacement < d_min AND
    |net heading change| < yaw_min.
Stuck time: when a stuck period is first detected, the window that proved it
(t - t_anchor) is counted retroactively — otherwise every stuck period would be
under-counted by W; afterwards each further stuck tick adds its dt. The
retroactive part never reaches back past the end of the previous stuck period.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence, Tuple

import numpy as np


def count_collision_events(contact: Sequence[bool], times: Sequence[float], merge_gap_s: float) -> int:
    """Number of distinct contact episodes.

    A new event starts at a rising edge of ``contact``; a rising edge less than
    ``merge_gap_s`` after the previous contact ended is merged into it
    (bumping along a wall is one collision, not dozens).
    """
    c = np.asarray(contact, dtype=bool)
    t = np.asarray(times, dtype=float)
    if c.shape != t.shape:
        raise ValueError("contact and times differ in length")
    events = 0
    last_contact_end = -math.inf
    prev = False
    for ci, ti in zip(c, t):
        if ci and not prev and ti - last_contact_end >= merge_gap_s:
            events += 1
        if prev and not ci:
            last_contact_end = ti
        prev = bool(ci)
    return events


@dataclass(frozen=True)
class StuckParams:
    """Thresholds of the stuck detector: the robot is commanded to move but barely moves or turns.

    Attributes:
        window_s: Length of the sliding time window.
        min_displacement_m: Minimum translation within the window to count as moving.
        min_yaw_change_rad: Minimum rotation within the window to count as turning.
        min_cmd_lin: Linear command at or above which the robot counts as commanded to translate.
        min_cmd_ang: Angular command at or above which the robot counts as commanded to turn.
    """
    window_s: float
    min_displacement_m: float
    min_yaw_change_rad: float
    min_cmd_lin: float
    min_cmd_ang: float


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class StuckMonitor:
    """Online stuck detector; feed one tick at a time."""

    def __init__(self, params: StuckParams):
        self.p = params
        self._hist: Deque[Tuple[float, float, float, float, bool, bool]] = deque()
        self._last_t: float = -math.inf
        self._was_stuck = False
        self._last_stuck_t: float = -math.inf
        self.stuck_time_s = 0.0
        self.current_stuck_s = 0.0

    def update(self, t: float, x: float, y: float, yaw: float, v_cmd: float, w_cmd: float) -> bool:
        """Feed one control tick and report whether the robot is stuck now.

        Args:
            t: Time of the tick in seconds; must increase strictly.
            x: Position x in metres.
            y: Position y in metres.
            yaw: Heading in radians.
            v_cmd: Commanded linear velocity.
            w_cmd: Commanded angular velocity.

        Returns:
            True if, over a full window of commanded motion, the robot neither moved nor turned enough.

        Raises:
            ValueError: If ``t`` does not increase.
        """
        if t <= self._last_t:
            raise ValueError("times must be strictly increasing")
        dt = 0.0 if self._last_t == -math.inf else t - self._last_t
        self._last_t = t
        translating = abs(v_cmd) >= self.p.min_cmd_lin
        commanded = translating or abs(w_cmd) >= self.p.min_cmd_ang
        self._hist.append((t, x, y, yaw, commanded, translating))
        while len(self._hist) > 1 and self._hist[1][0] <= t - self.p.window_s:
            self._hist.popleft()
        t0, x0, y0, yaw0, _, _ = self._hist[0]
        full_window = t - t0 >= self.p.window_s - 1e-9
        moved = math.hypot(x - x0, y - y0) >= self.p.min_displacement_m
        turned = abs(_wrap(yaw - yaw0)) >= self.p.min_yaw_change_rad
        if not full_window or not all(h[4] for h in self._hist):
            stuck = False
        elif all(h[5] for h in self._hist):
            stuck = not moved
        else:
            stuck = not moved and not turned
        if stuck:
            add = min(t - t0, t - self._last_stuck_t) if not self._was_stuck else dt
            self.stuck_time_s += add
            self.current_stuck_s += add
            self._last_stuck_t = t
        else:
            self.current_stuck_s = 0.0
        self._was_stuck = stuck
        return stuck


def stuck_time(times: Sequence[float], poses: Sequence[Sequence[float]], cmds: Sequence[Sequence[float]],
               params: StuckParams) -> float:
    """Offline total stuck time from (t, [x, y, yaw], [v_cmd, w_cmd]) series."""
    mon = StuckMonitor(params)
    for t, (x, y, yaw), (v, w) in zip(times, poses, cmds):
        mon.update(float(t), float(x), float(y), float(yaw), float(v), float(w))
    return mon.stuck_time_s
