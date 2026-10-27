"""Grid planner on the robot's own occupancy map (no privileged navmesh).

Cost of a cell: infinite inside obstacles inflated by (robot radius + margin); 1 for known-free
cells; ``unknown_cost`` for unobserved cells (optimistic exploration); plus a proximity penalty
``w * max(0, 1 - d / range)`` where d is the distance to the nearest obstacle, which keeps paths
away from walls. One Dijkstra from the robot gives the path cost to every cell, so all frontiers
are ranked with a single search. The search runs on a crop around the observed area.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..perception.occupancy import OccupancyMap

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class PlannerParams:
    """Planner parameters (``policy.planner``): inflation margin, unknown-cell cost, wall proximity penalty, crop."""
    inflation_margin_m: float
    unknown_cost: float
    proximity_weight: float
    proximity_range_m: float
    crop_margin_m: float
    start_escape_m: float       # if the robot's cell is inside the inflated zone, start from the nearest free cell within this

    @classmethod
    def from_config(cls, c: Mapping[str, Any]) -> "PlannerParams":
        """Build from the ``policy.planner`` config block."""
        return cls(**{k: float(c[k]) for k in cls.__dataclass_fields__})


@dataclass
class PlanResult:
    """Path costs from the robot to every cell of the map, with predecessors to recover paths."""
    ok: bool
    start: Tuple[int, int]
    dist: np.ndarray                 # full-map path cost (inf where unreachable)
    pred: np.ndarray                 # full-map predecessor index (-1 none), flat indices into the full grid
    blocked: np.ndarray
    start_escaped: bool = False

    def cost_to(self, cell: Tuple[int, int]) -> float:
        """Path cost to a cell (inf if unreachable)."""
        return float(self.dist[cell])

    def path_cells(self, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Cells of the cheapest path from the start to ``goal``; empty if unreachable."""
        if not np.isfinite(self.dist[goal]):
            return []
        n = self.dist.shape[1]
        out = []
        idx = goal[0] * n + goal[1]
        start_idx = self.start[0] * n + self.start[1]
        guard = self.dist.size
        while idx != start_idx and guard > 0:
            out.append((idx // n, idx % n))
            idx = self.pred.flat[idx]
            if idx < 0:
                return []
            guard -= 1
        out.append(self.start)
        return out[::-1]


class GridPlanner:
    """Dijkstra planner on the robot's own occupancy map (see the module docstring for the cell costs).

    Args:
        params: Planner parameters.
        robot_radius_m: Radius of the robot's footprint, added to the inflation margin.
    """
    def __init__(self, params: PlannerParams, robot_radius_m: float):
        self.p = params
        self.radius = robot_radius_m

    def plan(self, omap: OccupancyMap, start_xy: Tuple[float, float],
             extra_cells: Sequence[Tuple[int, int]] = (), margin_m: Optional[float] = None) -> PlanResult:
        """Costs from the robot to every cell of a crop around the observed area.

        Goals outside the observed area (e.g. a detected object seen from afar) must be passed in
        ``extra_cells`` so that the crop includes them. ``margin_m`` overrides the inflation margin
        (0 = "tight" planning with the bare robot radius).
        """
        margin = self.p.inflation_margin_m if margin_m is None else float(margin_m)
        blocked = omap.inflated_obstacles(self.radius + margin)
        rr, cc = omap.to_cell(np.array([start_xy[0]]), np.array([start_xy[1]]))
        sr, sc = int(rr[0]), int(cc[0])
        empty = PlanResult(False, (sr, sc), np.full(blocked.shape, np.inf), np.full(blocked.shape, -1), blocked)
        if not omap.in_bounds(np.array([sr]), np.array([sc]))[0]:
            return empty
        escaped = False
        if blocked[sr, sc]:
            cand = self._nearest_unblocked(blocked, (sr, sc), int(math.ceil(self.p.start_escape_m / omap.res)))
            if cand is None:
                return empty
            (sr, sc), escaped = cand, True

        # crop to observed area + requested cells + margin
        obs_r, obs_c = np.nonzero(omap.observed)
        rows = np.concatenate([obs_r, [sr], [c[0] for c in extra_cells]]).astype(int)
        cols = np.concatenate([obs_c, [sc], [c[1] for c in extra_cells]]).astype(int)
        m = int(math.ceil(self.p.crop_margin_m / omap.res))
        r0, r1 = max(0, rows.min() - m), min(omap.n, rows.max() + m + 1)
        c0, c1 = max(0, cols.min() - m), min(omap.n, cols.max() + m + 1)
        blk = blocked[r0:r1, c0:c1]
        cost = np.where(omap.observed[r0:r1, c0:c1], 1.0, self.p.unknown_cost)
        if self.p.proximity_weight > 0:
            occ = omap.occupied[r0:r1, c0:c1]
            d = ndimage.distance_transform_edt(~occ) * omap.res if occ.any() else np.full(occ.shape, np.inf)
            cost = cost + self.p.proximity_weight * np.clip(1.0 - d / self.p.proximity_range_m, 0.0, 1.0)
        h, w = blk.shape
        g = self._graph(blk, cost, omap.res)
        s_local = (sr - r0) * w + (sc - c0)
        dist_l, pred_l = dijkstra(g, directed=True, indices=s_local, return_predecessors=True)

        dist = np.full(blocked.shape, np.inf)
        dist[r0:r1, c0:c1] = dist_l.reshape(h, w)
        pred = np.full(blocked.shape, -1, dtype=np.int64)
        valid = pred_l >= 0
        pl = pred_l.reshape(h, w)
        vr, vc = np.nonzero(valid.reshape(h, w))
        pr, pc = pl[vr, vc] // w + r0, pl[vr, vc] % w + c0
        pred[vr + r0, vc + c0] = pr * blocked.shape[1] + pc
        return PlanResult(True, (sr, sc), dist, pred, blocked, escaped)

    @staticmethod
    def _graph(blocked: np.ndarray, cost: np.ndarray, res: float) -> csr_matrix:
        """8-connected directed graph over unblocked cells; edge weight = length * mean cell cost."""
        h, w = blocked.shape
        idx = np.arange(h * w).reshape(h, w)
        ok = ~blocked
        flat_cost = cost.ravel()
        src, dst, wt = [], [], []
        for dr, dc in _OFFSETS:
            ra0, ra1 = max(0, -dr), h - max(0, dr)
            ca0, ca1 = max(0, -dc), w - max(0, dc)
            if ra1 <= ra0 or ca1 <= ca0:
                continue
            both = ok[ra0:ra1, ca0:ca1] & ok[ra0 + dr:ra1 + dr, ca0 + dc:ca1 + dc]
            if dr and dc:
                # no corner cutting: a diagonal step needs both orthogonal neighbours free
                both &= ok[ra0 + dr:ra1 + dr, ca0:ca1] & ok[ra0:ra1, ca0 + dc:ca1 + dc]
            ia = idx[ra0:ra1, ca0:ca1][both]
            ib = idx[ra0 + dr:ra1 + dr, ca0 + dc:ca1 + dc][both]
            src.append(ia)
            dst.append(ib)
            wt.append(math.hypot(dr, dc) * res * 0.5 * (flat_cost[ia] + flat_cost[ib]))
        return csr_matrix((np.concatenate(wt), (np.concatenate(src), np.concatenate(dst))), shape=(h * w, h * w))

    @staticmethod
    def _nearest_unblocked(blocked: np.ndarray, cell: Tuple[int, int], radius_cells: int) -> Optional[Tuple[int, int]]:
        r, c = cell
        best, best_d = None, math.inf
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < blocked.shape[0] and 0 <= cc < blocked.shape[1] and not blocked[rr, cc]:
                    d = dr * dr + dc * dc
                    if d < best_d:
                        best, best_d = (rr, cc), d
        return best


def simplify_path(cells: List[Tuple[int, int]], blocked: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy line-of-sight shortcutting: keep the farthest cell visible without crossing ``blocked``."""
    if len(cells) <= 2:
        return list(cells)
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_free(cells[i], cells[j], blocked):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _line_free(a: Tuple[int, int], b: Tuple[int, int], blocked: np.ndarray) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    t = np.linspace(0.0, 1.0, n)
    rr = np.round(a[0] + (b[0] - a[0]) * t).astype(int)
    cc = np.round(a[1] + (b[1] - a[1]) * t).astype(int)
    return not blocked[rr, cc].any()
