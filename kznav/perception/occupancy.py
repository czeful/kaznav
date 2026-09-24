"""Probabilistic 2D occupancy map (log-odds) built from RGB-D, for a given embodiment.

What counts as an obstacle depends on the robot: a point is an obstacle if its height above the
floor under the robot lies in the band [h_min, h_max], where h_min is the lowest thing the robot
cannot drive over and h_max is the robot's top plus a margin. Anything higher (a table top for a
0.17 m robot) is passable and does not enter the map — this is where the low robot and the 0.88 m
Habitat agent see different worlds.

Update rule per frame (cells counted once per frame):
* rays to floor points and to obstacle points clear the cells they cross (log-odds ``miss``);
  for a camera inside the band such a ray runs below the band top, so nothing in the band blocks it;
* obstacle endpoints get ``hit``; floor endpoints get ``miss``; a cell with both kinds of evidence
  in the same frame gets only the ``hit`` (the lowest points of a wall look like floor);
* rays to points above the band clear nothing: the floor under them was not observed;
* a pixel looking down with NO depth return where the floor plane should be (within
  ``missing_return_max_range_m``) is a drop-off -- a hole, a stairwell, the edge of a scan: the
  point where its ray meets the floor plane gets ``hit`` (D-031). Pixels at or above the horizon
  (windows, open doors, far walls) never do.

Map frame: x_map = Habitat x, y_map = -Habitat z (kznav.sim.frames). Grid row index grows with
y_map, column index with x_map; the grid is centred on the episode start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from scipy import ndimage

from .camera import Intrinsics, backproject, pixel_grid, to_world


@dataclass(frozen=True)
class MapParams:
    size_m: float
    resolution_m: float
    pixel_stride: int
    max_range_m: float
    l_hit: float
    l_miss: float
    l_min: float
    l_max: float
    p_occupied: float
    p_free: float
    obstacle_min_height_m: float
    obstacle_max_height_m: float
    cliff_depth_m: float = 0.05
    missing_return_max_range_m: float = 0.0      # 0 disables the drop-off rule

    @classmethod
    def from_config(cls, m: Mapping[str, Any], emb: Mapping[str, Any]) -> "MapParams":
        lo = m["log_odds"]
        return cls(
            size_m=float(m["size_m"]), resolution_m=float(m["resolution_m"]), pixel_stride=int(m["pixel_stride"]),
            max_range_m=float(m["max_range_m"]), l_hit=float(lo["hit"]), l_miss=float(lo["miss"]),
            l_min=float(lo["min"]), l_max=float(lo["max"]), p_occupied=float(m["p_occupied"]), p_free=float(m["p_free"]),
            obstacle_min_height_m=max(float(m["obstacle_min_height_m"]), float(emb["navmesh"]["agent_max_climb_m"])),
            obstacle_max_height_m=embodiment_top(emb) + float(m["obstacle_margin_above_top_m"]),
            cliff_depth_m=float(m["cliff_depth_m"]),
            missing_return_max_range_m=float(m.get("missing_return_max_range_m", 0.0)),
        )


def embodiment_top(emb: Mapping[str, Any]) -> float:
    """Highest point of the embodiment above the floor."""
    if emb["kind"] == "wheeled":
        from ..sim.urdf import RobotGeometry  # pure python, no Habitat import

        return RobotGeometry.from_config(emb).top_height
    return float(emb["body"]["height_m"])


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


class OccupancyMap:
    def __init__(self, params: MapParams, intrinsics: Intrinsics, origin_xy: Tuple[float, float]):
        self.p = params
        self.K = intrinsics
        self.res = params.resolution_m
        self.n = int(round(params.size_m / self.res))
        self.x0 = origin_xy[0] - params.size_m / 2
        self.y0 = origin_xy[1] - params.size_m / 2
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)
        self.observed = np.zeros((self.n, self.n), dtype=bool)
        self._l_occ = _logit(params.p_occupied)
        self._l_free = _logit(params.p_free)
        self._v, self._u = pixel_grid(intrinsics, params.pixel_stride)
        self.n_updates = 0

    # ------------------------------------------------------------------ geometry
    def to_cell(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(row, col) for map coordinates; may be out of bounds."""
        return (np.floor((np.asarray(y) - self.y0) / self.res).astype(np.int64),
                np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64))

    def to_xy(self, row: np.ndarray, col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map coordinates of cell centres."""
        return self.x0 + (np.asarray(col) + 0.5) * self.res, self.y0 + (np.asarray(row) + 0.5) * self.res

    def in_bounds(self, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)

    # ------------------------------------------------------------------ state
    @property
    def occupied(self) -> np.ndarray:
        return self.observed & (self.log_odds > self._l_occ)

    @property
    def free(self) -> np.ndarray:
        return self.observed & (self.log_odds < self._l_free)

    @property
    def unknown(self) -> np.ndarray:
        return ~self.observed

    def inflated_obstacles(self, radius_m: float) -> np.ndarray:
        r = int(np.ceil(radius_m / self.res))
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (yy * yy + xx * xx) * self.res * self.res <= radius_m * radius_m
        return ndimage.binary_dilation(self.occupied, structure=disk)

    # ------------------------------------------------------------------ update
    def classify_points(self, pts_world: np.ndarray, floor_y: float) -> np.ndarray:
        """0 = floor, 1 = obstacle, 2 = above the band (passable).

        Points more than ``cliff_depth_m`` below the floor (a scan hole, a step or stairs down) are
        obstacles too — the map's equivalent of a robot vacuum's cliff sensors."""
        h = pts_world[:, 1] - floor_y
        cat = np.full(len(h), 2, dtype=np.int8)
        cat[h < self.p.obstacle_min_height_m] = 0
        cat[(h >= self.p.obstacle_min_height_m) & (h <= self.p.obstacle_max_height_m)] = 1
        cat[h < -self.p.cliff_depth_m] = 1
        return cat

    def update(self, depth: np.ndarray, T_world_cam: np.ndarray, floor_y: float) -> np.ndarray:
        """Integrate one depth frame; returns the (row, col) cells observed in it."""
        d = depth[self._v, self._u]
        valid = (d > 0) & (d <= self.p.max_range_m) & np.isfinite(d)
        drops = self.missing_floor_points(depth, d, T_world_cam, floor_y)
        if not valid.any() and len(drops) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        pts = to_world(backproject(depth, self.K, self._v[valid], self._u[valid]), T_world_cam) if valid.any() \
            else np.zeros((0, 3))
        return self.update_from_points(pts, T_world_cam[:3, 3], floor_y, drop_points=drops)

    def missing_floor_points(self, depth: np.ndarray, d: np.ndarray, T_world_cam: np.ndarray,
                             floor_y: float) -> np.ndarray:
        """World points where downward rays of no-return pixels meet the floor plane (drop-offs)."""
        rmax = self.p.missing_return_max_range_m
        miss = (d == 0) | ~np.isfinite(d)
        if rmax <= 0 or not miss.any():
            return np.zeros((0, 3))
        unit = to_world(backproject(np.ones_like(depth, dtype=np.float32), self.K, self._v[miss], self._u[miss]),
                        T_world_cam)
        cam = T_world_cam[:3, 3]
        dirs = unit - cam[None, :]
        down = dirs[:, 1] < -1e-6
        if not down.any():
            return np.zeros((0, 3))
        t = (floor_y - cam[1]) / dirs[down, 1]
        hit = cam[None, :] + t[:, None] * dirs[down]
        horiz = np.hypot(hit[:, 0] - cam[0], hit[:, 2] - cam[2])
        return hit[(t > 0) & (horiz <= rmax)]

    def update_from_points(self, pts_world: np.ndarray, cam_world: np.ndarray, floor_y: float,
                           drop_points: Optional[np.ndarray] = None) -> np.ndarray:
        cat = self.classify_points(pts_world, floor_y)
        ex, ey = pts_world[:, 0], -pts_world[:, 2]
        er, ec = self.to_cell(ex, ey)
        cr, cc = self.to_cell(np.array([cam_world[0]]), np.array([-cam_world[2]]))
        clearing = cat <= 1
        free_cells = self._ray_cells(cr[0], cc[0], er[clearing], ec[clearing])
        floor_end = np.stack([er[cat == 0], ec[cat == 0]], axis=1)
        hit_end = np.stack([er[cat == 1], ec[cat == 1]], axis=1)
        if drop_points is not None and len(drop_points):
            dr, dc = self.to_cell(drop_points[:, 0], -drop_points[:, 2])
            hit_end = np.concatenate([hit_end, np.stack([dr, dc], axis=1)])   # no clearing along these rays
        free_cells = np.unique(np.concatenate([free_cells, floor_end]), axis=0) if len(free_cells) or len(floor_end) \
            else np.zeros((0, 2), dtype=np.int64)
        hit_cells = np.unique(hit_end, axis=0) if len(hit_end) else np.zeros((0, 2), dtype=np.int64)
        if len(hit_cells) and len(free_cells):
            # within one frame, obstacle evidence wins over the floor / pass-through evidence of the same cell
            key = lambda a: a[:, 0] * (self.n + 1) + a[:, 1]  # noqa: E731
            free_cells = free_cells[~np.isin(key(free_cells), key(hit_cells))]
        seen = []
        for cells, delta in ((free_cells, self.p.l_miss), (hit_cells, self.p.l_hit)):
            if len(cells) == 0:
                continue
            ok = self.in_bounds(cells[:, 0], cells[:, 1])
            r, c = cells[ok, 0], cells[ok, 1]
            self.log_odds[r, c] = np.clip(self.log_odds[r, c] + delta, self.p.l_min, self.p.l_max)
            self.observed[r, c] = True
            seen.append(cells[ok])
        self.n_updates += 1
        return np.concatenate(seen) if seen else np.zeros((0, 2), dtype=np.int64)

    @staticmethod
    def _ray_cells(r0: int, c0: int, r1: np.ndarray, c1: np.ndarray) -> np.ndarray:
        """Cells crossed by segments (r0, c0) -> (r1, c1), excluding the end cells (unique per call)."""
        if len(r1) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        ends = np.unique(np.stack([r1, c1], axis=1), axis=0)
        dr = ends[:, 0] - r0
        dc = ends[:, 1] - c0
        steps = np.maximum(np.abs(dr), np.abs(dc))
        k = int(steps.max()) * 2 + 1
        t = np.linspace(0.0, 1.0, k, endpoint=False)[None, :]          # excludes t = 1 (the end cell)
        rr = np.floor(r0 + 0.5 + dr[:, None] * t).astype(np.int64)
        cc = np.floor(c0 + 0.5 + dc[:, None] * t).astype(np.int64)
        # drop samples that already landed in the end cell (short rays)
        keep = ~((rr == ends[:, 0:1]) & (cc == ends[:, 1:2]))
        cells = np.stack([rr[keep], cc[keep]], axis=1)
        return np.unique(cells, axis=0)

    def mark_free_disk(self, xy: Tuple[float, float], radius_m: float) -> None:
        """The robot's own footprint is free space (the camera cannot see under the robot)."""
        r = int(np.ceil(radius_m / self.res))
        cr, cc = self.to_cell(np.array([xy[0]]), np.array([xy[1]]))
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (yy * yy + xx * xx) * self.res * self.res <= radius_m * radius_m
        rr, cols = (yy[disk] + cr[0]), (xx[disk] + cc[0])
        ok = self.in_bounds(rr, cols)
        rr, cols = rr[ok], cols[ok]
        self.log_odds[rr, cols] = np.minimum(self.log_odds[rr, cols], self.p.l_miss)
        self.observed[rr, cols] = True

    def mark_obstacle_disk(self, xy: Tuple[float, float], radius_m: float) -> None:
        """Virtual obstacle (e.g. where the robot got blocked by something below the obstacle band)."""
        r = int(np.ceil(radius_m / self.res))
        cr, cc = self.to_cell(np.array([xy[0]]), np.array([xy[1]]))
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (yy * yy + xx * xx) * self.res * self.res <= radius_m * radius_m
        rr, cols = (yy[disk] + cr[0]), (xx[disk] + cc[0])
        ok = self.in_bounds(rr, cols)
        self.log_odds[rr[ok], cols[ok]] = self.p.l_max
        self.observed[rr[ok], cols[ok]] = True

    # ------------------------------------------------------------------ export
    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def explored_area_m2(self) -> float:
        return float(self.observed.sum()) * self.res * self.res
