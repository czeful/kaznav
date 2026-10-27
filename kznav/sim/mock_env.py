"""Analytic box-world simulator with the HabitatEnv interface — for GPU-free smoke tests and CI.

The world is a flat floor (y = 0) with axis-aligned boxes (walls, furniture, goal objects).
Depth, semantic and a flat-shaded RGB image are ray-cast exactly (planar depth like Habitat);
the wheeled robot moves by the same speed/acceleration limiter as the real one with unicycle
kinematics and stops on contact (no physics); the point agent takes Habitat-style discrete
actions. Geodesic distances come from a grid Dijkstra over free space inflated by the robot radius.
Numbers from this simulator only test plumbing; they are never results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra

from ..perception.camera import Intrinsics
from .diff_drive import DriveLimits, VelocityLimiter, integrate_unicycle
from .env import Observation, Tick
from .episodes import SemanticObject
from .frames import habitat_to_map, map_to_habitat


@dataclass(frozen=True)
class MockBox:
    """An axis-aligned box of the mock world (Habitat frame corners, semantic id, colour, category)."""
    lo: Tuple[float, float, float]      # habitat (x, y, z) min corner
    hi: Tuple[float, float, float]
    semantic_id: int = 0
    color: Tuple[int, int, int] = (150, 150, 150)
    category: str = ""


def room_with_boxes(size: float = 6.0, wall_h: float = 2.5, extra: Sequence[MockBox] = ()) -> List[MockBox]:
    """Square room centred at the origin (walls 0.1 m thick) plus ``extra`` boxes."""
    h, t = size / 2, 0.1
    walls = [MockBox((-h - t, 0, -h - t), (h + t, wall_h, -h)), MockBox((-h - t, 0, h), (h + t, wall_h, h + t)),
             MockBox((-h - t, 0, -h), (-h, wall_h, h)), MockBox((h, 0, -h), (h + t, wall_h, h))]
    return walls + list(extra)


class MockEnv:
    """GPU-free box-world simulator with the ``HabitatEnv`` interface (see the module docstring).

    Args:
        cfg: The run config (the ``embodiment`` block is read).
        boxes: The world's boxes.
        grid_res: Cell size of the free-space grid used for snapping and geodesics.
        seed: Seed of the random navigable points.
    """
    def __init__(self, cfg: Mapping[str, Any], boxes: Sequence[MockBox], grid_res: float = 0.05, seed: int = 0):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self.emb = cfg["embodiment"]
        self.kind = str(self.emb["kind"])
        self.boxes = list(boxes)
        rgbd = self.emb["sensors"]["rgbd"]
        self.H, self.W = (int(v) for v in rgbd["resolution_hw"])
        self.K = Intrinsics.from_hfov(float(rgbd["hfov_deg"]), self.W, self.H)
        self.max_depth = float(rgbd["depth"]["max_m"])
        self.radius = float(self.emb["body"]["radius_m"])
        self.robot = None                       # the runner uses this to detect the physical robot
        self.sim_time = 0.0
        if self.kind == "wheeled":
            self.limiter = VelocityLimiter(DriveLimits.from_config(self.emb["drive"]["limits"]))
            self.control_dt = float(self.emb["physics"]["control_dt_s"])
        self._build_grid(grid_res)
        self.pose = (0.0, 0.0, 0.0)             # map frame (x, y, yaw)

    # ------------------------------------------------------------------ geometry
    def _build_grid(self, res: float) -> None:
        xs = [b.lo[0] for b in self.boxes] + [b.hi[0] for b in self.boxes]
        zs = [b.lo[2] for b in self.boxes] + [b.hi[2] for b in self.boxes]
        self.g_res, self.g_x0, self.g_z0 = res, min(xs), min(zs)
        nx, nz = int(math.ceil((max(xs) - self.g_x0) / res)) + 1, int(math.ceil((max(zs) - self.g_z0) / res)) + 1
        occ = np.zeros((nz, nx), dtype=bool)
        for b in self.boxes:
            c0, c1 = int((b.lo[0] - self.g_x0) / res), int(math.ceil((b.hi[0] - self.g_x0) / res))
            r0, r1 = int((b.lo[2] - self.g_z0) / res), int(math.ceil((b.hi[2] - self.g_z0) / res))
            occ[r0:r1, c0:c1] = True
        edt = ndimage.distance_transform_edt(~occ) * res
        self.free_grid = edt > self.radius + 1e-9

    def _cell(self, p: Sequence[float]) -> Tuple[int, int]:
        return int((p[2] - self.g_z0) / self.g_res), int((p[0] - self.g_x0) / self.g_res)

    def _collides(self, x: float, y_map: float) -> bool:
        px, pz = x, -y_map
        for b in self.boxes:
            dx = max(b.lo[0] - px, 0.0, px - b.hi[0])
            dz = max(b.lo[2] - pz, 0.0, pz - b.hi[2])
            if dx * dx + dz * dz < self.radius * self.radius and b.lo[1] < 0.3:
                return True
        return False

    # ------------------------------------------------------------------ rendering
    def camera_transform_at(self, position: Sequence[float], yaw: float) -> np.ndarray:
        """4x4 world transform of the camera at a reference position and map heading (as ``HabitatEnv``)."""
        mount = self.emb["sensors"]["rgbd"]["mount"]
        theta = yaw - math.pi / 2
        c, s = math.cos(theta), math.sin(theta)
        R_yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        p = math.radians(float(mount["pitch_deg"]))
        R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]])
        T = np.eye(4)
        T[:3, :3] = R_yaw @ R_pitch
        T[:3, 3] = np.asarray(position, dtype=np.float64)[:3] + R_yaw @ np.array(
            [0.0, float(mount["height_m"]), -float(mount["forward_m"])])
        return T

    def _render(self, T: np.ndarray) -> Dict[str, np.ndarray]:
        v, u = np.mgrid[0 : self.H, 0 : self.W]
        d_cam = np.stack([(u + 0.5 - self.K.cx) / self.K.fx, -(v + 0.5 - self.K.cy) / self.K.fy,
                          -np.ones_like(u, dtype=float)], axis=-1).reshape(-1, 3)
        R, o = T[:3, :3], T[:3, 3]
        d = d_cam @ R.T                                       # world directions, planar depth = parameter t
        t_best = np.full(len(d), np.inf)
        sem = np.zeros(len(d), dtype=np.int32)
        col = np.zeros((len(d), 3), dtype=np.uint8)
        down = d[:, 1] < -1e-9
        t_floor = np.where(down, -o[1] / np.where(down, d[:, 1], -1.0), np.inf)
        t_best = np.minimum(t_best, t_floor)
        col[:] = (90, 80, 70)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
        for b in self.boxes:
            t1 = (np.asarray(b.lo) - o) * inv
            t2 = (np.asarray(b.hi) - o) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 1e-6)) & (tmin < t_best)
            t_best[hit] = tmin[hit]
            sem[hit] = b.semantic_id
            col[hit] = b.color
        floor_hit = np.isfinite(t_best) & (t_best == t_floor)
        sem[floor_hit] = 0
        col[floor_hit] = (90, 80, 70)
        depth = np.where(np.isfinite(t_best) & (t_best <= self.max_depth), t_best, 0.0).astype(np.float32)
        return {"rgb": col.reshape(self.H, self.W, 3), "depth": depth.reshape(self.H, self.W),
                "semantic": sem.reshape(self.H, self.W)}

    # ------------------------------------------------------------------ HabitatEnv interface
    def reset(self, position: Sequence[float], yaw: float) -> Observation:
        """Place the robot at ``position`` with map heading ``yaw`` and return the first observation."""
        x, y = habitat_to_map(position)
        self.pose = (x, y, yaw)
        self.sim_time = 0.0
        if self.kind == "wheeled":
            self.limiter.reset()
        return self.observe()

    def observe(self) -> Observation:
        """Ray-cast RGB, depth and semantics at the current pose."""
        pos = map_to_habitat(self.pose[0], self.pose[1])
        T = self.camera_transform_at(pos, self.pose[2])
        r = self._render(T)
        return Observation(rgb=r["rgb"], depth=r["depth"], semantic=r["semantic"], pose=self.pose, position=pos,
                           camera_transform=T, sim_time=self.sim_time)

    def drive(self, v_cmd: float, w_cmd: float, n_ticks: int = 1) -> List[Tick]:
        """Apply a twist command for ``n_ticks`` ticks; the robot stops in place on contact."""
        out = []
        for _ in range(n_ticks):
            v, w = self.limiter.step(v_cmd, w_cmd, self.control_dt)
            nx, ny, nyaw = integrate_unicycle(*self.pose, v, w, self.control_dt)
            hit = self._collides(nx, ny)
            self.pose = (self.pose[0], self.pose[1], nyaw) if hit else (nx, ny, nyaw)
            if hit:
                self.limiter.v = 0.0
            self.sim_time += self.control_dt
            pos = map_to_habitat(self.pose[0], self.pose[1])
            out.append(Tick(t=self.sim_time, pose=self.pose, position=tuple(float(c) for c in pos),  # type: ignore[arg-type]
                            v_cmd=v, w_cmd=w, collision=hit, collision_links=("base_link",) if hit else ()))
        return out

    def act(self, action: str) -> Tick:
        """Take one discrete action; a blocked forward step slides along the obstacle's free axis."""
        d = self.emb["drive"]
        x, y, yaw = self.pose
        hit = False
        if action == "move_forward":
            nx, ny = x + float(d["forward_step_m"]) * math.cos(yaw), y + float(d["forward_step_m"]) * math.sin(yaw)
            hit = self._collides(nx, ny)
            if not hit:
                x, y = nx, ny
            else:
                # slide along the obstacle like Habitat's navmesh (axis-aligned approximation)
                for sx, sy in ((nx, y), (x, ny)):
                    if not self._collides(sx, sy):
                        x, y = sx, sy
                        break
        elif action in ("turn_left", "turn_right"):
            yaw += math.radians(float(d["turn_angle_deg"])) * (1 if action == "turn_left" else -1)
        self.pose = (x, y, yaw)
        self.sim_time += 1.0
        pos = map_to_habitat(x, y)
        return Tick(t=self.sim_time, pose=self.pose, position=tuple(float(c) for c in pos),  # type: ignore[arg-type]
                    v_cmd=0.0, w_cmd=0.0, collision=hit)

    def render_at(self, position: Sequence[float], yaw: float) -> Dict[str, np.ndarray]:
        """Sensor images from a hypothetical pose (the robot does not move)."""
        return self._render(self.camera_transform_at(position, yaw))

    def snap(self, point: Sequence[float], pathfinder: Any = None) -> np.ndarray:
        """The point itself (on the floor) if its cell is free; NaN otherwise."""
        r, c = self._cell(point)
        if 0 <= r < self.free_grid.shape[0] and 0 <= c < self.free_grid.shape[1] and self.free_grid[r, c]:
            return np.array([point[0], 0.0, point[2]], dtype=np.float64)
        return np.array([np.nan, np.nan, np.nan])

    def random_navigable_point(self, pathfinder: Any = None) -> np.ndarray:
        """Centre of a random free cell."""
        rr, cc = np.nonzero(self.free_grid)
        i = int(self._rng.integers(len(rr)))
        return np.array([self.g_x0 + (cc[i] + 0.5) * self.g_res, 0.0, self.g_z0 + (rr[i] + 0.5) * self.g_res])

    def geodesic_distance(self, start: Sequence[float], goals: Sequence[Sequence[float]], pathfinder: Any = None) -> float:
        """Grid-Dijkstra distance from ``start`` to the nearest of ``goals`` (inf if unreachable)."""
        from ..policy.planner import GridPlanner  # reuse the graph builder

        h, w = self.free_grid.shape
        g = GridPlanner._graph(~self.free_grid, np.ones((h, w)), self.g_res)
        s = self._cell(start)
        if not (0 <= s[0] < h and 0 <= s[1] < w) or not self.free_grid[s]:
            return math.inf
        dist = dijkstra(g, directed=True, indices=s[0] * w + s[1])
        best = math.inf
        for p in goals:
            r, c = self._cell(p)
            if 0 <= r < h and 0 <= c < w:
                best = min(best, float(dist[r * w + c]))
        return best

    def semantic_objects(self) -> Dict[int, SemanticObject]:
        """Boxes with a semantic id, as semantic objects."""
        return {b.semantic_id: SemanticObject(b.semantic_id, str(b.semantic_id), b.category, list(b.lo), list(b.hi))
                for b in self.boxes if b.semantic_id > 0}

    def close(self) -> None:
        """Nothing to release."""
        pass
