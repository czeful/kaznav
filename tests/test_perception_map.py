"""Occupancy mapping, frontiers and grid planning on analytically generated depth."""
import math

import numpy as np
import pytest

from kznav.perception.camera import Intrinsics, backproject, pixel_grid, to_world
from kznav.perception.frontiers import find_frontiers
from kznav.perception.occupancy import MapParams, OccupancyMap
from kznav.policy.planner import GridPlanner, PlannerParams, simplify_path

K = Intrinsics.from_hfov(69.0, 160, 120)
CAM_H = 0.15


def params(**kw):
    base = dict(size_m=10.0, resolution_m=0.05, pixel_stride=1, max_range_m=4.5, l_hit=0.85, l_miss=-0.5,
                l_min=-3.5, l_max=3.5, p_occupied=0.6, p_free=0.4, obstacle_min_height_m=0.03,
                obstacle_max_height_m=0.195)
    base.update(kw)
    return MapParams(**base)


def cam_T(x=0.0, z=0.0, h=CAM_H):
    """Camera at (x, h, z), looking along -Z (map +y), no pitch."""
    T = np.eye(4)
    T[:3, 3] = [x, h, z]
    return T


def ray_dirs():
    v, u = np.mgrid[0:K.height, 0:K.width]
    dx = (u + 0.5 - K.cx) / K.fx
    dy = -(v + 0.5 - K.cy) / K.fy
    return dx, dy          # direction (dx, dy, -1) per unit planar depth


def synth_depth(wall_at=None, table=None):
    """Planar depth for a flat floor (y=0), optional wall at map y = wall_at, optional table top.

    table = (y_near, y_far, height): a horizontal slab over the floor between those map-y distances.
    """
    dx, dy = ray_dirs()
    depth = np.full(dx.shape, np.inf)
    down = dy < 0
    depth[down] = CAM_H / -dy[down]                     # floor: CAM_H + dy * z = 0
    if wall_at is not None:
        depth = np.minimum(depth, wall_at)
    if table is not None:
        y0, y1, th = table
        up = dy > 0
        z_top = np.where(up, (th - CAM_H) / np.where(up, dy, 1), np.inf)   # hits the slab underside
        hit = up & (z_top >= y0) & (z_top <= y1)
        depth[hit] = np.minimum(depth[hit], z_top[hit])
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


class TestCamera:
    def test_backproject_project_round_trip(self):
        v, u = pixel_grid(K, 7)
        depth = np.full((K.height, K.width), 2.5, dtype=np.float32)
        p = backproject(depth, K, v, u)
        uu, vv = K.project(p)
        assert np.allclose(uu, u) and np.allclose(vv, v)
        assert np.allclose(p[:, 2], -2.5)

    def test_floor_points_land_on_floor(self):
        depth = synth_depth()
        v, u = pixel_grid(K, 2)
        ok = depth[v, u] > 0
        pts = to_world(backproject(depth, K, v[ok], u[ok]), cam_T())
        assert np.abs(pts[:, 1]).max() < 1e-6          # float32 depth


class TestOccupancy:
    def test_open_floor_is_free_not_occupied(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        m.update(synth_depth(), cam_T(), floor_y=0.0)
        assert not m.occupied.any()
        r, c = m.to_cell(np.array([0.0]), np.array([1.0]))
        assert m.free[r[0], c[0]]
        r, c = m.to_cell(np.array([0.0]), np.array([-1.0]))       # behind the camera: never seen
        assert m.unknown[r[0], c[0]]

    def test_wall_is_occupied_space_before_it_free_behind_unknown(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        for _ in range(3):
            m.update(synth_depth(wall_at=2.02), cam_T(), floor_y=0.0)
        ys = np.array([1.0, 2.02, 2.6])
        r, c = m.to_cell(np.zeros(3), ys)
        assert m.free[r[0], c[0]]
        assert m.occupied[r[1], c[1]]
        assert m.unknown[r[2], c[2]]

    def test_overhead_table_is_passable_for_a_low_robot(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        m.update(synth_depth(table=(1.0, 3.0, 0.72)), cam_T(), floor_y=0.0)
        assert not m.occupied.any()

    def test_same_table_blocks_a_tall_agent(self):
        m = OccupancyMap(params(obstacle_max_height_m=0.91), K, (0.0, 0.0))
        m.update(synth_depth(table=(1.0, 3.0, 0.72)), cam_T(), floor_y=0.0)
        assert m.occupied.any()

    def test_log_odds_clamped(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        for _ in range(20):
            m.update(synth_depth(wall_at=2.0), cam_T(), floor_y=0.0)
        assert m.log_odds.max() <= 3.5 + 1e-6 and m.log_odds.min() >= -3.5 - 1e-6

    def test_invalid_depth_ignored(self):
        m = OccupancyMap(params(), K, (0.0, 0.0))
        m.update(np.zeros((K.height, K.width), np.float32), cam_T(), floor_y=0.0)
        assert not m.observed.any()


def grid_map(free_box=None, walls=(), size=6.0):
    """Hand-built map: everything unknown except ``free_box`` (x0, y0, x1, y1) free; ``walls`` occupied boxes."""
    m = OccupancyMap(params(size_m=size), K, (0.0, 0.0))
    def box(b, value):
        r0, c0 = m.to_cell(np.array([b[0]]), np.array([b[1]]))
        r1, c1 = m.to_cell(np.array([b[2]]), np.array([b[3]]))
        m.log_odds[r0[0]:r1[0] + 1, c0[0]:c1[0] + 1] = value
        m.observed[r0[0]:r1[0] + 1, c0[0]:c1[0] + 1] = True
    if free_box:
        box(free_box, -3.0)
    for w in walls:
        box(w, 3.0)
    return m


PP = PlannerParams(inflation_margin_m=0.03, unknown_cost=1.5, proximity_weight=0.0, proximity_range_m=0.3,
                   crop_margin_m=1.0, start_escape_m=0.3)


class TestFrontiers:
    def test_frontier_at_edge_of_free_area(self):
        m = grid_map(free_box=(-1.0, -1.0, 1.0, 1.0))
        start = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([0.0])))
        fr = find_frontiers(m.free, m.unknown, m.inflated_obstacles(0.2), start, min_size=6)
        assert len(fr) == 1 and fr[0].size > 100      # the whole boundary ring is one region

    def test_walled_off_frontier_is_unreachable(self):
        # free room with a full wall ring; outside the ring unknown
        m = grid_map(free_box=(-2.0, -2.0, 2.0, 2.0), walls=[(-1.0, -1.0, 1.0, -0.9), (-1.0, 0.9, 1.0, 1.0),
                                                           (-1.0, -1.0, -0.9, 1.0), (0.9, -1.0, 1.0, 1.0)])
        start = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([0.0])))
        fr = find_frontiers(m.free, m.unknown, m.inflated_obstacles(0.2), start, min_size=6)
        assert fr == []

    def test_min_size_filter(self):
        m = grid_map(free_box=(-1.0, -1.0, 1.0, 1.0))
        start = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([0.0])))
        assert find_frontiers(m.free, m.unknown, m.inflated_obstacles(0.2), start, min_size=10_000) == []


class TestPlanner:
    def test_straight_line_cost_equals_distance(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5))
        res = GridPlanner(PP, 0.17).plan(m, (-1.5, 0.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([1.5]), np.array([0.0])))
        assert res.ok and res.cost_to(goal) == pytest.approx(3.0, abs=0.06)

    def test_goes_through_gap(self):
        # wall across y = 0 with a 0.8 m gap at x in [0.6, 1.4]
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-3.0, -0.05, 0.6, 0.05), (1.4, -0.05, 3.0, 0.05)])
        res = GridPlanner(PP, 0.17).plan(m, (-1.0, -1.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([-1.0]), np.array([1.0])))
        cells = res.path_cells(goal)
        xs, ys = m.to_xy(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
        crossing = xs[np.argmin(np.abs(ys))]
        assert 0.6 + 0.2 <= crossing <= 1.4 - 0.2
        assert res.cost_to(goal) > 2.0 + 1e-6

    def test_gap_narrower_than_robot_is_closed(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-3.0, -0.05, 0.6, 0.05), (0.9, -0.05, 3.0, 0.05)])
        res = GridPlanner(PP, 0.17).plan(m, (-1.0, -1.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([-1.0]), np.array([1.0])))
        assert math.isinf(res.cost_to(goal))

    def test_unknown_is_optimistic_but_costlier(self):
        m = grid_map(free_box=(-2.5, -0.5, 2.5, 0.5))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([0.0]), np.array([2.0])))
        res = GridPlanner(PP, 0.17).plan(m, (0.0, 0.0), extra_cells=[goal])   # goal lies outside the observed crop
        # 0.5 m of free cells then 1.5 m of unknown at 1.5x
        assert res.cost_to(goal) == pytest.approx(0.5 + 1.5 * 1.5, abs=0.08)

    def test_start_escape_when_touching_wall(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-2.5, 0.0, 2.5, 0.1)])
        res = GridPlanner(PP, 0.17).plan(m, (0.0, -0.15))
        assert res.ok and res.start_escaped

    def test_simplified_path_is_short_and_collision_free(self):
        m = grid_map(free_box=(-2.5, -2.5, 2.5, 2.5), walls=[(-3.0, -0.05, 0.6, 0.05), (1.4, -0.05, 3.0, 0.05)])
        res = GridPlanner(PP, 0.17).plan(m, (-1.0, -1.0))
        goal = tuple(int(v[0]) for v in m.to_cell(np.array([-1.0]), np.array([1.0])))
        cells = res.path_cells(goal)
        simple = simplify_path(cells, res.blocked)
        assert simple[0] == res.start and simple[-1] == goal
        assert len(simple) < len(cells) / 5
        from kznav.policy.planner import _line_free
        assert all(_line_free(a, b, res.blocked) for a, b in zip(simple[:-1], simple[1:]))
