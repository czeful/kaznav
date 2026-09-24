"""SPL / SoftSPL / DTS / success on synthetic trajectories with hand-computed answers.

In an empty room the geodesic distance equals the Euclidean distance, so the
expected values below follow directly from the definitions in
kznav/eval/metrics.py (habitat-lab 0.2.4 semantics).
"""
import math

import numpy as np
import pytest

from kznav.eval.metrics import (
    EpisodeGeometry,
    SuccessRule,
    compute_episode_metrics,
    dts,
    is_success,
    path_length,
    soft_spl,
    spl,
)

RULE = SuccessRule(success_distance=0.1)


def densify(waypoints, step=0.01):
    """Sample a polyline densely, as a controller running at a high rate would record it."""
    pts = [np.asarray(waypoints[0], float)]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * k / n)
    return np.array(pts)


def geometry_from_trajectory(traj, goal, called_stop=True):
    start, end = traj[0], traj[-1]
    return EpisodeGeometry(
        start_geodesic=float(np.linalg.norm(goal - start)),
        path_length=path_length(traj),
        final_geodesic=float(np.linalg.norm(goal - end)),
        called_stop=called_stop,
    )


class TestPathLength:
    def test_straight_line(self):
        assert path_length([[0, 0, 0], [1, 0, 0], [3, 0, 0]]) == pytest.approx(3.0)

    def test_closed_square(self):
        sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        assert path_length(sq) == pytest.approx(4.0)

    def test_single_point_and_empty(self):
        assert path_length([[1.0, 2.0, 3.0]]) == 0.0
        assert path_length(np.zeros((0, 3))) == 0.0

    def test_dense_sampling_of_polyline_is_exact(self):
        traj = densify([[0, 0], [3, 0], [3, 4]])
        assert path_length(traj) == pytest.approx(7.0, abs=1e-9)

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            path_length([1.0, 2.0, 3.0])


class TestSPLOnSyntheticTrajectories:
    goal = np.array([5.0, 0.0])

    def test_optimal_straight_path_gives_spl_1(self):
        traj = densify([[0, 0], [4.95, 0]])
        m = compute_episode_metrics(geometry_from_trajectory(traj, self.goal), RULE)
        assert m.success
        assert m.spl == pytest.approx(1.0)       # p = 4.95 < l = 5.0 -> capped at 1 by max(l, p)
        assert m.dts == 0.0

    def test_detour_twice_as_long_gives_half(self):
        # Out 2.5 m and back, then 5 m to the goal: p = 10, l = 5.
        traj = densify([[0, 0], [0, 2.5], [0, 0], [5.0, 0.0]])
        g = geometry_from_trajectory(traj, self.goal)
        assert g.path_length == pytest.approx(10.0)
        m = compute_episode_metrics(g, RULE)
        assert m.success and m.spl == pytest.approx(0.5)

    def test_l_shaped_path(self):
        goal = np.array([3.0, 4.0])
        traj = densify([[0, 0], [3, 0], [3, 4]])
        m = compute_episode_metrics(geometry_from_trajectory(traj, goal), RULE)
        assert m.success
        assert m.spl == pytest.approx(5.0 / 7.0)

    def test_no_stop_is_failure_with_zero_spl_but_soft_spl(self):
        traj = densify([[0, 0], [4.95, 0]])
        m = compute_episode_metrics(geometry_from_trajectory(traj, self.goal, called_stop=False), RULE)
        assert not m.success and m.spl == 0.0
        assert m.soft_spl == pytest.approx((1 - 0.05 / 5.0) * 1.0)

    def test_success_boundary_is_strict(self):
        g = EpisodeGeometry(start_geodesic=5.0, path_length=5.0, final_geodesic=0.1, called_stop=True)
        assert not is_success(g, RULE)
        g2 = EpisodeGeometry(start_geodesic=5.0, path_length=5.0, final_geodesic=0.0999, called_stop=True)
        assert is_success(g2, RULE)

    def test_soft_spl_halfway(self):
        assert soft_spl(5.0, 5.0, 2.5) == pytest.approx(0.5)
        assert soft_spl(5.0, 10.0, 2.5) == pytest.approx(0.25)

    def test_soft_spl_zero_when_farther_than_start(self):
        assert soft_spl(5.0, 3.0, 7.0) == 0.0

    def test_unreachable_final_pose(self):
        g = EpisodeGeometry(start_geodesic=5.0, path_length=3.0, final_geodesic=math.inf, called_stop=True)
        m = compute_episode_metrics(g, RULE)
        assert not m.success and m.spl == 0.0 and m.soft_spl == 0.0 and math.isinf(m.dts)

    def test_dts(self):
        assert dts(1.1, 0.1) == pytest.approx(1.0)
        assert dts(0.05, 0.1) == 0.0

    @pytest.mark.parametrize("bad_l", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_start_geodesic_raises(self, bad_l):
        with pytest.raises(ValueError):
            spl(True, bad_l, 1.0)
        with pytest.raises(ValueError):
            soft_spl(bad_l, 1.0, 0.5)

    def test_negative_path_raises(self):
        with pytest.raises(ValueError):
            spl(True, 1.0, -0.1)

    def test_mean_spl_over_episodes(self):
        goal = np.array([5.0, 0.0])
        eps = [
            geometry_from_trajectory(densify([[0, 0], [4.95, 0]]), goal),                 # 1.0
            geometry_from_trajectory(densify([[0, 0], [0, 2.5], [0, 0], [5, 0]]), goal),  # 0.5
            geometry_from_trajectory(densify([[0, 0], [2, 0]]), goal, called_stop=True),  # stop too far: 0
        ]
        vals = [compute_episode_metrics(g, RULE).spl for g in eps]
        assert vals == pytest.approx([1.0, 0.5, 0.0])
        assert np.mean(vals) == pytest.approx(0.5)


class TestVisibilityRule:
    def test_visibility_required(self):
        rule = SuccessRule(0.1, require_visible_at_stop=True)
        g = EpisodeGeometry(5.0, 5.0, 0.05, True, goal_visible_at_stop=False)
        assert not is_success(g, rule)
        g_ok = EpisodeGeometry(5.0, 5.0, 0.05, True, goal_visible_at_stop=True)
        assert is_success(g_ok, rule)

    def test_visibility_unknown_raises(self):
        rule = SuccessRule(0.1, require_visible_at_stop=True)
        with pytest.raises(ValueError):
            is_success(EpisodeGeometry(5.0, 5.0, 0.05, True), rule)
