import math

import numpy as np
import pytest

from kznav.eval.failures import Outcome, OutcomeInputs, Termination, classify_episode
from kznav.eval.robot import StuckMonitor, StuckParams, count_collision_events, stuck_time


class TestOutcomes:
    def test_success(self):
        assert classify_episode(OutcomeInputs(Termination.STOP_CALLED, True, True)) is Outcome.SUCCESS

    def test_success_requires_stop(self):
        with pytest.raises(ValueError):
            classify_episode(OutcomeInputs(Termination.STEP_LIMIT, True))

    @pytest.mark.parametrize("is_goal,expected", [
        (False, Outcome.FALSE_STOP), (True, Outcome.STOP_TOO_FAR), (None, Outcome.STOP_UNLABELED)])
    def test_stop_failures(self, is_goal, expected):
        assert classify_episode(OutcomeInputs(Termination.STOP_CALLED, False, is_goal)) is expected

    @pytest.mark.parametrize("term,expected", [
        (Termination.STEP_LIMIT, Outcome.STEP_LIMIT),
        (Termination.NO_FRONTIERS, Outcome.NOT_FOUND),
        (Termination.STUCK_ABORT, Outcome.STUCK),
        (Termination.COLLISION_ABORT, Outcome.COLLISIONS),
        (Termination.MODEL_ERROR, Outcome.MODEL_FAILURE),
        (Termination.SIM_ERROR, Outcome.SIM_FAILURE),
    ])
    def test_non_stop_terminations(self, term, expected):
        assert classify_episode(OutcomeInputs(term, False)) is expected

    def test_every_termination_is_mapped(self):
        for t in Termination:
            if t is not Termination.STOP_CALLED:
                classify_episode(OutcomeInputs(t, False))


PARAMS = StuckParams(window_s=3.0, min_displacement_m=0.05, min_yaw_change_rad=0.1, min_cmd_lin=0.05, min_cmd_ang=0.2)


def series(n, dt=0.1):
    return [round(i * dt, 10) for i in range(n)]


class TestStuck:
    def test_moving_robot_is_never_stuck(self):
        t = series(100)
        poses = [(0.3 * ti, 0.0, 0.0) for ti in t]
        assert stuck_time(t, poses, [(0.3, 0.0)] * len(t), PARAMS) == 0.0

    def test_blocked_robot(self):
        t = series(100)  # 0.0 .. 9.9 s, commanded forward, not moving
        poses = [(1.0, 1.0, 0.5)] * len(t)
        # Detected at t = 3.0; the proving window [0, 3] counts retroactively -> the whole 0..9.9 s.
        assert stuck_time(t, poses, [(0.3, 0.0)] * len(t), PARAMS) == pytest.approx(9.9)

    def test_short_block_below_window(self):
        t = series(20)  # 2 s
        assert stuck_time(t, [(0.0, 0.0, 0.0)] * 20, [(0.3, 0.0)] * 20, PARAMS) == 0.0

    def test_rotating_in_place_is_not_stuck(self):
        t = series(100)
        poses = [(0.0, 0.0, 0.5 * ti) for ti in t]
        assert stuck_time(t, poses, [(0.0, 0.5)] * len(t), PARAMS) == 0.0

    def test_idle_robot_is_not_stuck(self):
        t = series(100)
        assert stuck_time(t, [(0.0, 0.0, 0.0)] * 100, [(0.0, 0.0)] * 100, PARAMS) == 0.0

    def test_command_gap_resets(self):
        t = series(100)
        cmds = [(0.3, 0.0)] * 100
        cmds[50] = (0.0, 0.0)  # one idle tick at t = 5.0 breaks the window
        total = stuck_time(t, [(0.0, 0.0, 0.0)] * 100, cmds, PARAMS)
        # stuck 0.0..4.9 (4.9 s) and 5.1..9.9 (4.8 s; first full window after the idle tick is at 8.1)
        assert total == pytest.approx(9.7)

    def test_current_stuck_duration_for_abort(self):
        mon = StuckMonitor(PARAMS)
        for ti in series(60):
            mon.update(ti, 0.0, 0.0, 0.0, 0.3, 0.0)
        assert mon.current_stuck_s == pytest.approx(5.9)

    def test_retroactive_window_does_not_double_count(self):
        # Stuck at pose A until 4.9, jolted at 5.0, stuck at pose B afterwards.
        t = series(100)
        poses = [(0.0, 0.0, 0.0) if ti < 5.0 else (0.5, 0.0, 0.0) for ti in t]
        total = stuck_time(t, poses, [(0.3, 0.0)] * 100, PARAMS)
        assert total <= 9.9 + 1e-9
        assert total == pytest.approx(4.9 + 4.9)  # 0..4.9 and 5.0..9.9

    def test_time_must_increase(self):
        mon = StuckMonitor(PARAMS)
        mon.update(1.0, 0, 0, 0, 0, 0)
        with pytest.raises(ValueError):
            mon.update(1.0, 0, 0, 0, 0, 0)


class TestCollisions:
    def test_merging_and_counting(self):
        t = [round(0.1 * i, 10) for i in range(30)]
        c = [False] * 30
        for i in (1, 2):      # contact 0.1-0.2, ends at 0.3
            c[i] = True
        for i in (5, 6):      # new contact at 0.5, only 0.2 s later -> same event
            c[i] = True
        for i in (20, 21):    # contact at 2.0 -> new event
            c[i] = True
        assert count_collision_events(c, t, merge_gap_s=0.5) == 2
        assert count_collision_events(c, t, merge_gap_s=0.0) == 3

    def test_no_contact(self):
        assert count_collision_events([False] * 5, [0, 1, 2, 3, 4], 0.5) == 0

    def test_contact_from_start(self):
        assert count_collision_events([True, True, False], [0, 1, 2], 0.5) == 1
