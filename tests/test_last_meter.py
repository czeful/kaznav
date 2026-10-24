"""D-070 "last metre": commit to the candidate being approached, ask the verifier from afar when the whole object is
in view, and stop where the object is in the camera's vertical field of view (visible stop mode)."""
import math
from pathlib import Path

import numpy as np
import pytest

from kznav.language.base import ListSink, TeeSink
from kznav.language.goal import parse_command
from kznav.language.lexicon import Lexicon
from kznav.language.registry import create_backend
from kznav.language.schemas import VerificationResult, VerifyQuery
from kznav.perception.base import create_detector, create_segmenter
from kznav.perception.objects import ObjectCandidate
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.policy.verification import Verifier, VerifyParams, make_verifier
from kznav.run.episode import Executor, run_episode
from kznav.sim.mock_env import MockEnv, room_with_boxes

from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg

LAST_METRE = ["policy.approach.stop_mode=visible", "policy.verification.commit_approach=true",
              "policy.verification.verify_range_m=1.5"]


class FakeBackend:
    def __init__(self, ps):
        self.ps, self.calls = list(ps), 0

    def verify_candidate(self, image, query):
        p = self.ps[min(self.calls, len(self.ps) - 1)]
        self.calls += 1
        return VerificationResult(p, "logprobs", 1.0, "", None)


def verifier(ps, max_wait_steps=2):
    params = VerifyParams(stop_threshold=0.6, reject_below=0.25, base_rate=0.5, max_calls=3, regrow_factor=2.0,
                          max_wait_steps=max_wait_steps, crop_padding=0.25, min_crop_px=16, calibrated=False)
    return Verifier(FakeBackend(ps), VerifyQuery("chair", "en"), params, None, view_discount=0.5, first_view_full=True)


def policy_for(embodiment="wheeled_vacuum", extra=(), verifier_=None):
    cfg = small_cfg(embodiment, detector="none", extra=extra)
    env = MockEnv(cfg, room_with_boxes(6.0))
    p = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), ["chair"], env.K,
                        verifier=verifier_)
    p.reset((0.0, 0.0, 0.0))
    p.spin_left = 0
    p._rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    return p


def candidate(x, heights, box=(60, 40, 100, 90), conf=0.6, step=0):
    """Points on a short wall of the object at map x = ``x`` (y from -0.2 to 0.2), at the given heights."""
    ys = np.linspace(-0.2, 0.2, len(heights))
    pts = np.stack([np.full(len(heights), x), ys], axis=1)
    return ObjectCandidate(0, 0, np.array([x, 0.5, 0.0]), pts, 0.6, 1, step, step, scores={0: 0.6},
                           last_box=box, conf=conf, points_h=np.asarray(heights, dtype=float))


# ---------------------------------------------------------------- visibility geometry
def test_a_tv_on_a_stand_is_out_of_the_vacuum_cameras_view_at_0_5_m_and_in_view_at_0_85_m():
    p = policy_for(extra=["policy.approach.stop_mode=visible"])
    tv = candidate(2.0, np.linspace(0.6, 1.2, 40))
    near = p._visible_fraction(tv, np.array(1.5), np.array(0.0))      # robot centre 0.5 m from the screen
    ring = p._visible_fraction(tv, np.array(1.15), np.array(0.0))     # 0.85 m
    assert float(near) < p.visible_fraction <= float(ring)
    assert p._in_stop_range(tv, 0.85, (1.15, 0.0, 0.0))


def test_distance_mode_keeps_the_old_stop_range():
    p = policy_for()
    tv = candidate(2.0, np.linspace(0.6, 1.2, 40))
    assert p.stop_mode == "distance"
    assert not p._in_stop_range(tv, 0.85, (1.15, 0.0, 0.0)) and p._in_stop_range(tv, 0.55, (1.45, 0.0, 0.0))


def test_within_the_old_stop_distance_is_always_in_range_so_a_wall_tv_cannot_hold_the_robot():
    p = policy_for(extra=["policy.approach.stop_mode=visible"])
    wall_tv = candidate(2.0, np.linspace(1.6, 2.0, 40))           # out of view from every cell of the ring
    assert float(p._visible_fraction(wall_tv, np.array(1.15), np.array(0.0))) < p.visible_fraction
    assert not p._in_stop_range(wall_tv, 0.85, (1.15, 0.0, 0.0))
    assert p._in_stop_range(wall_tv, 0.5, (1.5, 0.0, 0.0))


def test_the_standard_agents_camera_sees_the_tv_from_close_by():
    p = policy_for("habitat_agent", extra=["policy.approach.stop_mode=visible"])
    tv = candidate(2.0, np.linspace(0.6, 1.2, 40))
    assert float(p._visible_fraction(tv, np.array(1.5), np.array(0.0))) > 0.9


# ---------------------------------------------------------------- commit to the approached candidate
def test_commit_keeps_a_candidate_whose_fused_confidence_dropped():
    """D-065: one missed frame halves the goal's fused confidence (0.51 -> 0.26) and the robot left it. Committed,
    it stays the target until the verifier decides."""
    extra = ["perception.detection.fusion.enabled=true", "policy.verification.commit_approach=true"]
    p = policy_for(extra=extra, verifier_=verifier([0.55]))
    c = candidate(2.0, np.linspace(0.1, 0.6, 20), conf=0.26)      # suspected: below the 0.45 bar
    p.memory.candidates.append(c)
    assert p._best() is None
    p.committed = c.cid
    assert p._best() is c
    c.rejected = True
    assert p._best() is None and p.committed is None


def test_approaching_commits_only_with_a_verifier_and_the_flag():
    for flag, v, expected in [(True, "v", 0), (False, "v", None), (True, None, None)]:
        extra = ["policy.verification.commit_approach=" + str(flag).lower()]
        p = policy_for(extra=extra, verifier_=verifier([0.55]) if v else None)
        p.memory.candidates.append(candidate(2.5, np.linspace(0.1, 0.6, 20)))
        d = p._decide((0.0, 0.0, 0.0))
        assert d.kind == "approach" and p.committed == expected, (flag, v, d)


# ---------------------------------------------------------------- asking from afar
def test_whole_object_in_view_from_afar_is_asked_and_a_yes_needs_no_second_call():
    v = verifier([0.95])
    p = policy_for(extra=LAST_METRE, verifier_=v)
    c = candidate(1.2, np.linspace(0.2, 0.6, 20))
    p.memory.candidates.append(c)
    d = p._decide((0.0, 0.0, 0.0))                                   # 1.2 m away: outside the stop range
    assert v.backend.calls == 1 and c.confirmed and d.kind == "approach"
    assert v.events[-1].dist_m == pytest.approx(1.2, abs=1e-3)
    d = p._decide((0.4, 0.0, 0.0))                                   # 0.8 m: in the ring, in view
    assert d.kind == "stop" and "accepted from afar" in d.reason and v.backend.calls == 1


def test_a_box_cut_by_the_image_border_is_not_asked_from_afar():
    v = verifier([0.95])
    p = policy_for(extra=LAST_METRE, verifier_=v)
    p.memory.candidates.append(candidate(1.2, np.linspace(0.2, 0.6, 20), box=(0, 40, 50, 90)))
    d = p._decide((0.0, 0.0, 0.0))
    assert v.backend.calls == 0 and d.kind == "approach"


def test_a_no_from_afar_rejects_the_candidate():
    v = verifier([0.05])
    p = policy_for(extra=LAST_METRE, verifier_=v)
    c = candidate(1.2, np.linspace(0.2, 0.6, 20))
    p.memory.candidates.append(c)
    p._decide((0.0, 0.0, 0.0))
    assert v.backend.calls == 1 and c.rejected and p._best() is None


def test_an_uncertain_answer_from_afar_does_not_use_up_the_wait_budget_while_approaching():
    """The verifier rejects a candidate after max_wait_steps checks without a new call; asked from afar only when a
    call would be made, the approach itself does not count as waiting."""
    v = verifier([0.55], max_wait_steps=2)
    p = policy_for(extra=LAST_METRE, verifier_=v)
    c = candidate(1.4, np.linspace(0.2, 0.6, 20))
    p.memory.candidates.append(c)
    for i, x in enumerate([0.0, 0.05, 0.1, 0.15, 0.2]):             # same box: no grown view, no new call
        p.step_idx = c.last_step = i
        assert p._decide((x, 0.0, 0.0)).kind == "approach"
    assert v.backend.calls == 1 and not c.rejected


# ---------------------------------------------------------------- full loop on MockEnv
def _loop(embodiment, extra, verify_p=None, steps=250):
    cfg = small_cfg(embodiment, steps=steps, extra=extra)
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    if verify_p is None:
        policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), ["chair"],
                                 env.K)
        res, _ = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", use_oracle_aux=True)
        return res, policy
    from kznav.run.config import compose_config

    cfg = {**cfg, "backend": compose_config(["backend=mock", "backend.mock.verify_mode=constant",
                                             f"backend.mock.verify_p={verify_p}"])["backend"]}
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=TeeSink(calls))
    lex = Lexicon.load(Path(__file__).resolve().parents[1] / "lexicon" / "hm3d_objectnav.yaml")
    goal = parse_command(backend, calls, command_id="c0", text="Орындық қайда? Тауып бер.", lang="kk",
                         category="chair", word_type="universal", lexicon=lex)
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]),
                             goal.hypotheses, env.K)
    policy.query_weights = goal.weights
    policy.verifier = make_verifier(cfg, backend, goal.target, "kk", goal.hypotheses, goal.weights)
    res, _ = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", use_oracle_aux=True,
                         language=goal, calls=calls)
    return res, policy


@pytest.mark.parametrize("embodiment", ["wheeled_vacuum", "habitat_agent"])
def test_visible_stop_mode_finds_the_goal_without_a_verifier(embodiment):
    res, _ = _loop(embodiment, ["policy.approach.stop_mode=visible"])
    assert res.outcome == "success", (res.outcome, res.termination)


def test_last_metre_with_a_positive_verifier_succeeds():
    res, policy = _loop("wheeled_vacuum", LAST_METRE, verify_p=0.95)
    assert res.outcome == "success", (res.outcome, res.termination)
    assert res.goal_ever_verified_positive and res.vlm_calls_verify >= 1
