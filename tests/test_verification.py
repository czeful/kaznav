"""Verify-before-stop: verifier decisions, candidate memory across hypotheses, and the full loop on MockEnv."""
from pathlib import Path

import numpy as np
import pytest

from kznav.language.base import InvalidOutputError, ListSink, TeeSink
from kznav.language.goal import GoalCache, LanguageGoal, parse_command
from kznav.language.lexicon import Lexicon
from kznav.language.registry import create_backend
from kznav.language.schemas import VerificationResult, VerifyQuery
from kznav.perception.base import Detection, create_detector, create_segmenter
from kznav.perception.camera import Intrinsics
from kznav.perception.objects import DetectionParams, ObjectCandidate, ObjectMemory
from kznav.policy.belief import BeliefParams, HypothesisBelief
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.policy.verification import Verifier, VerifyParams, crop_box, make_verifier
from kznav.run.episode import Executor, run_episode
from kznav.sim.mock_env import MockEnv, room_with_boxes

from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg

RGB = np.zeros((120, 160, 3), dtype=np.uint8)


class FakeBackend:
    def __init__(self, ps):
        self.ps = list(ps)
        self.calls = 0

    def verify_candidate(self, image, query):
        p = self.ps[min(self.calls, len(self.ps) - 1)]
        self.calls += 1
        if p == "error":
            raise InvalidOutputError("boom")
        return VerificationResult(p, "logprobs" if p is not None else "unavailable", 1.0, "", None)


def params(**kw):
    d = dict(stop_threshold=0.6, reject_below=0.25, base_rate=0.5, max_calls=3, regrow_factor=2.0,
             max_wait_steps=2, crop_padding=0.25, min_crop_px=16, calibrated=False)
    d.update(kw)
    return VerifyParams(**d)


def cand(box=(40, 40, 60, 60), scores=None):
    return ObjectCandidate(0, 0, np.zeros(3), np.zeros((1, 2)), 0.5, 1, 0, 0, scores=scores or {0: 0.5}, last_box=box)


def test_crop_box_pads_and_clamps():
    img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    crop, area = crop_box(img, (90, 40, 110, 60), padding=0.5, min_px=8)
    assert crop.shape[:2] == (40, 40) and area == 400
    crop, _ = crop_box(img, (0, 0, 4, 4), padding=0.25, min_px=16)      # tiny box at the corner
    assert crop.shape[:2] == (10, 10)                                     # min size, clamped to the image


def test_confident_yes_stops_after_one_call():
    v = Verifier(FakeBackend([0.95]), VerifyQuery("банан", "kk"), params(), None, view_discount=0.5)
    ev = v.check(cand(), RGB, 0, 1)
    assert ev.called and ev.decision == "stop" and ev.p_candidate > 0.6


def test_confident_no_rejects_and_marks_candidate():
    c = cand()
    v = Verifier(FakeBackend([0.05]), VerifyQuery("банан", "kk"), params(), None)
    assert v.check(c, RGB, 0, 1).decision == "reject" and c.rejected


def test_uncertain_waits_for_a_bigger_view_then_reverifies():
    c = cand(box=(40, 40, 60, 60))
    fb = FakeBackend([0.55, 0.55, 0.9])
    v = Verifier(fb, VerifyQuery("x", "kk"), params(), None)
    assert v.check(c, RGB, 0, 1).decision == "wait" and fb.calls == 1
    assert v.check(c, RGB, 1, 1).decision == "wait" and fb.calls == 1          # same view: no new call
    c.last_box = (30, 30, 60, 60)                                               # 2.25x area: closer view
    ev = v.check(c, RGB, 2, 1)
    assert fb.calls == 2 and ev.called


def test_no_new_view_eventually_rejects():
    c = cand()
    v = Verifier(FakeBackend([0.55]), VerifyQuery("x", "kk"), params(max_wait_steps=2), None)
    decisions = [v.check(c, RGB, i, 1).decision for i in range(4)]
    assert decisions == ["wait", "wait", "wait", "reject"]


def test_missing_logprobs_and_errors_give_no_evidence():
    c = cand()
    v = Verifier(FakeBackend([None, "error"]), VerifyQuery("x", "kk"), params(), None)
    ev = v.check(c, RGB, 0, 1)
    assert ev.p_raw is None and ev.source == "unavailable" and ev.decision == "wait" and ev.p_candidate == 0.5
    c.last_box = (20, 20, 60, 60)
    ev = v.check(c, RGB, 1, 1)
    assert ev.source == "error" and not c.verification


def test_call_budget_rejects():
    c = cand(box=(50, 50, 52, 52))
    v = Verifier(FakeBackend([0.55]), VerifyQuery("x", "kk"), params(max_calls=2, max_wait_steps=10), None)
    v.check(c, RGB, 0, 1)
    c.last_box = (40, 40, 60, 60)
    assert v.check(c, RGB, 1, 1).decision == "reject"


def test_positive_verification_raises_weight_of_the_matching_hypothesis():
    belief = HypothesisBelief(["a", "b"], [0.5, 0.5], BeliefParams(0.7, 0.1, 0.5, weight_floor=0.01))
    v = Verifier(FakeBackend([0.95]), VerifyQuery("x", "kk"), params(), belief)
    v.check(cand(scores={1: 0.6}), RGB, 0, 2)          # detected only by hypothesis "b"
    w = v.weights
    assert w[1] > 0.5 > w[0]


def test_memory_merges_across_queries_and_skips_rejected():
    K = Intrinsics.from_hfov(90.0, 160, 120)
    mem = ObjectMemory(DetectionParams(0.1, 5, 0.3, 200, 5.0), K)
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    mask = np.zeros((120, 160), dtype=bool)
    mask[50:70, 70:90] = True
    T = np.eye(4)
    a = mem.add(Detection((70, 50, 90, 70), 0.4, 0, "a"), mask, depth, T, 0)
    b = mem.add(Detection((68, 48, 92, 72), 0.6, 1, "b"), mask, depth, T, 1)
    assert a is b and len(mem.candidates) == 1
    assert a.scores == {0: 0.4, 1: 0.6} and a.last_box == (68, 48, 92, 72)
    np.testing.assert_allclose(a.match_vector(2), [0.4 / 0.6, 1.0])
    assert a.priority([1.0, 0.0]) == pytest.approx(0.4)
    a.rejected = True
    assert mem.best() is None


# ---------------------------------------------------------------- full loop on MockEnv
def _run(verify_p, embodiment="habitat_agent", steps=250):
    cfg = small_cfg(embodiment, steps=steps)
    cfg = {**cfg, "backend": {**_mock_spec(verify_p)}}
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=TeeSink(calls))
    lex = Lexicon.load(Path(__file__).resolve().parents[1] / "lexicon" / "hm3d_objectnav.yaml")
    goal = parse_command(backend, calls, command_id="c0", text="Орындық қайда? Тауып бер.", lang="kk",
                         category="chair", word_type="universal", lexicon=lex)
    assert goal.parse_ok and goal.hypotheses
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]),
                             goal.hypotheses, env.K)
    policy.query_weights = goal.weights
    policy.verifier = make_verifier(cfg, backend, goal.target, "kk", goal.hypotheses, goal.weights)
    res, trace = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", use_oracle_aux=True,
                             language=goal, calls=calls)
    return res, policy


def _mock_spec(p):
    from kznav.run.config import compose_config

    return compose_config(["backend=mock", "backend.mock.verify_mode=constant", f"backend.mock.verify_p={p}"])["backend"]


def test_loop_with_positive_verifier_succeeds_and_counts_calls():
    res, policy = _run(0.95)
    assert res.outcome == "success", (res.outcome, res.termination)
    assert res.goal_ever_verified_positive
    assert res.vlm_calls_verify >= 1 and res.vlm_calls == res.vlm_calls_verify + 1   # + the parse call
    assert res.parse_ok and res.n_hypotheses == len(policy.queries)


def test_loop_with_negative_verifier_never_stops_at_the_rejected_goal():
    res, policy = _run(0.05, steps=150)
    assert not res.success and res.termination != "stop_called"
    assert res.goal_ever_detected and not res.goal_ever_verified_positive
    assert any(c.rejected for c in policy.memory.candidates)


def test_parse_failure_is_a_model_error():
    cfg = small_cfg("habitat_agent")
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), [], env.K)
    goal = LanguageGoal("c0", "???", "kk", "chair", "universal", False, 3, error="invalid JSON")
    res, _ = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", language=goal)
    assert res.termination == "model_error" and res.outcome == "model_failure" and not res.parse_ok


def test_goal_cache_parses_each_command_once():
    calls = ListSink()
    backend = create_backend(_mock_spec(0.5), sink=calls)
    cache = GoalCache(backend, calls, None)
    a = cache.get(command_id="a", text="Орындық қайда?", lang="kk", category="chair", word_type="universal")
    b = cache.get(command_id="b", text="Орындық қайда?", lang="kk", category="chair", word_type="universal")
    assert len(calls.records) == 1 and a.hypotheses == b.hypotheses and b.command_id == "b"
    assert b.parse_record is calls.records[0]


class OnceDetector:
    """Oracle that fires only on the first frame with a sizeable detection (a transient false positive)."""

    def __init__(self, inner, min_area_px=300):
        self.inner, self.fired, self.min_area = inner, False, min_area_px

    def detect(self, rgb, queries, aux=None):
        if self.fired:
            return []
        dets = [d for d in self.inner.detect(rgb, queries, aux)
                if (d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1]) >= self.min_area]
        self.fired = bool(dets)
        return dets


@pytest.mark.parametrize("embodiment", ["wheeled_vacuum", "habitat_agent"])
def test_candidate_never_seen_again_is_given_up_not_livelocked(embodiment):
    cfg = small_cfg(embodiment, steps=200)
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    policy = ObjectNavPolicy(cfg, OnceDetector(create_detector(cfg["detector"])), create_segmenter(cfg["segmenter"]),
                             ["chair"], env.K)
    res, trace = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", use_oracle_aux=True)
    kinds = [d["kind"] for d in trace.decisions]
    assert policy.memory.candidates and policy.memory.candidates[0].lost
    assert "face" in kinds and res.termination != "stop_called"
    # after giving up it goes back to exploring (or runs out of frontiers) instead of idling at the candidate
    last_face = max(i for i, k in enumerate(kinds) if k == "face")
    assert all(k != "approach" for k in kinds[last_face + 1:])


def test_tiny_box_gets_no_call_until_it_is_big_enough():
    c = cand(box=(50, 50, 63, 63))                                          # 13 x 13 px: too small to judge
    fb = FakeBackend([0.9])
    v = Verifier(fb, VerifyQuery("x", "kk"), params(min_box_side_px=32), None)
    ev = v.check(c, RGB, 0, 1)
    assert fb.calls == 0 and not ev.called and ev.decision == "wait"
    c.last_box = (30, 30, 70, 70)                                           # 40 x 40 px
    ev = v.check(c, RGB, 1, 1)
    assert fb.calls == 1 and ev.decision == "stop"
