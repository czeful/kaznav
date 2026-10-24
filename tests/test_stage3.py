"""D-072 (D-070 stage 3): the language model's rooms as a prior in the value map, and stored crops of suspected
candidates asked about before the robot drives to them."""
import numpy as np
import pytest

from kznav.perception.base import ValueEncoder, create_detector, create_segmenter
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.run.episode import Executor, run_episode
from kznav.sim.mock_env import MockEnv, room_with_boxes

from test_last_meter import FakeBackend, _loop, candidate, policy_for
from test_second_look import verifier
from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg


class FixedEncoder(ValueEncoder):
    """0.1 for object texts, 0.5 for room texts; remembers what it was asked."""

    def __init__(self):
        self.asked = []

    def score(self, rgb, texts):
        self.asked.append(list(texts))
        return np.array([0.5 if t == "bedroom" else 0.1 for t in texts])


@pytest.mark.parametrize("weight,expected", [(0.0, 0.1), (0.25, 0.75 * 0.1 + 0.25 * 0.5)])
def test_rooms_enter_the_value_only_with_a_room_weight(weight, expected):
    cfg = small_cfg("habitat_agent", steps=3, extra=[f"++policy.frontier_selection.room_weight={weight}"])
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    enc = FixedEncoder()
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), ["chair"],
                             env.K, value_encoder=enc)
    policy.rooms = ["bedroom"]
    run_episode(env, policy, Executor(env, cfg), episode_for(env, cfg), cfg, run_id="smoke", use_oracle_aux=True)
    assert policy.last_value == pytest.approx(expected)
    assert enc.asked[-1] == (["chair", "bedroom"] if weight else ["chair"])


def suspected_policy(ps):
    v = verifier(ps, new_view_deg=0.0)
    p = policy_for(extra=["perception.detection.fusion.enabled=true", "++policy.verification.recheck_suspected_crops=true"],
                   verifier_=v)
    c = candidate(2.0, np.linspace(0.2, 0.6, 20), conf=0.2)              # suspected: below the 0.45 bar
    c.crop = np.zeros((60, 60, 3), dtype=np.uint8)
    reliable = candidate(3.0, np.linspace(0.2, 0.6, 20), conf=0.8)
    reliable.cid = 1
    reliable.crop = np.zeros((60, 60, 3), dtype=np.uint8)
    p.memory.candidates += [c, reliable]
    return p, v, c, reliable


def test_a_confident_no_on_a_stored_crop_drops_the_suspected_candidate():
    p, v, c, reliable = suspected_policy([0.05])
    p._recheck_suspected()
    assert c.rejected and not reliable.rejected and v.backend.calls == 1      # reliable ones are verified on the spot
    assert v.events[-1].box_xyxy == [] and v.events[-1].called


def test_an_uncertain_answer_on_a_stored_crop_keeps_it_and_its_first_real_view_is_new():
    p, v, c, _ = suspected_policy([0.5])
    p._recheck_suspected()
    assert not c.rejected and v.calls(c.cid) == 1 and v.can_call(c)
    p._recheck_suspected()
    assert v.backend.calls == 1                                                 # asked once per candidate


def test_crops_are_stored_only_with_the_flag():
    for flag in (False, True):
        extra = ["perception.detection.fusion.enabled=true", f"++policy.verification.recheck_suspected_crops={flag}"]
        _, policy = _loop("wheeled_vacuum", extra, verify_p=0.05, steps=60)
        assert any(c.crop is not None for c in policy.memory.candidates) == flag


def test_the_models_short_name_is_added_as_a_query_weighted_like_its_top_hypothesis():
    from kznav.language.goal import with_target_name

    t, w = with_target_name(["bed in a bedroom", "bed with pillows"], [0.6, 0.4], "bed")
    assert t == ["bed in a bedroom", "bed with pillows", "bed"]
    assert w == pytest.approx([0.6 / 1.6, 0.4 / 1.6, 0.6 / 1.6])
    assert with_target_name(["Bed", "bed frame"], [0.5, 0.5], "bed") == (["Bed", "bed frame"], [0.5, 0.5])
    assert with_target_name(["door"], [1.0], "") == (["door"], [1.0])
