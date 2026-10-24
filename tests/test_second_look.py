"""D-071 (D-070 stage 2b): an undecided verifier answer with no bigger view to come is asked again from a different
side of the object (at least ``second_look_deg`` around it)."""
import math

import numpy as np

from kznav.language.schemas import VerifyQuery
from kznav.policy.verification import Verifier, VerifyParams

from test_last_meter import FakeBackend, candidate, policy_for

RGB = np.zeros((120, 160, 3), dtype=np.uint8)


def verifier(ps, new_view_deg=40.0):
    params = VerifyParams(stop_threshold=0.6, reject_below=0.25, base_rate=0.5, max_calls=3, regrow_factor=2.0,
                          max_wait_steps=2, crop_padding=0.25, min_crop_px=16, calibrated=False,
                          new_view_deg=new_view_deg)
    return Verifier(FakeBackend(ps), VerifyQuery("chair", "en"), params, None, view_discount=0.5, first_view_full=True)


def test_a_view_from_far_enough_around_the_object_allows_a_new_call():
    v = verifier([0.5, 0.9])
    c = candidate(0.5, np.linspace(0.2, 0.6, 20))
    v.check(c, RGB, 0, 1, bearing=math.pi)
    assert v.calls(c.cid) == 1 and v.undecided(c.cid)
    assert not v.can_call(c, math.pi) and not v.can_call(c, math.pi - math.radians(30))
    assert v.can_call(c, math.pi / 2)
    assert v.check(c, RGB, 1, 1, bearing=math.pi / 2).called and v.calls(c.cid) == 2
    off = verifier([0.5, 0.9], new_view_deg=0.0)                       # second look off: a new side is not a new view
    off.check(c, RGB, 0, 1, bearing=math.pi)
    assert not off.can_call(c, math.pi / 2)


def test_an_undecided_answer_sends_the_robot_around_the_object_and_asks_again():
    v = verifier([0.5, 0.9])
    p = policy_for(extra=["++policy.verification.second_look_deg=40"], verifier_=v)
    c = candidate(0.5, np.linspace(0.2, 0.6, 20))
    p.memory.candidates.append(c)
    assert p._decide((0.0, 0.0, 0.0)).kind == "approach" and v.backend.calls == 1     # undecided: wait
    p.step_idx = c.last_step = 1
    d = p._decide((0.0, 0.0, 0.0))                                                     # same view: look elsewhere
    assert d.kind == "approach" and "second look" in d.reason and v.backend.calls == 1
    gx, gy = d.path_xy[-1]
    turn = abs((math.atan2(gy - c.xy[1], gx - c.xy[0]) - math.pi + math.pi) % (2 * math.pi) - math.pi)
    assert math.degrees(turn) >= 50
    p.step_idx = c.last_step = 2
    d = p._decide((0.5, -0.5, 0.0))                                                    # 90 deg around: new call
    assert v.backend.calls == 2 and d.kind == "stop"


def test_without_second_look_the_old_wait_is_kept():
    v = verifier([0.5], new_view_deg=0.0)
    p = policy_for(verifier_=v)
    c = candidate(0.5, np.linspace(0.2, 0.6, 20))
    p.memory.candidates.append(c)
    p._decide((0.0, 0.0, 0.0))
    p.step_idx = c.last_step = 1
    d = p._decide((0.0, 0.0, 0.0))
    assert d.kind == "approach" and "second look" not in d.reason and v.backend.calls == 1


def test_a_second_look_that_takes_too_long_rejects_the_candidate():
    v = verifier([0.5])
    p = policy_for(extra=["++policy.verification.second_look_deg=40", "++policy.verification.second_look_max_steps=2"],
                   verifier_=v)
    c = candidate(0.5, np.linspace(0.2, 0.6, 20))
    p.memory.candidates.append(c)
    p._decide((0.0, 0.0, 0.0))
    for i in (1, 2, 3):
        p.step_idx = c.last_step = i
        p._decide((0.0, 0.0, 0.0))
    assert c.rejected and v.backend.calls == 1
