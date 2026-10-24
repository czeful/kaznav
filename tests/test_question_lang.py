"""D-071 (D-070 stage 2): the verifier's question in English about the model's own English name of the target."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from kznav.language.base import ListSink
from kznav.language.goal import parse_command
from kznav.language.lexicon import Lexicon
from kznav.language.registry import create_backend
from kznav.language.schemas import GoalHypotheses, VerifyQuery
from kznav.policy.verification import make_verifier, question
from kznav.run.config import compose_config

LEX = Path(__file__).resolve().parents[1] / "lexicon" / "hm3d_objectnav.yaml"


def cfg(mode=None):
    extra = [f"++policy.verification.question_lang={mode}"] if mode else []
    return compose_config(["backend=mock", "backend.prompts.parse_goal=default/parse_goal/v3", *extra])


def test_target_en_is_optional_and_english_only():
    base = {"target": "төсек", "hypotheses": [{"en": "bed", "w": 1.0}]}
    assert GoalHypotheses.model_validate(base).target_en == ""
    assert GoalHypotheses.model_validate({**base, "target_en": "  bed "}).target_en == "bed"
    with pytest.raises(ValidationError):
        GoalHypotheses.model_validate({**base, "target_en": "төсек"})


def test_command_language_question_is_the_default():
    c = cfg()
    assert question(c, "төсек", "kk", "bed", ["bed with pillows"]) == VerifyQuery("төсек", "kk")


def test_english_question_uses_the_models_english_name_then_its_top_hypothesis():
    c = cfg("en")
    assert question(c, "төсек", "kk", "bed", ["bed with pillows"]) == VerifyQuery("bed", "en")
    assert question(c, "төсек", "kk", "", ["bed with pillows", "bed"]) == VerifyQuery("bed with pillows", "en")
    with pytest.raises(ValueError):
        question(cfg("kk"), "төсек", "kk")


def test_parse_v3_gives_target_en_and_the_verifier_asks_in_english():
    c = cfg("en")
    calls = ListSink()
    backend = create_backend(c["backend"], sink=calls)
    g = parse_command(backend, calls, command_id="c0", text="Төсек қайда?", lang="kk", category="bed",
                      word_type="universal", lexicon=Lexicon.load(LEX))
    assert g.parse_ok and g.target_en and g.target_en.isascii() and g.summary()["target_en"] == g.target_en
    v = make_verifier(c, backend, g.target, "kk", g.hypotheses, g.weights, g.target_en)
    assert v.query == VerifyQuery(g.target_en, "en")
    v = make_verifier(cfg(), backend, g.target, "kk", g.hypotheses, g.weights, g.target_en)
    assert v.query == VerifyQuery(g.target, "kk")
