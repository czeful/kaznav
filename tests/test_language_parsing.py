import json
import math

import pytest
from pydantic import ValidationError

from kznav.language.goal_parsing import ParseLimits, validate_goal_text
from kznav.language.jsonx import JSONExtractionError, extract_json_object
from kznav.language.logprobs import (
    assert_options_distinguishable,
    label_from_text,
    option_probabilities,
)
from kznav.language.schemas import GoalHypotheses

KK_OPTIONS = {"yes": ["иә", "ия"], "no": ["жоқ"]}
EN_OPTIONS = {"yes": ["yes"], "no": ["no"]}


class TestJsonExtraction:
    @pytest.mark.parametrize("raw,expected", [
        ('{"a": 1, "b": [1,2,3,]}', {"a": 1, "b": [1, 2, 3]}),
        ('<think>ойлаймын...</think>\n```json\n{"target": "қазан"}\n```', {"target": "қазан"}),
        ('Here is the answer:\n{"x": "y"}\nHope that helps!', {"x": "y"}),
        ('{"s": "brace } inside string", "n": 2}', {"s": "brace } inside string", "n": 2}),
    ])
    def test_recovers(self, raw, expected):
        assert extract_json_object(raw) == expected

    @pytest.mark.parametrize("raw", ["no json here at all", '{"a": 1', "[1, 2, 3]"])
    def test_fails(self, raw):
        with pytest.raises(JSONExtractionError):
            extract_json_object(raw)


def goal_json(**over):
    d = {"target": "қазан", "hypotheses": [{"en": "cauldron", "w": 0.6}, {"en": "cooking pot", "w": 0.3},
                                           {"en": "wok", "w": 0.1}],
         "attributes": {"material": ["cast iron"], "color": [], "size": ["large"]},
         "rooms": ["kitchen"], "landmarks": []}
    d.update(over)
    return json.dumps(d, ensure_ascii=False)


class TestGoalSchema:
    def test_valid_and_normalised(self):
        g = validate_goal_text(goal_json(hypotheses=[{"en": "wok", "w": 0.2}, {"en": "cauldron", "w": 0.6}]),
                               ParseLimits())
        assert g.texts == ["cauldron", "wok"]
        assert sum(g.weights) == pytest.approx(1.0)
        assert g.weights == pytest.approx([0.75, 0.25])

    def test_cyrillic_hypothesis_rejected(self):
        with pytest.raises(ValidationError, match="English"):
            validate_goal_text(goal_json(hypotheses=[{"en": "қазан", "w": 1.0}]), ParseLimits())

    def test_weight_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            validate_goal_text(goal_json(hypotheses=[{"en": "pot", "w": 60}]), ParseLimits())

    def test_duplicates_merged_case_insensitively(self):
        g = validate_goal_text(goal_json(hypotheses=[{"en": "Pot", "w": 0.3}, {"en": "pot", "w": 0.2},
                                                     {"en": "cauldron", "w": 0.5}]), ParseLimits())
        assert [t.casefold() for t in g.texts] == ["pot", "cauldron"]
        assert g.weights == pytest.approx([0.5, 0.5])

    def test_zero_weights_become_uniform(self):
        g = validate_goal_text(goal_json(hypotheses=[{"en": "a", "w": 0}, {"en": "b", "w": 0}]), ParseLimits())
        assert g.weights == pytest.approx([0.5, 0.5])

    @pytest.mark.parametrize("bad", [
        {"hypotheses": []},
        {"target": "   "},
    ])
    def test_structural_errors(self, bad):
        with pytest.raises(ValidationError):
            validate_goal_text(goal_json(**bad), ParseLimits())

    def test_missing_target(self):
        d = json.loads(goal_json())
        del d["target"]
        with pytest.raises(ValidationError):
            GoalHypotheses.model_validate(d)

    def test_hypothesis_count_limits(self):
        with pytest.raises(ValueError, match="distinct hypotheses"):
            validate_goal_text(goal_json(), ParseLimits(min_hypotheses=4))

    def test_optional_fields_default(self):
        g = validate_goal_text('{"target": "x", "hypotheses": [{"en": "thing", "w": 1}]}', ParseLimits())
        assert g.rooms == [] and g.attributes.color == []


class TestOptionProbabilities:
    def test_first_letter_tokens(self):
        top = {"и": math.log(0.7), "ж": math.log(0.2), "«": math.log(0.05)}
        op = option_probabilities(top, KK_OPTIONS)
        assert op.probs["yes"] == pytest.approx(0.7 / 0.9)
        assert op.covered == pytest.approx(0.9)

    def test_whole_word_and_case(self):
        top = {"yes": math.log(0.6), "Yes": math.log(0.2), " no": math.log(0.1), "maybe": math.log(0.05)}
        op = option_probabilities(top, EN_OPTIONS)
        assert op.probs["yes"] == pytest.approx(0.8 / 0.9)
        assert op.matched["yes"] == ["yes", "Yes"]

    def test_nothing_matched(self):
        op = option_probabilities({"«": -0.1, "The": -2.0}, EN_OPTIONS)
        assert op.probs == {} and op.p("yes") is None and op.covered == 0.0

    def test_ambiguous_options_rejected(self):
        with pytest.raises(ValueError, match="first letter"):
            assert_options_distinguishable({"yes": ["да"], "no": ["два"]})

    def test_label_from_text(self):
        assert label_from_text("Иә.", KK_OPTIONS) == "yes"
        assert label_from_text("«жоқ»", KK_OPTIONS) == "no"
        assert label_from_text("maybe", EN_OPTIONS) is None
        assert label_from_text("", EN_OPTIONS) is None
