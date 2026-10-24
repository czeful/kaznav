"""Goal ablations (D-038 / D-039): hypothesis selection, translate-first modes, translator logging and replay."""
import json
from pathlib import Path

import pytest

from kznav.language.base import BackendConfigError, ListSink, TeeSink
from kznav.language.goal import GoalCache, GoalSettings, make_goal_cache, parse_command, select_hypotheses
from kznav.language.lexicon import Lexicon
from kznav.language.registry import create_backend
from kznav.language.translation import MockTranslator, TranslationError, create_translator
from kznav.perception.base import create_detector, create_segmenter
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.policy.verification import make_verifier
from kznav.run.config import compose_config
from kznav.run.episode import Executor, run_episode
from kznav.sim.mock_env import MockEnv, room_with_boxes

from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg

REPO = Path(__file__).resolve().parents[1]
CMD = "Орындық қайда? Тауып бер."
MOCK_TR = {"name": "mock_mt", "type": "mock", "model_id": "mock-mt-0", "target_lang": "en",
           "lang_codes": {"kk": "kaz_Cyrl", "kk_ru": "kaz_Cyrl", "en": "eng_Latn"},
           "mock": {"lexicon": "lexicon/hm3d_objectnav.yaml"}}


@pytest.fixture
def lexicon():
    return Lexicon.load(REPO / "lexicon" / "hm3d_objectnav.yaml")


def mock_backend(sink):
    return create_backend(compose_config(["backend=mock"])["backend"], sink=sink)


def parse(settings, translator=None, lexicon=None, text=CMD, lang="kk"):
    calls = ListSink()
    backend = mock_backend(calls)
    return parse_command(backend, calls, command_id="c0", text=text, lang=lang, category="chair",
                         word_type="universal", lexicon=lexicon, settings=settings, translator=translator), calls


# ---------------------------------------------------------------- hypothesis selection
class TestSelectHypotheses:
    def test_all_by_default(self):
        assert select_hypotheses(["a", "b"], [0.7, 0.3], None) == (["a", "b"], [0.7, 0.3])
        assert select_hypotheses(["a", "b"], [0.7, 0.3], 5) == (["a", "b"], [0.7, 0.3])

    def test_top1_gets_full_weight(self):
        assert select_hypotheses(["a", "b", "c"], [0.2, 0.5, 0.3], 1) == (["b"], [1.0])

    def test_ties_keep_parser_order(self):
        assert select_hypotheses(["a", "b", "c"], [0.4, 0.4, 0.2], 1) == (["a"], [1.0])

    def test_topk_renormalised_and_sorted(self):
        t, w = select_hypotheses(["a", "b", "c"], [0.2, 0.5, 0.3], 2)
        assert t == ["b", "c"] and w == pytest.approx([0.625, 0.375])

    def test_invalid_settings(self):
        with pytest.raises(BackendConfigError):
            GoalSettings(max_hypotheses=0)
        with pytest.raises(BackendConfigError):
            GoalSettings(translation="sentence")


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("goal,mode,max_h", [("multi", "none", None), ("top1", "none", 1),
                                             ("mt_command", "command", None), ("mt_word", "word", None)])
def test_goal_configs_compose(goal, mode, max_h):
    g = compose_config([f"goal={goal}"])["goal"]
    st = GoalSettings.from_config(g)
    assert (st.name, st.translation, st.max_hypotheses) == (goal, mode, max_h)
    assert (g["translator"] is None) == (mode == "none")


def test_mt_word_inherits_the_translator():
    c, t = (compose_config([f"goal={g}"])["goal"]["translator"] for g in ("mt_command", "mt_word"))
    assert c["model_id"] == t["model_id"] == "facebook/nllb-200-distilled-600M"
    assert t["lang_codes"]["kk"] == "kaz_Cyrl" and t["target_lang"] == "en"
    assert t["generation"]["max_new_tokens"] < c["generation"]["max_new_tokens"]


def test_translation_and_translator_must_agree():
    with pytest.raises(BackendConfigError, match="needs goal.translator"):
        GoalSettings.from_config({"name": "x", "translation": "command", "translator": None})
    with pytest.raises(BackendConfigError, match="translation is 'none'"):
        GoalSettings.from_config({"name": "x", "translation": "none", "translator": MOCK_TR})
    with pytest.raises(BackendConfigError, match="needs a translator"):
        parse(GoalSettings("mt_command", translation="command"))


def test_no_translator_configured():
    assert create_translator(None) is None and create_translator({"type": "none"}) is None


def test_translator_rejects_unknown_language():
    tr = MockTranslator(MOCK_TR)
    with pytest.raises(BackendConfigError, match="no model language code"):
        tr.translate("x", "ru")


# ---------------------------------------------------------------- the modes
def test_multi_is_unchanged(lexicon):
    g, calls = parse(None, lexicon=lexicon)
    assert g.goal_mode == "multi" and g.effective_lang == "kk" and g.translation == ""
    assert g.hypotheses == g.parsed_hypotheses == ["chair", "armchair"] and len(calls.records) == 1


def test_top1_keeps_the_parse_but_uses_one_hypothesis(lexicon):
    g, calls = parse(GoalSettings("top1", max_hypotheses=1), lexicon=lexicon)
    assert g.hypotheses == ["chair"] and g.weights == [1.0]
    assert g.parsed_hypotheses == ["chair", "armchair"]
    assert g.hypothesis_match is True and g.parse_record is calls.records[0]


def test_mt_command_parses_the_translation_in_english(lexicon):
    sink = ListSink()
    tr = MockTranslator(MOCK_TR, sink=sink)
    g, calls = parse(GoalSettings("mt_command", translation="command"), tr, lexicon)
    assert g.parse_ok and g.translation == "chair" and g.effective_lang == "en" and g.lang == "kk"
    assert calls.records[0].inputs == {"text": "chair", "lang": "en"}      # the parser never saw the Kazakh text
    assert g.target == "chair" and g.target_match is True                  # matched against the English forms
    rec = g.translation_record
    assert rec is sink.records[0] and rec.method == "translate" and rec.ok and rec.output == {"text": "chair"}
    assert rec.inputs["src_code"] == "kaz_Cyrl" and rec.inputs["tgt_code"] == "eng_Latn"


def test_mt_word_translates_the_given_name_without_a_vlm_call(lexicon):
    sink, calls = ListSink(), ListSink()
    g = parse_command(mock_backend(calls), calls, command_id="c0", text=CMD, lang="kk", category="chair",
                      word_type="universal", lexicon=lexicon, settings=GoalSettings("mt_word", translation="word"),
                      translator=MockTranslator(MOCK_TR, sink=sink), word="орындық", word_lang="kk")
    assert not calls.records and g.parse_record is None and g.parse_attempts == 0
    assert g.target == "орындық" and g.effective_lang == "kk"              # verification stays in Kazakh
    assert g.hypotheses == ["chair"] and g.weights == [1.0] and g.translation == "chair"
    assert g.hypothesis_match is True and g.target_match is None
    assert sink.records[0].inputs["text"] == "орындық" and sink.records[0].inputs["src_code"] == "kaz_Cyrl"


def test_mt_word_needs_a_name_in_another_language():
    st, tr = GoalSettings("mt_word", translation="word"), MockTranslator(MOCK_TR)
    with pytest.raises(BackendConfigError, match="needs the object name"):
        parse(st, tr)
    calls = ListSink()
    with pytest.raises(BackendConfigError, match="already in"):
        parse_command(mock_backend(calls), calls, command_id="c", text="Where is the chair?", lang="en",
                      category="chair", word_type="universal", lexicon=None, settings=st, translator=tr,
                      word="chair", word_lang="en")


class EmptyTranslator(MockTranslator):
    def _translate(self, text, src_code, tgt_code, rec):
        return "   "


@pytest.mark.parametrize("mode", ["command", "word"])
def test_failed_translation_is_a_model_error(mode):
    sink, calls = ListSink(), ListSink()
    g = parse_command(mock_backend(calls), calls, command_id="c0", text=CMD, lang="kk", category="chair",
                      word_type="universal", lexicon=None, settings=GoalSettings("mt", translation=mode),
                      translator=EmptyTranslator(MOCK_TR, sink=sink), word="орындық", word_lang="kk")
    assert not g.parse_ok and "empty translation" in g.error
    assert g.translation_record is sink.records[0] and not g.translation_record.ok


def test_goal_cache_translates_each_command_once():
    calls, tr_sink = ListSink(), ListSink()
    backend = mock_backend(calls)
    goals = make_goal_cache(compose_config(["goal=mt_command"])["goal"] | {"translator": MOCK_TR}, backend, calls,
                            None, translator_sink=tr_sink)
    assert goals.translator.context is backend.context
    backend.context.episode_id = "ep7"
    a = goals.get(command_id="a", text=CMD, lang="kk", category="chair", word_type="universal")
    b = goals.get(command_id="b", text=CMD, lang="kk", category="chair", word_type="universal")
    assert len(tr_sink.records) == 1 and len(calls.records) == 1
    assert a.translation_record is b.translation_record and b.command_id == "b"
    assert tr_sink.records[0].context.episode_id == "ep7"


# ---------------------------------------------------------------- replay
def test_replay_translator_round_trip(tmp_path):
    sink = ListSink()
    out, _ = MockTranslator(MOCK_TR, sink=sink).translate(CMD, "kk")
    log = tmp_path / "vlm_calls.jsonl"
    log.write_text("\n".join(json.dumps(r.to_json_dict(), ensure_ascii=False) for r in sink.records) + "\n",
                   encoding="utf-8")
    replay = create_translator({**MOCK_TR, "type": "replay", "replay": {"sources": [str(log)]}})
    again, rec = replay.translate(CMD, "kk")
    assert again == out and rec.inputs["replayed_from"]["started_utc"] == sink.records[0].started_utc
    with pytest.raises(TranslationError):
        replay.translate("Басқа сөйлем", "kk")
    changed = {**MOCK_TR, "type": "replay", "replay": {"sources": [str(log)]}, "generation": {"num_beams": 4}}
    with pytest.raises(TranslationError):                                   # other settings -> other input key
        create_translator(changed).translate(CMD, "kk")


def test_replay_translator_needs_sources():
    with pytest.raises(BackendConfigError):
        create_translator({**MOCK_TR, "type": "replay"})


# ---------------------------------------------------------------- full loop on MockEnv
@pytest.mark.parametrize("goal", ["top1", "mt_word", "mt_command"])
def test_loop_runs_with_each_ablation(goal, lexicon):
    cfg = small_cfg("habitat_agent")
    cfg = {**cfg, "backend": compose_config(["backend=mock", "backend.mock.verify_mode=constant",
                                             "backend.mock.verify_p=0.95"])["backend"]}
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=TeeSink(calls))
    goal_cfg = compose_config([f"goal={goal}"])["goal"]
    if goal_cfg["translator"]:
        goal_cfg = goal_cfg | {"translator": MOCK_TR}
    goals = make_goal_cache(goal_cfg, backend, calls, lexicon)
    lg = goals.get(command_id="c0", text=CMD, lang="kk", category="chair", word_type="universal",
                   word="орындық", word_lang="kk")
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]),
                             lg.hypotheses, env.K)
    policy.query_weights = lg.weights
    policy.verifier = make_verifier(cfg, backend, lg.target, lg.effective_lang, lg.hypotheses, lg.weights)
    res, _ = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="smoke", use_oracle_aux=True,
                         language=lg, calls=calls)
    assert res.outcome == "success", (res.outcome, res.termination)
    assert res.goal_mode == goal and res.pipeline_lang == ("en" if goal == "mt_command" else "kk")
    assert res.n_hypotheses == (2 if goal == "mt_command" else 1)
    assert res.vlm_calls == res.vlm_calls_verify + (goal != "mt_word")   # the parse; translation is not a VLM call ...
    assert (res.translation_latency_s > 0) == (goal != "top1")           # ... it has its own cost column
    assert res.translation == ("" if goal == "top1" else "chair")
    verify = [r for r in calls.records if r.method == "verify_candidate"]
    assert verify and all(r.inputs["lang"] == res.pipeline_lang for r in verify)


def test_without_a_goal_cache_the_default_is_multi():
    calls = ListSink()
    g = GoalCache(mock_backend(calls), calls, None).get(command_id="c", text=CMD, lang="kk", category="chair",
                                                       word_type="universal")
    assert g.goal_mode == "multi" and g.translation_record is None
