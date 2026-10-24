"""D-078: the parse replayed from recorded calls (spoken commands, cascade), verification by the live backend."""
import json

import numpy as np
import pytest

from kznav.language.backends.parse_replay import ParseReplayBackend, rekey_parse_record
from kznav.language.base import BackendError, ListSink
from kznav.language.goal import GoalSettings, parse_command
from kznav.language.lexicon import Lexicon
from kznav.language.registry import create_backend
from kznav.language.schemas import VerifyQuery

COMMAND = "Төсек қайда? Шаршап тұрмын"
HEARD = "Төсек қайда"


def record(backend_spec, tmp_path, prompt, text, name="src.jsonl"):
    """Parse ``text`` with the mock under ``prompt`` and save the call, as a source run would."""
    spec = backend_spec("mock")
    spec["prompts"] = dict(spec["prompts"], parse_goal=prompt)
    sink = ListSink()
    goal = create_backend(spec, sink=sink).parse_goal(text, "kk")
    path = tmp_path / name
    path.write_text(json.dumps(sink.records[-1].to_json_dict(), ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    return goal, sink.records[-1], path


def live(backend_spec, sources, prompt, sink):
    spec = backend_spec("mock")
    spec["prompts"] = dict(spec["prompts"], parse_goal="default/parse_goal/v4")
    spec["parse_replay"] = {"sources": [str(s) for s in sources], "prompt": prompt}
    return create_backend(spec, sink=sink)


def test_parse_is_replayed_and_verification_stays_live(backend_spec, tmp_path):
    goal, _, src = record(backend_spec, tmp_path, "speech/parse_goal/v1", COMMAND)
    sink = ListSink()
    be = live(backend_spec, [src], "speech/parse_goal/v1", sink)
    assert isinstance(be, ParseReplayBackend)
    be.context.episode_id = "ep1"
    assert be.parse_goal(COMMAND, "kk") == goal
    rec = sink.records[-1]
    assert rec.backend == "replay" and rec.prompt_ref == "speech/parse_goal/v1" and rec.context.episode_id == "ep1"
    assert rec.inputs["replayed_from"]["model_id"] == "mock-0"
    be.verify_candidate(np.full((32, 32, 3), 90, np.uint8), VerifyQuery("bed", "en"))
    assert sink.records[-1].backend == "mock" and sink.records[-1].method == "verify_candidate"
    assert be.verify_prompt is be.inner.verify_prompt and be.parse_prompt.ref == "speech/parse_goal/v1"
    cfg = be.describe_config()
    assert cfg["name"] == "mock" and cfg["parse_replay"]["prompt"] == "speech/parse_goal/v1"


def test_a_command_without_a_record_is_not_understood_and_never_reaches_the_live_model(backend_spec, tmp_path):
    _, _, src = record(backend_spec, tmp_path, "speech/parse_goal/v1", COMMAND)
    sink = ListSink()
    be = live(backend_spec, [src], "speech/parse_goal/v1", sink)
    with pytest.raises(BackendError, match="no recorded"):
        be.parse_goal("Диванды тап", "kk")
    lex = Lexicon.load(__import__("pathlib").Path(__file__).resolve().parents[1] / "lexicon/hm3d_objectnav.yaml")
    g = parse_command(be, sink, command_id="x", text="Диванды тап", lang="kk", category="sofa", word_type="universal",
                      lexicon=lex, settings=GoalSettings())
    assert not g.parse_ok and all(r.backend != "mock" for r in sink.records if r.method == "parse_goal")
    with pytest.raises(BackendError):                   # the same text under the text arms' prompt: another key
        live(backend_spec, [src], "default/parse_goal/v4", ListSink()).parse_goal(COMMAND, "kk")


def test_cascade_records_are_found_by_the_command_text(backend_spec, tmp_path):
    goal, rec, _ = record(backend_spec, tmp_path, "default/parse_goal/v4", HEARD)
    d = rekey_parse_record(rec.to_json_dict(), COMMAND, "kk", transcript_cer=0.1)
    assert d["inputs"]["parsed_text"] == HEARD and d["inputs"]["parsed_text_input_key"] == rec.input_key
    src = tmp_path / "cascade.jsonl"
    src.write_text(json.dumps(d, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    be = live(backend_spec, [src], "default/parse_goal/v4", ListSink())
    assert be.parse_goal(COMMAND, "kk") == goal
    with pytest.raises(BackendError):
        be.parse_goal(HEARD, "kk")


def test_parse_replay_needs_sources_and_prompt(backend_spec):
    spec = backend_spec("mock")
    spec["parse_replay"] = {"sources": [], "prompt": "speech/parse_goal/v1"}
    with pytest.raises(ValueError, match="parse_replay"):
        create_backend(spec)
