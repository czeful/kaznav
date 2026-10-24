"""Backend contract tests: mock, OpenAI-compatible (against an in-process HTTP stub), Qolda rules, replay."""
import json
import math
import shutil

import httpx
import numpy as np
import pytest

from kznav.language.backends.openai_compat import OpenAICompatBackend
from kznav.language.backends.qolda import QoldaBackend
from kznav.language.base import BackendConfigError, InvalidOutputError, ListSink, TransportError
from kznav.language.prompts import PromptLibrary
from kznav.language.registry import BACKENDS, create_backend
from kznav.language.schemas import VerifyQuery
from kznav.run.logging import RunLogger

IMG = np.zeros((32, 48, 3), dtype=np.uint8)
IMG2 = np.full((32, 48, 3), 200, dtype=np.uint8)

GOAL = {"target": "қазан", "hypotheses": [{"en": "cauldron", "w": 0.6}, {"en": "cooking pot", "w": 0.4}],
        "attributes": {"material": [], "color": [], "size": []}, "rooms": ["kitchen"], "landmarks": []}


def chat_response(content, top=None):
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    if top is not None:
        choice["logprobs"] = {"content": [{"token": next(iter(top)), "logprob": next(iter(top.values())),
                                           "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top.items()]}]}
    return {"id": "x", "object": "chat.completion", "choices": [choice]}


class Stub:
    """Scripted HTTP server: each request pops the next scripted reply (or raises it)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return httpx.Response(status, json=body)


def qolda(backend_spec, prompt_lib, stub, sink=None, **over):
    spec = backend_spec("qolda")
    spec["http"]["backoff_s"] = 0.0
    spec.update(over)
    return QoldaBackend(spec, prompt_lib, sink or ListSink(), transport=httpx.MockTransport(stub))


class TestMockBackend:
    def test_parse_from_lexicon(self, backend_spec, prompt_lib):
        b = create_backend(backend_spec("mock"), prompts=prompt_lib)
        g = b.parse_goal("Маған орындықты тауып берші", "kk")
        assert g.target == "орындық"
        assert g.texts[:2] == ["chair", "armchair"]
        rec = b.sink.records[-1]
        assert rec.ok and rec.method == "parse_goal" and rec.json_retries == 0

    def test_malformed_first_attempt_exercises_retry(self, backend_spec, prompt_lib):
        spec = backend_spec("mock")
        spec["mock"]["malformed_first_attempt_rate"] = 1.0
        b = create_backend(spec, prompts=prompt_lib)
        g = b.parse_goal("Диванды тап", "kk")
        rec = b.sink.records[-1]
        assert g.texts[0] == "sofa"
        assert rec.json_retries == 1
        assert [a.ok for a in rec.attempts] == [False, True]

    def test_verify_is_deterministic(self, backend_spec, prompt_lib):
        b = create_backend(backend_spec("mock"), prompts=prompt_lib)
        q = VerifyQuery("диван", "kk")
        r1, r2, r3 = b.verify_candidate(IMG, q), b.verify_candidate(IMG, q), b.verify_candidate(IMG2, q)
        assert r1 == r2
        assert r1.p_target != r3.p_target
        assert 0 < r1.p_target < 1 and r1.source == "mock"

    def test_warmup_is_flagged(self, backend_spec, prompt_lib):
        b = create_backend(backend_spec("mock"), prompts=prompt_lib)
        b.warmup(2)
        b.describe(IMG, "hi")
        assert [r.context.warmup for r in b.sink.records] == [True, True, False]


class TestOpenAICompat:
    def test_parse_request_and_no_think(self, backend_spec, prompt_lib):
        stub = Stub([(200, chat_response(json.dumps(GOAL, ensure_ascii=False)))])
        b = qolda(backend_spec, prompt_lib, stub)
        g = b.parse_goal("Ас үйдегі қазанды тап", "kk")
        assert g.texts == ["cauldron", "cooking pot"]
        body = stub.requests[0]
        assert body["enable_thinking"] is False          # top-level, as openai-python extra_body does
        assert body["model"] == "qolda-4b" and body["temperature"] == 0.0
        assert "logprobs" not in body
        rec = b.sink.records[-1]
        assert rec.ok and rec.attempts[0].response == chat_response(json.dumps(GOAL, ensure_ascii=False))

    def test_invalid_json_then_valid(self, backend_spec, prompt_lib):
        stub = Stub([(200, chat_response("Мен таптым!")), (200, chat_response(json.dumps(GOAL, ensure_ascii=False)))])
        b = qolda(backend_spec, prompt_lib, stub)
        b.parse_goal("қазанды тап", "kk")
        rec = b.sink.records[-1]
        assert rec.json_retries == 1 and rec.http_retries == 0
        second = stub.requests[1]["messages"]
        assert second[-2] == {"role": "assistant", "content": "Мен таптым!"}
        assert "схемаға сай емес" in second[-1]["content"]

    def test_json_attempts_exhausted(self, backend_spec, prompt_lib):
        stub = Stub([(200, chat_response("no"))] * 3)
        b = qolda(backend_spec, prompt_lib, stub)
        with pytest.raises(InvalidOutputError):
            b.parse_goal("x", "kk")
        rec = b.sink.records[-1]
        assert not rec.ok and rec.json_retries == 2 and "InvalidOutputError" in rec.error

    def test_http_500_is_retried(self, backend_spec, prompt_lib):
        stub = Stub([(500, {"error": "boom"}), (200, chat_response(json.dumps(GOAL, ensure_ascii=False)))])
        b = qolda(backend_spec, prompt_lib, stub)
        b.parse_goal("x", "kk")
        rec = b.sink.records[-1]
        assert rec.http_retries == 1 and rec.json_retries == 0
        assert [a.kind for a in rec.attempts] == ["http", "http", "json"]
        assert rec.attempts[0].response == {"error": "boom"}

    def test_timeouts_exhaust_attempts(self, backend_spec, prompt_lib):
        stub = Stub([httpx.ReadTimeout("slow")] * 3)
        b = qolda(backend_spec, prompt_lib, stub)
        with pytest.raises(TransportError):
            b.parse_goal("x", "kk")
        rec = b.sink.records[-1]
        assert rec.http_retries == 2 and len(rec.attempts) == 3 and not rec.ok
        assert rec.latency_total_s >= 0

    def test_4xx_not_retried(self, backend_spec, prompt_lib):
        stub = Stub([(400, {"error": "bad request"})])
        b = qolda(backend_spec, prompt_lib, stub)
        with pytest.raises(TransportError, match="non-retryable"):
            b.parse_goal("x", "kk")
        assert len(stub.requests) == 1

    def test_verify_with_logprobs(self, backend_spec, prompt_lib):
        top = {"и": math.log(0.9), "ж": math.log(0.08)}
        stub = Stub([(200, chat_response("иә", top))])
        sink = ListSink()
        b = qolda(backend_spec, prompt_lib, stub, sink)
        r = b.verify_candidate(IMG, VerifyQuery("қазан", "kk_ru"))
        assert r.p_target == pytest.approx(0.9 / 0.98)
        assert r.source == "logprobs" and r.coverage == pytest.approx(0.98) and r.label_from_text is True
        body = stub.requests[0]
        assert body["logprobs"] is True and body["top_logprobs"] == 10
        content = body["messages"][-1]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        rec = sink.records[-1]
        assert rec.inputs["verify_lang"] == "kk" and len(rec.images) == 1

    def test_verify_without_logprobs(self, backend_spec, prompt_lib):
        stub = Stub([(200, chat_response("жоқ"))])
        b = qolda(backend_spec, prompt_lib, stub)
        r = b.verify_candidate(IMG, VerifyQuery("қазан", "kk"))
        assert r.p_target is None and r.source == "unavailable" and r.label_from_text is False


class TestQoldaRules:
    def test_enable_thinking_must_be_explicit(self, backend_spec, prompt_lib):
        with pytest.raises(BackendConfigError, match="explicitly"):
            qolda(backend_spec, prompt_lib, Stub([]), extra_body={})

    def test_think_only_for_control(self, backend_spec, prompt_lib):
        with pytest.raises(BackendConfigError, match="control"):
            qolda(backend_spec, prompt_lib, Stub([]), extra_body={"enable_thinking": True})
        b = qolda(backend_spec, prompt_lib, Stub([]), extra_body={"enable_thinking": True},
                  control_think_measurement=True)
        assert b.extra_body["enable_thinking"] is True

    def test_qwen_thinking_checkpoint_rejected(self, backend_spec, prompt_lib):
        spec = backend_spec("qwen3vl_4b")
        spec["hf_repo"] = "Qwen/Qwen3-VL-4B-Thinking"
        with pytest.raises(BackendConfigError):
            create_backend(spec, prompts=prompt_lib)


class TestRegistry:
    def test_builtin_names(self):
        assert {"mock", "qolda", "qwen_vl", "openai_compat", "replay"} <= set(BACKENDS.names())

    def test_class_path(self, backend_spec, prompt_lib):
        spec = backend_spec("mock")
        del spec["type"]
        spec["class_path"] = "kznav.language.backends.mock:MockBackend"
        assert type(create_backend(spec, prompts=prompt_lib)).__name__ == "MockBackend"

    def test_unknown_type(self, backend_spec, prompt_lib):
        spec = backend_spec("mock")
        spec["type"] = "nope"
        with pytest.raises(KeyError, match="registered"):
            create_backend(spec, prompts=prompt_lib)


class TestReplay:
    def record_run(self, tmp_path, backend_spec, prompt_lib):
        logger = RunLogger(tmp_path / "run")
        b = create_backend(backend_spec("mock"), prompts=prompt_lib, sink=logger.call_sink)
        b.context.run_id = "r1"
        g = b.parse_goal("Төсекті тап", "kk")
        v = b.verify_candidate(IMG, VerifyQuery("төсек", "kk"))
        logger.close()
        return g, v, tmp_path / "run" / "vlm_calls.jsonl"

    def test_round_trip(self, tmp_path, backend_spec, prompt_lib):
        g, v, calls = self.record_run(tmp_path, backend_spec, prompt_lib)
        lines = calls.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and all(json.loads(l)["type"] == "vlm_call" for l in lines)
        assert json.loads(lines[1])["image_files"][0].startswith("images/")
        spec = backend_spec("replay")
        spec["replay"]["sources"] = [str(calls)]
        rb = create_backend(spec, prompts=prompt_lib)
        assert rb.parse_goal("Төсекті тап", "kk") == g
        rv = rb.verify_candidate(IMG, VerifyQuery("төсек", "kk"))
        assert rv.p_target == v.p_target and rv.source == "replay"
        with pytest.raises(InvalidOutputError):
            rb.verify_candidate(IMG2, VerifyQuery("төсек", "kk"))

    def test_changed_prompt_misses(self, tmp_path, backend_spec, prompt_lib, repo_root):
        _, _, calls = self.record_run(tmp_path, backend_spec, prompt_lib)
        changed = tmp_path / "prompts"
        shutil.copytree(repo_root / "prompts", changed)
        spec = backend_spec("replay")
        f = changed / f"{spec['prompts']['parse_goal']}.yaml"           # the prompt version the config uses
        f.write_text(f.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
        spec["replay"]["sources"] = [str(calls)]
        rb = create_backend(spec, prompts=PromptLibrary(changed))
        with pytest.raises(InvalidOutputError):
            rb.parse_goal("Төсекті тап", "kk")

    def test_needs_sources(self, backend_spec, prompt_lib):
        with pytest.raises(BackendConfigError):
            create_backend(backend_spec("replay"), prompts=prompt_lib)
