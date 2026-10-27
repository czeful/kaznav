"""servers/gemma_server.py without a GPU: message conversion, and the HTTP contract the bench's backend reads
(D-083). Needs fastapi (the Qolda venv); skipped in habitat_env, which has no server packages."""
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servers"))
import gemma_server as srv  # noqa: E402
from test_qwen_server import VERIFY  # noqa: E402

from kznav.language.http import first_message_text, first_token_top_logprobs  # noqa: E402


def test_messages_are_typed_parts_with_images_first():
    msgs, images = srv.to_gemma_messages(VERIFY)
    assert msgs[0] == {"role": "system", "content": [{"type": "text", "text": VERIFY[0]["content"]}]}
    assert msgs[1]["content"][0] == {"type": "image"} and msgs[1]["content"][1]["type"] == "text"
    assert len(images) == 1 and images[0].size == (40, 30)
    text_only, none = srv.to_gemma_messages([{"role": "user", "content": "hi"}])
    assert text_only == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}] and none == []


class FakeEngine:
    def __init__(self):
        self.kwargs = None

    def chat(self, messages, **kw):
        self.kwargs = kw
        return {"text": "yes", "latency_s": 0.1, "prompt_tokens": 300, "completion_tokens": 1,
                "top_logprobs": [{"token": "yes", "logprob": -0.2}, {"token": "no", "logprob": -1.8}],
                "image_tokens": [256]}


def test_response_is_what_the_backend_reads():
    eng = FakeEngine()
    client = TestClient(srv.create_app(eng, "gemma-3-4b"))
    assert client.get("/v1/models").json()["data"][0]["id"] == "gemma-3-4b"
    r = client.post("/v1/chat/completions", json={"model": "gemma-3-4b", "messages": VERIFY, "max_tokens": 4,
                                                  "logprobs": True, "top_logprobs": 10}).json()
    assert first_message_text(r) == "yes" and first_token_top_logprobs(r) == {"yes": -0.2, "no": -1.8}
    assert r["kznav"]["image_tokens"] == [256] and eng.kwargs["top_logprobs"] == 10
    bad = client.post("/v1/chat/completions", json={"model": "m", "messages": VERIFY, "enable_thinking": True})
    assert bad.status_code == 400


def test_backend_config_matches_the_server(repo_root):
    spec = yaml.safe_load((repo_root / "configs/backend/gemma3_4b.yaml").read_text(encoding="utf-8"))
    assert spec["base_url"].endswith(":23335/v1") and spec["model_id"] == "gemma-3-4b"
    assert spec["weights_precision"] == "bf16" and spec["capabilities"]["supports_logprobs"]
