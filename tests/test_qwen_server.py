"""servers/qwen_vl_server.py without a GPU: message conversion, and the HTTP contract the bench's backend reads
(D-040). Needs fastapi (the Qolda venv); skipped in habitat_env, which has no server packages."""
import base64
import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servers"))
import qwen_vl_server as srv  # noqa: E402

from kznav.language.http import first_message_text, first_token_top_logprobs  # noqa: E402


def png_uri(w=40, h=30):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


VERIFY = [{"role": "system", "content": "Тек «иә» немесе «жоқ»."},
          {"role": "user", "content": [{"type": "text", "text": "Суреттегі негізгі зат «банан» болып табыла ма?"},
                                       {"type": "image_url", "image_url": {"url": png_uri()}}]}]


def test_messages_put_images_before_text_and_keep_plain_strings():
    msgs, images = srv.to_qwen_messages(VERIFY)
    assert msgs[0] == VERIFY[0]
    assert msgs[1]["content"][0] == {"type": "image"} and msgs[1]["content"][1]["type"] == "text"
    assert len(images) == 1 and images[0].size == (40, 30)
    text_only, none = srv.to_qwen_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert text_only == [{"role": "user", "content": "hi"}] and none == []


class FakeEngine:
    def __init__(self):
        self.kwargs = None

    def chat(self, messages, **kw):
        self.kwargs = kw
        return {"text": "иә", "latency_s": 0.1, "prompt_tokens": 50, "completion_tokens": 1,
                "top_logprobs": [{"token": "и", "logprob": -0.1}, {"token": "ж", "logprob": -2.4}],
                "image_grid_thw": [[1, 28, 28]], "image_tokens": [196]}


@pytest.fixture
def client_engine():
    eng = FakeEngine()
    return TestClient(srv.create_app(eng, "qwen3-vl-4b", 200704, 802816)), eng


def test_response_is_what_the_backend_reads(client_engine):
    client, eng = client_engine
    assert client.get("/v1/models").json()["data"][0]["id"] == "qwen3-vl-4b"
    r = client.post("/v1/chat/completions", json={"model": "qwen3-vl-4b", "messages": VERIFY, "max_tokens": 4,
                                                  "logprobs": True, "top_logprobs": 10, "max_pixels": 401408}).json()
    assert first_message_text(r) == "иә"
    assert first_token_top_logprobs(r) == {"и": -0.1, "ж": -2.4}
    assert eng.kwargs["min_pixels"] == 200704 and eng.kwargs["max_pixels"] == 401408    # per-request budget
    assert r["kznav"]["image_tokens"] == [196] and r["kznav"]["enable_thinking"] is False
    assert r["usage"]["total_tokens"] == 51


def test_no_logprobs_unless_asked(client_engine):
    client, _ = client_engine
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": VERIFY}).json()
    assert r["choices"][0]["logprobs"] is None


def test_thinking_is_refused(client_engine):
    client, eng = client_engine
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": VERIFY, "enable_thinking": True})
    assert r.status_code == 400 and eng.kwargs is None
