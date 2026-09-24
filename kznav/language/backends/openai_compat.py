"""Backend for any model served behind an OpenAI-compatible chat-completions API.

Covers lmdeploy, vLLM and the project's own FastAPI fallback server. Model
specifics (request extras, prompt set, limits) come from the config block;
subclasses only add model-specific *validation* of that config.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..base import BackendCapabilities, BackendConfigError, CallRecord, VLMBackend
from ..goal_parsing import ParseLimits, run_parse_loop
from ..http import ChatClient, first_message_text, first_token_top_logprobs
from ..images import data_uri, encode_image
from ..logprobs import label_from_text, option_probabilities
from ..prompts import PromptLibrary, build_describe_messages, build_verify_messages
from ..registry import BACKENDS
from ..schemas import GoalHypotheses, VerificationResult, VerifyQuery


@BACKENDS.register("openai_compat")
class OpenAICompatBackend(VLMBackend):
    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink=None, transport=None):
        super().__init__(spec, prompts, sink)
        s = self.spec
        for key in ("base_url", "max_tokens", "capabilities"):
            if key not in s:
                raise BackendConfigError(f"{self.name}: config is missing {key!r}")
        self._caps = BackendCapabilities.from_spec(s["capabilities"])
        self.extra_body: Dict[str, Any] = dict(s.get("extra_body") or {})
        self.temperature = float(s.get("temperature", 0.0))
        self.top_logprobs = int(s.get("top_logprobs", 10))
        self.seed = s.get("seed")
        img = s.get("image") or {}
        self.image_format = str(img.get("format", "png"))
        self.image_max_side = img.get("max_side")
        p = s.get("parse") or {}
        self.parse_limits = ParseLimits(
            max_attempts=int(p.get("max_attempts", 3)),
            min_hypotheses=int(p.get("min_hypotheses", 1)),
            max_hypotheses=int(p.get("max_hypotheses", 8)),
        )
        http = s.get("http") or {}
        self.client = ChatClient(
            s["base_url"],
            api_key=str(s.get("api_key", "EMPTY")),
            timeout_s=float(http.get("timeout_s", 60.0)),
            max_attempts=int(http.get("max_attempts", 3)),
            backoff_s=float(http.get("backoff_s", 1.0)),
            transport=transport,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    # ------------------------------------------------------------------ request building
    def _body(self, messages: List[dict], max_tokens: int, logprobs: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
        }
        if self.seed is not None:
            body["seed"] = int(self.seed)
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        # Same semantics as openai-python's extra_body: merged at the top level.
        body.update(self.extra_body)
        return body

    def _with_image(self, messages: List[dict], image: np.ndarray, rec: CallRecord) -> List[dict]:
        if not self._caps.supports_images:
            raise BackendConfigError(f"{self.name} does not accept images")
        data = encode_image(image, self.image_format, self.image_max_side)
        # The original crop is logged by the base class; record how it was sent.
        rec.inputs["sent_image"] = {"format": self.image_format, "max_side": self.image_max_side,
                                    "bytes_sha256": hashlib.sha256(data).hexdigest()}
        last = messages[-1]
        assert last["role"] == "user"
        content = [{"type": "text", "text": last["content"]},
                   {"type": "image_url", "image_url": {"url": data_uri(data, self.image_format)}}]
        return messages[:-1] + [{"role": "user", "content": content}]

    def _max_tokens(self, task: str) -> int:
        return int(self.spec["max_tokens"][task])

    # ------------------------------------------------------------------ hooks
    def _parse_goal(self, text: str, lang: str, rec: CallRecord) -> GoalHypotheses:
        def generate(messages: List[dict]) -> str:
            data = self.client.chat(self._body(messages, self._max_tokens("parse_goal"), logprobs=False), rec)
            return first_message_text(data)

        return run_parse_loop(text, self.parse_prompt, self.parse_limits, generate, rec)

    def _verify(self, image: np.ndarray, query: VerifyQuery, rec: CallRecord) -> VerificationResult:
        messages, options, verify_lang = build_verify_messages(self.verify_prompt, query.target, query.lang)
        rec.inputs["verify_lang"] = verify_lang
        messages = self._with_image(messages, image, rec)
        want_lp = self._caps.supports_logprobs
        data = self.client.chat(self._body(messages, self._max_tokens("verify_candidate"), logprobs=want_lp), rec)
        text = first_message_text(data)
        lab = label_from_text(text, options)
        text_label = None if lab is None else lab == "yes"
        top = first_token_top_logprobs(data) if want_lp else None
        if top is None:
            return VerificationResult(None, "unavailable", None, text, text_label)
        op = option_probabilities(top, options)
        if not op.probs:
            return VerificationResult(None, "unavailable", 0.0, text, text_label)
        return VerificationResult(op.probs["yes"], "logprobs", op.covered, text, text_label)

    def _describe(self, image: np.ndarray, prompt: str, rec: CallRecord) -> str:
        messages = self._with_image(build_describe_messages(self.describe_prompt, prompt), image, rec)
        data = self.client.chat(self._body(messages, self._max_tokens("describe"), logprobs=False), rec)
        return first_message_text(data)
