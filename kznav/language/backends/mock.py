"""Deterministic stand-in backend for tests and GPU-free smoke runs.

It produces *text* (JSON for parsing, first-token logprobs for verification)
and pushes it through the same validation and option-probability code as the
real backends, so the smoke test exercises the real plumbing. Outputs are a
pure function of (seed, input key): identical inputs give identical outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..base import BackendCapabilities, CallRecord, VLMBackend
from ..goal_parsing import ParseLimits, run_parse_loop
from ..lexicon import Lexicon
from ..logprobs import label_from_text, normalize_token, option_probabilities
from ..prompts import PromptLibrary, build_verify_messages
from ..registry import BACKENDS
from ..schemas import GoalHypotheses, VerificationResult, VerifyQuery

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _unit(seed: int, *parts: str) -> float:
    h = hashlib.sha256("|".join([str(seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@BACKENDS.register("mock")
class MockBackend(VLMBackend):
    """Deterministic, GPU-free backend for tests and smoke runs.

    Parses commands by looking up the category in the lexicon and answers verification questions with
    pseudo-random but reproducible probabilities (a hash of the seed and the input key), pushed through the same
    validation code as the real backends. Never used for reported numbers.
    """
    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink=None):
        super().__init__(spec, prompts, sink)
        m = dict(self.spec.get("mock") or {})
        self.seed = int(m.get("seed", 0))
        self.malformed_rate = float(m.get("malformed_first_attempt_rate", 0.0))
        self.verify_mode = str(m.get("verify_mode", "hash"))
        self.verify_p = float(m.get("verify_p", 0.5))
        lex = m.get("lexicon")
        self.lexicon: Optional[Lexicon] = Lexicon.load(_REPO_ROOT / lex) if lex else None
        p = self.spec.get("parse") or {}
        self.parse_limits = ParseLimits(max_attempts=int(p.get("max_attempts", 3)),
                                        min_hypotheses=int(p.get("min_hypotheses", 1)),
                                        max_hypotheses=int(p.get("max_hypotheses", 8)))
        self._caps = BackendCapabilities.from_spec(self.spec.get("capabilities") or {
            "supports_logprobs": True, "supports_images": True, "max_context_tokens": None})

    @property
    def capabilities(self) -> BackendCapabilities:
        """Static capabilities from the config (logprobs and images supported by default)."""
        return self._caps

    def _goal_json(self, text: str) -> str:
        hit = self.lexicon.find_in_text(text) if self.lexicon else None
        if hit is None:
            payload: Dict[str, Any] = {"target": text.split()[-1] if text.split() else "?",
                                       "hypotheses": [{"en": "object", "w": 1.0}]}
        else:
            en = self.lexicon[hit["category"]].en()[:4] or ["object"]
            weights = [0.5 ** i for i in range(len(en))]
            payload = {"target": hit["form"], "target_en": en[0],
                       "hypotheses": [{"en": e, "w": round(w / sum(weights), 4)} for e, w in zip(en, weights)]}
        payload.update(attributes={"material": [], "color": [], "size": []}, rooms=[], landmarks=[])
        return json.dumps(payload, ensure_ascii=False)

    def _parse_goal(self, text: str, lang: str, rec: CallRecord) -> GoalHypotheses:
        calls = {"n": 0}

        def generate(messages: List[dict]) -> str:
            calls["n"] += 1
            out = self._goal_json(text)
            if calls["n"] == 1 and _unit(self.seed, rec.input_key, "malformed") < self.malformed_rate:
                return out[:-1]  # truncated JSON, exercises the retry path
            return out

        return run_parse_loop(text, self.parse_prompt, self.parse_limits, generate, rec)

    def _verify(self, image: np.ndarray, query: VerifyQuery, rec: CallRecord) -> VerificationResult:
        _, options, verify_lang = build_verify_messages(self.verify_prompt, query.target, query.lang)
        rec.inputs["verify_lang"] = verify_lang
        p = self.verify_p if self.verify_mode == "constant" else 0.02 + 0.96 * _unit(self.seed, rec.input_key)
        yes_tok = normalize_token(options["yes"][0])[0]
        no_tok = normalize_token(options["no"][0])[0]
        top = {yes_tok: math.log(p), no_tok: math.log(1.0 - p)}
        text = options["yes"][0] if p >= 0.5 else options["no"][0]
        op = option_probabilities(top, options)
        lab = label_from_text(text, options)
        return VerificationResult(op.probs["yes"], "mock", op.covered, text, None if lab is None else lab == "yes")

    def _describe(self, image: np.ndarray, prompt: str, rec: CallRecord) -> str:
        return f"mock description of a {image.shape[1]}x{image.shape[0]} image"
