"""Replay recorded model calls from a previous run's ``vlm_calls.jsonl``.

Lets navigation, metrics and reports be recomputed without the model (the
pilot lesson: a broken downstream step must never force re-querying the
model). Lookup is by ``input_key`` = hash(method, prompt sha256, inputs), so a
changed prompt or a different image cannot silently reuse an old answer; a
missing record is always an error (no fallback to a live model).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from ..base import BackendCapabilities, BackendConfigError, CallRecord, InvalidOutputError, VLMBackend
from ..prompts import PromptLibrary
from ..registry import BACKENDS
from ..schemas import GoalHypotheses, VerificationResult, VerifyQuery


@BACKENDS.register("replay")
class ReplayBackend(VLMBackend):
    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink=None):
        super().__init__(spec, prompts, sink)
        r = self.spec.get("replay") or {}
        sources = r.get("sources") or []
        if not sources:
            raise BackendConfigError(f"{self.name}: replay.sources must list at least one vlm_calls.jsonl")
        self._index: Dict[str, Dict[str, Any]] = {}
        self._source_model_ids = set()
        for src in sources:
            for line in Path(src).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("type") != "vlm_call" or not rec.get("ok"):
                    continue
                self._index.setdefault(rec["input_key"], rec)
                self._source_model_ids.add(rec.get("model_id"))
        self._caps = BackendCapabilities.from_spec(self.spec["capabilities"])

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def _lookup(self, rec: CallRecord) -> Dict[str, Any]:
        hit = self._index.get(rec.input_key)
        if hit is None:
            raise InvalidOutputError(f"no recorded {rec.method} call for input key {rec.input_key[:12]}…")
        rec.inputs["replayed_from"] = {"run_id": hit["context"].get("run_id"), "started_utc": hit["started_utc"],
                                       "model_id": hit["model_id"]}
        return hit["output"]

    def _parse_goal(self, text: str, lang: str, rec: CallRecord) -> GoalHypotheses:
        return GoalHypotheses.model_validate(self._lookup(rec))

    def _verify(self, image: np.ndarray, query: VerifyQuery, rec: CallRecord) -> VerificationResult:
        out = dict(self._lookup(rec))
        out["source"] = "replay"
        return VerificationResult(**out)

    def _describe(self, image: np.ndarray, prompt: str, rec: CallRecord) -> str:
        return str(self._lookup(rec)["text"])

    @property
    def source_model_ids(self) -> List[str]:
        return sorted(str(x) for x in self._source_model_ids)
