"""The VLM backend contract.

Navigation code talks to *any* model only through ``VLMBackend``:

    parse_goal(text, lang)            -> GoalHypotheses        (once per episode)
    verify_candidate(image, query)    -> VerificationResult    (per detection event)
    describe(image, prompt)           -> str                   (diagnostics / captions)
    capabilities                      -> BackendCapabilities

The public methods are template methods: they compute the input key, time the
call, collect every attempt (HTTP and JSON retries) and always emit a
``CallRecord`` to the sink — including failures. Subclasses implement only
the ``_parse_goal`` / ``_verify`` / ``_describe`` hooks.

Timing protocol: ``latency_total_s`` is ``time.perf_counter()`` from method
entry to return, i.e. what the robot waits for, including retries and image
encoding. Per-attempt latencies are recorded as well. See
docs/measurement_protocol.md.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from .images import encode_image
from .prompts import PromptLibrary, PromptTemplate
from .schemas import GoalHypotheses, VerificationResult, VerifyQuery


class BackendError(RuntimeError):
    """A model-side failure (transport, timeout, invalid output after retries)."""


class TransportError(BackendError):
    pass


class InvalidOutputError(BackendError):
    pass


class BackendConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    supports_logprobs: bool
    supports_images: bool
    max_context_tokens: Optional[int]
    #: Monetary cost per call for hosted APIs; None for local models, whose
    #: cost is reported as measured latency instead.
    cost_per_call_usd: Optional[float] = None

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "BackendCapabilities":
        return cls(
            supports_logprobs=bool(spec["supports_logprobs"]),
            supports_images=bool(spec["supports_images"]),
            max_context_tokens=spec.get("max_context_tokens"),
            cost_per_call_usd=spec.get("cost_per_call_usd"),
        )


@dataclass
class CallContext:
    run_id: str = ""
    episode_id: str = ""
    step: int = -1
    warmup: bool = False


@dataclass
class Attempt:
    kind: str                       # "http" | "json"
    started_utc: str
    latency_s: float
    ok: bool
    error: Optional[str] = None
    http_status: Optional[int] = None
    response: Optional[dict] = None  # raw server JSON, verbatim
    text: Optional[str] = None


@dataclass
class CallRecord:
    method: str
    backend: str
    model_id: str
    prompt_ref: str
    prompt_sha256: str
    input_key: str
    inputs: Dict[str, Any]
    context: CallContext
    started_utc: str
    latency_total_s: float = float("nan")
    attempts: List[Attempt] = field(default_factory=list)
    http_retries: int = 0
    json_retries: int = 0
    ok: bool = False
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    #: The input images as lossless PNG (the original arrays, whatever the backend sent);
    #: stored by the sink, never inlined in JSONL.
    images: List[bytes] = field(default_factory=list, repr=False)

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("images")
        d["type"] = "vlm_call"
        return d


CallSink = Callable[[CallRecord], None]


class ListSink:
    """In-memory sink (tests, notebooks)."""

    def __init__(self) -> None:
        self.records: List[CallRecord] = []

    def __call__(self, rec: CallRecord) -> None:
        self.records.append(rec)


class TeeSink:
    """Sends every record to several sinks (e.g. the run's JSONL log and a ListSink for per-episode costs)."""

    def __init__(self, *sinks: CallSink) -> None:
        self.sinks = sinks

    def __call__(self, rec: CallRecord) -> None:
        for s in self.sinks:
            s(rec)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def image_digest(image: np.ndarray) -> str:
    """Content hash of a raw image array (independent of the image encoder)."""
    arr = np.ascontiguousarray(image)
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{arr.shape}".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def input_key(method: str, prompt_sha256: str, inputs: Mapping[str, Any]) -> str:
    payload = json.dumps({"method": method, "prompt": prompt_sha256, "inputs": inputs},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VLMBackend(ABC):
    """Base class for every language/vision model used by the pipeline."""

    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink: Optional[CallSink] = None):
        self.spec: Dict[str, Any] = dict(spec)
        for key in ("name", "model_id", "prompts"):
            if key not in self.spec:
                raise BackendConfigError(f"backend config is missing {key!r}")
        self.name: str = str(self.spec["name"])
        self.model_id: str = str(self.spec["model_id"])
        self.sink: CallSink = sink if sink is not None else ListSink()
        self.context = CallContext()
        p = self.spec["prompts"]
        self.parse_prompt: PromptTemplate = prompts.load(p["parse_goal"])
        self.verify_prompt: PromptTemplate = prompts.load(p["verify_candidate"])
        self.describe_prompt: PromptTemplate = prompts.load(p["describe"])

    # ------------------------------------------------------------------ contract
    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    def parse_goal(self, text: str, lang: str) -> GoalHypotheses:
        rec = self._begin("parse_goal", self.parse_prompt, {"text": text, "lang": lang})
        return self._run(rec, lambda: self._parse_goal(text, lang, rec), lambda out: out.model_dump())

    def verify_candidate(self, image: np.ndarray, query: VerifyQuery) -> VerificationResult:
        inputs = {"image": {"sha256": image_digest(image), "shape": list(image.shape)},
                  "target": query.target, "lang": query.lang}
        rec = self._begin("verify_candidate", self.verify_prompt, inputs)
        rec.images.append(encode_image(image, "png"))
        return self._run(rec, lambda: self._verify(image, query, rec), lambda out: out.as_dict())

    def describe(self, image: np.ndarray, prompt: str) -> str:
        inputs = {"image": {"sha256": image_digest(image), "shape": list(image.shape)}, "prompt": prompt}
        rec = self._begin("describe", self.describe_prompt, inputs)
        rec.images.append(encode_image(image, "png"))
        return self._run(rec, lambda: self._describe(image, prompt, rec), lambda out: {"text": out})

    def warmup(self, n: int) -> None:
        """``n`` describe calls on a fixed synthetic image, logged with warmup=True and excluded from stats."""
        img = np.full((64, 64, 3), 127, dtype=np.uint8)
        prev = self.context.warmup
        self.context.warmup = True
        try:
            for _ in range(n):
                self.describe(img, "warmup")
        finally:
            self.context.warmup = prev

    def describe_config(self) -> Dict[str, Any]:
        """What goes into the run manifest for this backend."""
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_id": self.model_id,
            "spec": self.spec,
            "capabilities": asdict(self.capabilities),
            "prompts": {
                t.task: {"ref": t.ref, "sha256": t.sha256, "review_status": t.review_status}
                for t in (self.parse_prompt, self.verify_prompt, self.describe_prompt)
            },
        }

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _parse_goal(self, text: str, lang: str, rec: CallRecord) -> GoalHypotheses: ...

    @abstractmethod
    def _verify(self, image: np.ndarray, query: VerifyQuery, rec: CallRecord) -> VerificationResult: ...

    @abstractmethod
    def _describe(self, image: np.ndarray, prompt: str, rec: CallRecord) -> str: ...

    # ------------------------------------------------------------------ plumbing
    def _begin(self, method: str, tpl: PromptTemplate, inputs: Dict[str, Any]) -> CallRecord:
        ctx = CallContext(**asdict(self.context))
        return CallRecord(
            method=method, backend=self.name, model_id=self.model_id,
            prompt_ref=tpl.ref, prompt_sha256=tpl.sha256,
            input_key=input_key(method, tpl.sha256, inputs), inputs=inputs,
            context=ctx, started_utc=utc_now(),
        )

    def _run(self, rec: CallRecord, fn: Callable[[], Any], dump: Callable[[Any], Dict[str, Any]]) -> Any:
        t0 = time.perf_counter()
        try:
            out = fn()
            rec.ok = True
            rec.output = dump(out)
            return out
        except BackendError as e:
            rec.error = f"{type(e).__name__}: {e}"
            raise
        except Exception as e:  # programming error: record it, but do not disguise it as a model failure
            rec.error = f"internal {type(e).__name__}: {e}"
            raise
        finally:
            # Retry counters are incremented by the hooks / HTTP client where the retry happens.
            rec.latency_total_s = time.perf_counter() - t0
            self.sink(rec)


def load_prompt_library(root: Optional[Path] = None) -> PromptLibrary:
    return PromptLibrary(root or Path(__file__).resolve().parents[2] / "prompts")
