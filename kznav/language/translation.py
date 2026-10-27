"""Machine translation for the translate-first baselines (D-039).

A ``Translator`` turns text in one language into another (in practice kk -> en). It is not a
``VLMBackend``: it neither parses goals nor sees images. Every call is logged like a model call
(``CallRecord`` with ``method="translate"``, raw input and output, latency) to the same sink as the
VLM calls, so ``vlm_calls.jsonl`` holds the whole language side of a run and the ``replay`` translator
can recompute it without the model.

    translate(text, src_lang) -> (translation, CallRecord)

``src_lang`` is a pipeline language code (kk | kk_ru | ru | en); ``lang_codes`` in the config maps it
to the model's own code (NLLB: kaz_Cyrl, rus_Cyrl, eng_Latn).
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ..registry import Registry
from .base import BackendConfigError, BackendError, CallContext, CallRecord, CallSink, ListSink, input_key, utc_now
from .lexicon import Lexicon

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TranslationError(BackendError):
    """The translator produced no usable output (or a replayed call is missing)."""


class Translator(ABC):
    """Machine translation in front of the parser (translate-first baselines, D-039).

    Every translation is logged like a model call; the settings hash plays the role of the prompt hash in the
    replay key.

    Args:
        spec: Translator config (``name``, ``model_id``, ``lang_codes``, ``target_lang``, generation settings).
        sink: Receives one ``CallRecord`` per translation.

    Raises:
        BackendConfigError: If a required key or the target language code is missing.
    """
    def __init__(self, spec: Mapping[str, Any], sink: Optional[CallSink] = None):
        self.spec: Dict[str, Any] = dict(spec)
        for key in ("name", "model_id", "lang_codes", "target_lang"):
            if key not in self.spec:
                raise BackendConfigError(f"translator config is missing {key!r}")
        self.name = str(self.spec["name"])
        self.model_id = str(self.spec["model_id"])
        self.lang_codes: Dict[str, str] = {str(k): str(v) for k, v in self.spec["lang_codes"].items()}
        self.target_lang = str(self.spec["target_lang"])
        gen = dict(self.spec.get("generation") or {})
        self.num_beams = int(gen.get("num_beams", 1))
        self.max_new_tokens = int(gen.get("max_new_tokens", 64))
        # Plays the role of the prompt hash: a changed model or decoding setting changes every input key.
        self.settings_sha256 = hashlib.sha256(json.dumps(
            {"model_id": self.model_id, "lang_codes": self.lang_codes, "generation": gen},
            sort_keys=True).encode("utf-8")).hexdigest()
        self.code(self.target_lang)
        self.sink: CallSink = sink if sink is not None else ListSink()
        self.context = CallContext()
        self.last_record: Optional[CallRecord] = None

    def code(self, lang: str) -> str:
        """The model's language code for a command language.

        Raises:
            BackendConfigError: If the language has no configured code.
        """
        try:
            return self.lang_codes[lang]
        except KeyError:
            raise BackendConfigError(f"{self.name}: no model language code for {lang!r}; "
                                     f"configured: {sorted(self.lang_codes)}") from None

    def translate(self, text: str, src_lang: str) -> Tuple[str, CallRecord]:
        """Translate ``text`` from ``src_lang`` into the target language and record the call.

        Returns:
            The translation (whitespace normalised) and its call record.

        Raises:
            TranslationError: If the output is empty or a replayed translation is missing.
        """
        src, tgt = self.code(src_lang), self.code(self.target_lang)
        inputs = {"text": text, "src_lang": src_lang, "tgt_lang": self.target_lang, "src_code": src, "tgt_code": tgt}
        rec = CallRecord(method="translate", backend=self.name, model_id=self.model_id,
                         prompt_ref=f"{self.name}:{src}->{tgt}", prompt_sha256=self.settings_sha256,
                         input_key=input_key("translate", self.settings_sha256, inputs), inputs=inputs,
                         context=CallContext(**asdict(self.context)), started_utc=utc_now())
        self.last_record = rec
        t0 = time.perf_counter()
        try:
            out = " ".join(str(self._translate(text, src, tgt, rec)).split())
            if not out:
                raise TranslationError(f"{self.name}: empty translation of {text!r}")
            rec.ok = True
            rec.output = {"text": out}
            return out, rec
        except BackendError as e:
            rec.error = f"{type(e).__name__}: {e}"
            raise
        except Exception as e:  # programming error: record it, but do not disguise it as a model failure
            rec.error = f"internal {type(e).__name__}: {e}"
            raise
        finally:
            rec.latency_total_s = time.perf_counter() - t0
            self.sink(rec)

    def describe_config(self) -> Dict[str, Any]:
        """What goes into the run manifest for this translator."""
        return {"name": self.name, "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "model_id": self.model_id, "settings_sha256": self.settings_sha256, "spec": self.spec}

    @abstractmethod
    def _translate(self, text: str, src_code: str, tgt_code: str, rec: CallRecord) -> str: ...


TRANSLATORS: Registry[Translator] = Registry("translator", builtin_modules=["kznav.language.translation"])


def create_translator(spec: Optional[Mapping[str, Any]], sink: Optional[CallSink] = None) -> Optional[Translator]:
    """None when the config has no translator (``translator: null`` or ``type: none``)."""
    if not spec or str(spec.get("type", "none")) == "none":
        return None
    return TRANSLATORS.resolve(spec)(spec, sink)  # type: ignore[call-arg]


@TRANSLATORS.register("nllb")
class NLLBTranslator(Translator):
    """NLLB-200 through HF transformers (seq2seq; target language forced as the first generated token)."""

    def __init__(self, spec: Mapping[str, Any], sink: Optional[CallSink] = None):
        super().__init__(spec, sink)
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self._torch = torch
        self.device = str(self.spec.get("device", "cpu"))
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype=dtypes[str(self.spec.get("dtype", "fp32"))]).to(self.device).eval()

    def _translate(self, text: str, src_code: str, tgt_code: str, rec: CallRecord) -> str:
        self._tok.src_lang = src_code
        enc = self._tok(text, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            out = self._model.generate(**enc, forced_bos_token_id=self._tok.convert_tokens_to_ids(tgt_code),
                                       num_beams=self.num_beams, do_sample=False, max_new_tokens=self.max_new_tokens)
        return self._tok.batch_decode(out, skip_special_tokens=True)[0]


@TRANSLATORS.register("mock")
class MockTranslator(Translator):
    """Tests only: the first English name of the first lexicon category found in the text, else the text."""

    def __init__(self, spec: Mapping[str, Any], sink: Optional[CallSink] = None):
        super().__init__(spec, sink)
        lex = (self.spec.get("mock") or {}).get("lexicon")
        self.lexicon: Optional[Lexicon] = Lexicon.load(_REPO_ROOT / lex) if lex else None

    def _translate(self, text: str, src_code: str, tgt_code: str, rec: CallRecord) -> str:
        hit = self.lexicon.find_in_text(text) if self.lexicon else None
        if hit is None:
            return text
        en = self.lexicon[hit["category"]].en()
        return en[0] if en else text


@TRANSLATORS.register("replay")
class ReplayTranslator(Translator):
    """Serves translations recorded in earlier runs' vlm_calls.jsonl, by input key (no live fallback)."""

    def __init__(self, spec: Mapping[str, Any], sink: Optional[CallSink] = None):
        super().__init__(spec, sink)
        sources = (self.spec.get("replay") or {}).get("sources") or []
        if not sources:
            raise BackendConfigError(f"{self.name}: replay.sources must list at least one vlm_calls.jsonl")
        self._index: Dict[str, Dict[str, Any]] = {}
        for src in sources:
            for line in Path(src).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("method") == "translate" and r.get("ok"):
                        self._index.setdefault(r["input_key"], r)

    def _translate(self, text: str, src_code: str, tgt_code: str, rec: CallRecord) -> str:
        hit = self._index.get(rec.input_key)
        if hit is None:
            raise TranslationError(f"no recorded translation for input key {rec.input_key[:12]}…")
        rec.inputs["replayed_from"] = {"run_id": hit["context"].get("run_id"), "started_utc": hit["started_utc"]}
        return hit["output"]["text"]
