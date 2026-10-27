"""Parse from recorded calls, verify with the live model (D-078: spoken commands with the text arms' verifier).

A run's backend both parses the command (once per episode) and verifies candidates (during the episode). With
``parse_replay`` in the backend config, the parse is looked up in recorded calls of another source, and everything
else - verification, describe, warmup, capabilities - stays with the configured live backend::

    backend=qolda ++backend.parse_replay.prompt=speech/parse_goal/v1 \
        "++backend.parse_replay.sources=[runs/speech/tts_mms_skip/vlm_calls.jsonl]"

Sources: Qolda-AVL hearing the recording (tools/parse_speech_commands.py; prompt speech/parse_goal/v1) or the
cascade - Qolda-AVL's transcript parsed by the text model (tools/parse_transcripts.py; prompt default/parse_goal/v4).
The lookup key is the replay key of the episode's command text under ``parse_replay.prompt``; cascade records are
re-keyed to the command text (``rekey_parse_record``), the text the model actually read stays in their inputs. A
missing record is an invalid parse ("did not understand"), never a call to the live model.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from ..base import input_key
from .replay import ReplayBackend


class ParseReplayBackend:
    """Delegates everything to ``inner`` except ``parse_goal``, which replays ``parse_replay.sources``."""

    def __init__(self, inner: Any, prompts: Any):
        self.inner = inner
        pr = dict(inner.spec.get("parse_replay") or {})
        if not pr.get("sources") or not pr.get("prompt"):
            raise ValueError(f"{inner.name}: parse_replay needs 'sources' and 'prompt'")
        p = inner.spec["prompts"]
        spec = {"name": "replay", "type": "replay", "model_id": "replay", "replay": {"sources": list(pr["sources"])},
                "capabilities": dict(inner.spec["capabilities"]),
                "prompts": {"parse_goal": str(pr["prompt"]), "verify_candidate": p["verify_candidate"],
                            "describe": p["describe"]}}
        self.parse_source = ReplayBackend(spec, prompts, inner.sink)
        self.parse_source.context = inner.context          # one context: the run sets the episode id once

    @property
    def parse_prompt(self):
        """The parse prompt of the replayed source (its hash is part of the lookup key)."""
        return self.parse_source.parse_prompt

    def parse_goal(self, text: str, lang: str):
        """Return the recorded parse of ``text`` in ``lang``; never calls the live model.

        Raises:
            InvalidOutputError: If no recorded parse exists for this command (counted as "did not understand").
        """
        return self.parse_source.parse_goal(text, lang)

    def describe_config(self) -> Dict[str, Any]:
        """The live backend's manifest entry plus the replayed parse source (prompt, sources, source models)."""
        d = self.inner.describe_config()
        d["parse_replay"] = {"prompt": self.parse_prompt.ref, "prompt_sha256": self.parse_prompt.sha256,
                             "sources": list(self.inner.spec["parse_replay"]["sources"]),
                             "source_model_ids": self.parse_source.source_model_ids}
        return d

    def __getattr__(self, name: str) -> Any:            # verify_candidate, describe, warmup, context, spec, ...
        return getattr(self.inner, name)


def rekey_parse_record(rec: Mapping[str, Any], text: str, lang: str, **extra: Any) -> Dict[str, Any]:
    """A recorded parse of some other text (a transcript), keyed as the parse of the command ``text``.

    The prompt stays the record's own; the text the model read and its key move to ``inputs.parsed_text`` /
    ``inputs.parsed_text_input_key``."""
    d = dict(rec)
    inputs = dict(d.get("inputs") or {})
    d["inputs"] = {"text": text, "lang": lang, "parsed_text": inputs.get("text"),
                   "parsed_text_input_key": d.get("input_key"), **extra}
    d["input_key"] = input_key("parse_goal", d["prompt_sha256"], {"text": text, "lang": lang})
    return d
