"""Backend registry and factory.

Adding a model = one config file in ``configs/backend/`` plus either a
registered subclass here (``@BACKENDS.register("name")``) or an external
class referenced by ``class_path``. Navigation code never changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..registry import Registry
from .base import CallSink, VLMBackend, load_prompt_library
from .prompts import PromptLibrary

BACKENDS: Registry[VLMBackend] = Registry(
    "VLM backend",
    builtin_modules=[
        "kznav.language.backends.openai_compat",
        "kznav.language.backends.qolda",
        "kznav.language.backends.qwen_vl",
        "kznav.language.backends.mock",
        "kznav.language.backends.replay",
        "kznav.language.backends.qolda_avl",
    ],
)


def create_backend(
    spec: Mapping[str, Any],
    *,
    prompts: Optional[PromptLibrary] = None,
    prompt_root: Optional[Path] = None,
    sink: Optional[CallSink] = None,
) -> VLMBackend:
    cls = BACKENDS.resolve(spec)
    lib = prompts or load_prompt_library(prompt_root)
    backend = cls(spec, lib, sink)  # type: ignore[call-arg]
    if spec.get("parse_replay"):                       # D-078: parse from recorded calls, verify live
        from .backends.parse_replay import ParseReplayBackend

        return ParseReplayBackend(backend, lib)  # type: ignore[return-value]
    return backend
