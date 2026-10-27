"""Qolda (issai/Qolda, InternViT-300M + Qwen3-4B) behind an OpenAI-compatible server.

Everything Qolda-specific lives here and in ``configs/backend/qolda*.yaml``.
The one hard rule from the pilot is enforced at construction time: No-Think
mode must be requested explicitly in every request (``enable_thinking: false``
in ``extra_body``), otherwise the server may default to thinking and return
an empty answer. Think mode is only allowed for a declared control measurement.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..base import BackendConfigError
from ..prompts import PromptLibrary
from ..registry import BACKENDS
from .openai_compat import OpenAICompatBackend


@BACKENDS.register("qolda")
class QoldaBackend(OpenAICompatBackend):
    """Qolda (issai/Qolda) behind the project's OpenAI-compatible server.

    Adds one rule to ``OpenAICompatBackend``: No-Think mode must be requested explicitly in every request;
    Think mode is accepted only for the declared control measurement.

    Raises:
        BackendConfigError: If ``extra_body.enable_thinking`` is missing, or true outside a control measurement.
    """
    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink=None, transport=None):
        super().__init__(spec, prompts, sink, transport)
        thinking = self.extra_body.get("enable_thinking", None)
        if thinking is None:
            raise BackendConfigError(
                f"{self.name}: extra_body.enable_thinking must be set explicitly (false) in every request"
            )
        if thinking is not False and not self.spec.get("control_think_measurement", False):
            raise BackendConfigError(
                f"{self.name}: Think mode is only allowed with control_think_measurement: true"
            )
