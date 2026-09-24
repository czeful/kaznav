"""Qwen-VL family (e.g. Qwen3-VL-4B-Instruct) behind an OpenAI-compatible server (vLLM / lmdeploy / fallback).

Qwen3-VL *Instruct* checkpoints have no thinking mode; the *Thinking*
checkpoints are separate models and are rejected unless declared as a control
measurement, mirroring the Qolda rule.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..base import BackendConfigError
from ..prompts import PromptLibrary
from ..registry import BACKENDS
from .openai_compat import OpenAICompatBackend


@BACKENDS.register("qwen_vl")
class QwenVLBackend(OpenAICompatBackend):
    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink=None, transport=None):
        super().__init__(spec, prompts, sink, transport)
        hf = str(self.spec.get("hf_repo", ""))
        if "thinking" in hf.lower() and not self.spec.get("control_think_measurement", False):
            raise BackendConfigError(f"{self.name}: {hf} is a Thinking checkpoint; use the Instruct variant")
        ctk = (self.extra_body.get("chat_template_kwargs") or {}).get("enable_thinking")
        if ctk not in (None, False) and not self.spec.get("control_think_measurement", False):
            raise BackendConfigError(f"{self.name}: enable_thinking must be false outside control measurements")
