"""Qolda-AVL (issai/Qolda-AVL-5B): ISSAI's Kazakh audio-vision-language model; it hears the command (D-077).

Qwen3-VL-4B-Thinking with a fine-tuned Whisper-large-v3-turbo encoder; Apache 2.0. The model runs in-process
with transformers 5.x and its own remote code (venv ~/avl-env; ~/qolda-env stays on transformers 4.57.6 for the
Qolda server). Its 10.2 GB of bf16 weights do not fit on the 12 GB GPU next to the robot's perception, so the
spoken commands are parsed before the robot runs (tools/parse_speech_commands.py) and the runs replay those
parses (``backend=replay`` with the same prompt): the parse happens once per command either way.

``input: speech`` - the parser gets (text, lang) like every backend, finds the command's recording in the speech
bank (kznav/language/speech.py) and sends the recording in place of the text: the last user turn of the prompt
(speech/parse_goal/v1, which is default/parse_goal/v4 verbatim otherwise) becomes [audio, audio_instruction].
The command text never reaches the model; it only names the command (and the replay key). ``input: text`` - the
same model reads the command with the text arms' prompt, to tell the model's effect from the speech effect.

Qolda-AVL is a thinking model: the chat template opens a ``<think>`` block. ``thinking: native`` lets it think
and parses only what follows ``</think>`` (no ``</think>`` within max_new_tokens = an invalid answer, retried);
``thinking: skip`` prefills an empty think block. The thoughts are logged with the call (``inputs.thinking``).
Verification is not offered (the robot runs verify with the local SigLIP verifier, D-075).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..base import BackendCapabilities, BackendConfigError, CallRecord, VLMBackend
from ..goal_parsing import ParseLimits, run_parse_loop
from ..prompts import PromptLibrary, build_parse_messages
from ..registry import BACKENDS
from ..schemas import GoalHypotheses, VerificationResult, VerifyQuery
from ..speech import LEVELS, SpeechBank, SpeechClip

REPO = Path(__file__).resolve().parents[3]
INPUTS = ("speech", "text")
THINKING = ("native", "skip")
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def split_thinking(raw: str, thinking: str) -> Tuple[str, str]:
    """(thoughts, answer). The prompt already opened ``<think>``: native output is "thoughts</think>answer"."""
    if thinking == "skip":
        return "", raw.strip()
    if "</think>" not in raw:
        return raw, ""                              # thinking did not finish: no answer
    thought, answer = raw.split("</think>", 1)
    return thought.replace("<think>", "").strip(), answer.strip()


class HFAVLRuntime:
    """The model and processor, loaded once (transformers >= 5, trust_remote_code)."""

    def __init__(self, spec: Mapping[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        repo, rev = str(spec["hf_repo"]), spec.get("revision")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(repo, revision=rev, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=True,
                                                          dtype=torch.bfloat16, device_map=str(spec.get("device", "cuda")))
        self.model.eval()

    def generate(self, messages: List[dict], audio: Optional[Sequence[np.ndarray]], max_new_tokens: int,
                 thinking: str) -> str:
        """Generate a reply to chat ``messages`` with optional audio clips, greedily.

        Args:
            messages: Chat messages; audio parts are placeholders filled from ``audio`` in order.
            audio: Mono 16 kHz clips, one per audio part, or None for a text-only request.
            max_new_tokens: Generation budget (thinking included).
            thinking: ``native`` lets the model open its think block; ``skip`` prefills an empty one.

        Returns:
            The decoded new tokens without end-of-turn markers.
        """
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if thinking == "skip":
            prompt += "\n</think>\n\n"                 # the template ends with "<think>\n": an empty think block
        inputs = self.processor(text=[prompt], audio=list(audio) if audio else None, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        text = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for t in END_TOKENS:
            text = text.replace(t, "")
        return text


@BACKENDS.register("qolda_avl")
class QoldaAVLBackend(VLMBackend):
    """Qolda-AVL parses a spoken (or, as a control, typed) command; it does not verify (D-077).

    See the module docstring for the speech and thinking modes.

    Args:
        spec: The backend config block (``configs/backend/qolda_avl.yaml``).
        prompts: Prompt library.
        sink: Receives one ``CallRecord`` per call.
        runtime: Optional model runtime (tests inject a fake); loaded lazily otherwise.
        speech_bank: Optional speech bank (tests inject one); built from the config otherwise.

    Raises:
        BackendConfigError: If the input mode, thinking mode or parse prompt are inconsistent.
    """
    def __init__(self, spec: Mapping[str, Any], prompts: PromptLibrary, sink=None, runtime=None,
                 speech_bank: Optional[SpeechBank] = None):
        super().__init__(spec, prompts, sink)
        s = self.spec
        for key in ("hf_repo", "max_new_tokens", "capabilities"):
            if key not in s:
                raise BackendConfigError(f"{self.name}: config is missing {key!r}")
        self.input = str(s.get("input", "speech"))
        self.thinking = str(s.get("thinking", "native"))
        if self.input not in INPUTS:
            raise BackendConfigError(f"{self.name}: input must be one of {INPUTS}")
        if self.thinking not in THINKING:
            raise BackendConfigError(f"{self.name}: thinking must be one of {THINKING}")
        audio_prompt = self.parse_prompt.data.get("input", "text") == "audio"
        if audio_prompt != (self.input == "speech"):
            raise BackendConfigError(f"{self.name}: input={self.input} needs a {'audio' if self.input == 'speech' else 'text'}"
                                     f" parse prompt, got {self.parse_prompt.ref}")
        p = s.get("parse") or {}
        self.parse_limits = ParseLimits(max_attempts=int(p.get("max_attempts", 3)),
                                        min_hypotheses=int(p.get("min_hypotheses", 1)),
                                        max_hypotheses=int(p.get("max_hypotheses", 8)))
        self._caps = BackendCapabilities.from_spec(s["capabilities"])
        self._runtime = runtime
        self._bank = speech_bank

    @property
    def capabilities(self) -> BackendCapabilities:
        """Capabilities declared in the config (no logprobs, no images: parsing only)."""
        return self._caps

    @property
    def runtime(self):
        """The model runtime, loaded on first use (10 GB of weights; only the offline parse tool needs it)."""
        if self._runtime is None:
            self._runtime = HFAVLRuntime(self.spec)
        return self._runtime

    @property
    def bank(self) -> SpeechBank:
        """The speech bank of the configured speaker, built on first use from the command bank and the recordings.

        Raises:
            BackendConfigError: If ``speech.root``, ``speech.speaker`` or ``speech.commands`` is missing, or
                ``speech.level`` is unknown.
        """
        if self._bank is None:
            from ..commands import NaturalCommands

            sp = self.spec.get("speech") or {}
            for key in ("root", "speaker", "commands"):
                if not sp.get(key):
                    raise BackendConfigError(f"{self.name}: input=speech needs speech.{key}")
            level = str(sp.get("level", "none"))            # D-085; absent = as recorded
            if level not in LEVELS:
                raise BackendConfigError(f"{self.name}: speech.level must be one of {LEVELS}, got {level!r}")
            bank = NaturalCommands(REPO / sp["commands"], REPO, role="debug")
            self._bank = SpeechBank.from_commands(REPO / sp["root"], str(sp["speaker"]), bank.commands, level)
        return self._bank

    def _max_new_tokens(self, task: str) -> int:
        return int(self.spec["max_new_tokens"][task])

    def _audio_turn(self, instruction: str) -> dict:
        return {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": instruction}]}

    # ------------------------------------------------------------------ hooks
    def _parse_goal(self, text: str, lang: str, rec: CallRecord) -> GoalHypotheses:
        clip: Optional[SpeechClip] = None
        command_turn = len(build_parse_messages(self.parse_prompt, text)) - 1
        if self.input == "speech":
            clip = self.bank.clip(text, lang)
            rec.inputs["audio"] = clip.describe()
        rec.inputs["thinking_mode"] = self.thinking
        thoughts: List[str] = rec.inputs.setdefault("thinking", [])

        def generate(messages: List[dict]) -> str:
            if clip is not None:
                messages = list(messages)
                messages[command_turn] = self._audio_turn(str(self.parse_prompt.data["audio_instruction"]))
            raw = self.runtime.generate(messages, [clip.audio] if clip is not None else None,
                                        self._max_new_tokens("parse_goal"), self.thinking)
            thought, answer = split_thinking(raw, self.thinking)
            thoughts.append(thought)
            return answer

        return run_parse_loop(text, self.parse_prompt, self.parse_limits, generate, rec)

    def transcribe(self, clip: SpeechClip) -> Tuple[str, str]:
        """(transcript, thoughts): diagnostic speech recognition of a recording, outside the call log."""
        instruction = str(self.spec.get("transcribe_instruction") or "Transcribe this audio verbatim.")
        raw = self.runtime.generate([self._audio_turn(instruction)], [clip.audio],
                                    self._max_new_tokens("transcribe"), self.thinking)
        thought, answer = split_thinking(raw, self.thinking)
        return answer, thought

    def _verify(self, image: np.ndarray, query: VerifyQuery, rec: CallRecord) -> VerificationResult:
        raise BackendConfigError(f"{self.name}: parses commands only; verify with policy.verification.backend="
                                 f"siglip_zero (D-075)")

    def _describe(self, image: np.ndarray, prompt: str, rec: CallRecord) -> str:
        raise BackendConfigError(f"{self.name}: parses commands only")
