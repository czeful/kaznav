"""Versioned prompt files.

A prompt is referenced as ``<set>/<task>/v<N>`` and lives in
``prompts/<set>/<task>/v<N>.yaml``. The SHA-256 of the file (line endings
normalised to LF, so Windows and Linux checkouts agree) is recorded with every
model call and in the run manifest: a changed prompt is a changed experiment.
Prompt files are never edited after they have been used for a reported run —
a new version is added instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .logprobs import assert_options_distinguishable

TASKS = ("parse_goal", "verify_candidate", "describe")


@dataclass(frozen=True)
class PromptTemplate:
    ref: str
    sha256: str
    data: Mapping[str, Any]

    @property
    def task(self) -> str:
        return str(self.data["task"])

    @property
    def review_status(self) -> str:
        return str(self.data.get("review_status", "unspecified"))


class PromptLibrary:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: Dict[str, PromptTemplate] = {}

    def path_for(self, ref: str) -> Path:
        parts = ref.strip("/").split("/")
        if len(parts) != 3 or not parts[2].startswith("v"):
            raise ValueError(f"prompt ref must look like '<set>/<task>/v<N>', got {ref!r}")
        return self.root / parts[0] / parts[1] / f"{parts[2]}.yaml"

    def load(self, ref: str) -> PromptTemplate:
        if ref not in self._cache:
            path = self.path_for(ref)
            raw = path.read_bytes().replace(b"\r\n", b"\n")
            data = yaml.safe_load(raw.decode("utf-8"))
            _validate(ref, data)
            self._cache[ref] = PromptTemplate(ref, hashlib.sha256(raw).hexdigest(), data)
        return self._cache[ref]

    def all_refs(self) -> List[str]:
        refs = []
        for p in sorted(self.root.glob("*/*/v*.yaml")):
            refs.append(f"{p.parent.parent.name}/{p.parent.name}/{p.stem}")
        return refs


def _validate(ref: str, data: Mapping[str, Any]) -> None:
    task = data.get("task")
    if task not in TASKS:
        raise ValueError(f"{ref}: task must be one of {TASKS}, got {task!r}")
    _, task_dir, ver = ref.split("/")
    if task_dir != task:
        raise ValueError(f"{ref}: file is under {task_dir!r} but declares task {task!r}")
    if f"v{data.get('version')}" != ver:
        raise ValueError(f"{ref}: declared version {data.get('version')!r} does not match file name")
    if task == "parse_goal":
        for key in ("system", "user_template", "retry_template"):
            if key not in data:
                raise ValueError(f"{ref}: missing {key!r}")
    elif task == "verify_candidate":
        langs = data.get("languages") or {}
        if not langs:
            raise ValueError(f"{ref}: missing 'languages'")
        for code, block in langs.items():
            for key in ("system", "user_template", "options"):
                if key not in block:
                    raise ValueError(f"{ref}: language {code!r} missing {key!r}")
            if set(block["options"]) != {"yes", "no"}:
                raise ValueError(f"{ref}: language {code!r} options must be exactly 'yes' and 'no'")
            assert_options_distinguishable(block["options"])
        for cmd_lang, verify_lang in (data.get("lang_map") or {}).items():
            if verify_lang not in langs:
                raise ValueError(f"{ref}: lang_map {cmd_lang!r} -> {verify_lang!r} has no prompt block")
    elif task == "describe":
        if "user_template" not in data:
            raise ValueError(f"{ref}: missing 'user_template'")


def _system(data: Mapping[str, Any]) -> List[dict]:
    s = data.get("system")
    return [{"role": "system", "content": str(s).strip()}] if s else []


def build_parse_messages(tpl: PromptTemplate, text: str) -> List[dict]:
    d = tpl.data
    msgs = _system(d)
    for ex in d.get("fewshot") or []:
        msgs.append({"role": "user", "content": str(ex["user"]).strip()})
        msgs.append({"role": "assistant", "content": str(ex["assistant"]).strip()})
    msgs.append({"role": "user", "content": str(d["user_template"]).format(text=text)})
    return msgs


def build_parse_retry_messages(tpl: PromptTemplate, text: str, bad_response: str, error: str) -> List[dict]:
    msgs = build_parse_messages(tpl, text)
    msgs.append({"role": "assistant", "content": bad_response})
    msgs.append({"role": "user", "content": str(tpl.data["retry_template"]).format(error=error)})
    return msgs


def verify_block(tpl: PromptTemplate, lang: str) -> Tuple[str, Mapping[str, Any]]:
    d = tpl.data
    verify_lang = (d.get("lang_map") or {}).get(lang, lang)
    langs = d["languages"]
    if verify_lang not in langs:
        raise KeyError(f"{tpl.ref}: no verification prompt for language {lang!r}")
    return verify_lang, langs[verify_lang]


def build_verify_messages(tpl: PromptTemplate, target: str, lang: str) -> Tuple[List[dict], Mapping[str, Sequence[str]], str]:
    """Text-only messages (the caller attaches the image to the last user turn), options, verify language."""
    verify_lang, block = verify_block(tpl, lang)
    msgs = _system(block)
    msgs.append({"role": "user", "content": str(block["user_template"]).format(target=target)})
    return msgs, block["options"], verify_lang


def build_describe_messages(tpl: PromptTemplate, prompt: str) -> List[dict]:
    d = tpl.data
    return _system(d) + [{"role": "user", "content": str(d["user_template"]).format(prompt=prompt)}]


def fewshot_targets(tpl: PromptTemplate) -> List[str]:
    return [str(t) for t in (tpl.data.get("fewshot_targets") or [])]


def prompt_summary(tpl: Optional[PromptTemplate]) -> Dict[str, str]:
    if tpl is None:
        return {}
    return {"ref": tpl.ref, "sha256": tpl.sha256, "review_status": tpl.review_status}
