"""Recover one JSON object from raw model text.

Ported from the pilot (``qolda check/qolda_pilot/json_utils.py``) unchanged in
behaviour: strip <think> blocks, prefer a ```json fence, else the first
balanced {...} span, and tolerate trailing commas. The goal is that measured
JSON-validity failures are the model's, not the parser's.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JSONExtractionError(ValueError):
    """No syntactically valid JSON object could be recovered."""


def strip_think_block(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def _find_balanced_braces(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = strip_think_block(raw_text)
    candidates: List[str] = []
    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    braces = _find_balanced_braces(text)
    if braces:
        candidates.append(braces)
    candidates.append(text.strip())

    last_err: Optional[Exception] = None
    for cand in candidates:
        for attempt in (cand, _TRAILING_COMMA.sub(r"\1", cand)):
            try:
                obj = json.loads(attempt)
            except Exception as e:  # noqa: BLE001 - try the next candidate
                last_err = e
                continue
            if isinstance(obj, dict):
                return obj
    raise JSONExtractionError(f"could not recover a JSON object: {last_err}")
