"""Model-independent goal-parsing loop: generate -> extract JSON -> validate -> retry with the error.

Shared by every backend (including the mock) so that JSON validity is judged
by exactly the same code for all models.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List

from pydantic import ValidationError

from .base import Attempt, CallRecord, InvalidOutputError, utc_now
from .jsonx import extract_json_object
from .prompts import PromptTemplate, build_parse_messages, build_parse_retry_messages
from .schemas import GoalHypotheses

MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class ParseLimits:
    max_attempts: int = 3
    min_hypotheses: int = 1
    max_hypotheses: int = 8


def describe_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())
    else:
        msg = str(e)
    return msg[:MAX_ERROR_CHARS]


def validate_goal_text(text: str, limits: ParseLimits) -> GoalHypotheses:
    goal = GoalHypotheses.model_validate(extract_json_object(text)).normalized()
    n = len(goal.hypotheses)
    if not limits.min_hypotheses <= n <= limits.max_hypotheses:
        raise ValueError(f"expected {limits.min_hypotheses}-{limits.max_hypotheses} distinct hypotheses, got {n}")
    return goal


def run_parse_loop(
    text: str,
    tpl: PromptTemplate,
    limits: ParseLimits,
    generate: Callable[[List[dict]], str],
    rec: CallRecord,
) -> GoalHypotheses:
    """``generate(messages) -> raw text`` may raise TransportError; that is not retried here."""
    messages = build_parse_messages(tpl, text)
    last_error = ""
    for attempt in range(1, limits.max_attempts + 1):
        started = utc_now()
        t0 = time.perf_counter()
        raw = generate(messages)
        try:
            goal = validate_goal_text(raw, limits)
        except ValueError as e:
            last_error = describe_error(e)
            rec.attempts.append(Attempt("json", started, time.perf_counter() - t0, False, error=last_error, text=raw))
            if attempt < limits.max_attempts:
                rec.json_retries += 1
                messages = build_parse_retry_messages(tpl, text, raw, last_error)
            continue
        rec.attempts.append(Attempt("json", started, time.perf_counter() - t0, True, text=raw))
        return goal
    raise InvalidOutputError(f"no valid goal JSON after {limits.max_attempts} attempts; last error: {last_error}")
