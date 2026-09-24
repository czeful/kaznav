"""The episode's goal as the language model understood it: command -> target + weighted hypotheses.

``parse_command`` runs the backend's goal parser once per command and scores the result against
the lexicon (privileged, for the language columns of the metrics row; the policy only receives the
hypotheses). Parsing at temperature 0 is deterministic, so ``GoalCache`` reuses the parse of a
command already seen in the run; the call's cost is still charged to every episode that uses it
(``parse_record``), as a deployed robot would parse each command it is given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BackendError, CallRecord, ListSink
from .lexicon import Lexicon


@dataclass
class LanguageGoal:
    command_id: str
    text: str
    lang: str
    category: str
    word_type: str
    parse_ok: bool
    parse_attempts: int
    target: str = ""
    hypotheses: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    rooms: List[str] = field(default_factory=list)
    target_match: Optional[bool] = None
    hypothesis_match: Optional[bool] = None
    error: str = ""
    parse_record: Optional[CallRecord] = None

    def summary(self) -> Dict[str, Any]:
        return {"command_id": self.command_id, "text": self.text, "lang": self.lang, "category": self.category,
                "parse_ok": self.parse_ok, "target": self.target, "hypotheses": self.hypotheses,
                "weights": self.weights, "rooms": self.rooms, "target_match": self.target_match,
                "hypothesis_match": self.hypothesis_match, "error": self.error}


def parse_command(backend: Any, calls: ListSink, *, command_id: str, text: str, lang: str, category: str,
                  word_type: str, lexicon: Optional[Lexicon]) -> LanguageGoal:
    """``calls`` must be (or feed) the backend's sink: the parse call's record is taken from it."""
    n0 = len(calls.records)
    try:
        g = backend.parse_goal(text, lang)
    except BackendError as e:
        rec = calls.records[-1] if len(calls.records) > n0 else None
        return LanguageGoal(command_id, text, lang, category, word_type, False,
                            sum(1 for a in rec.attempts if a.kind == "json") if rec else 0,
                            error=str(e)[:300], parse_record=rec)
    rec = calls.records[-1] if len(calls.records) > n0 else None
    tm = hm = None
    if lexicon is not None and category in lexicon.categories:
        tm = lexicon.match_target(category, g.target, lang)
        hm = lexicon.match_any_en(category, g.texts)
    return LanguageGoal(command_id, text, lang, category, word_type, True,
                        sum(1 for a in rec.attempts if a.kind == "json") if rec else 1,
                        g.target, g.texts, [float(w) for w in g.weights], list(g.rooms), tm, hm, parse_record=rec)


class GoalCache:
    def __init__(self, backend: Any, calls: ListSink, lexicon: Optional[Lexicon]):
        self.backend, self.calls, self.lexicon = backend, calls, lexicon
        self._cache: Dict[tuple, LanguageGoal] = {}

    def get(self, *, command_id: str, text: str, lang: str, category: str, word_type: str) -> LanguageGoal:
        key = (text, lang, category)
        if key not in self._cache:
            self._cache[key] = parse_command(self.backend, self.calls, command_id=command_id, text=text, lang=lang,
                                             category=category, word_type=word_type, lexicon=self.lexicon)
        g = self._cache[key]
        return LanguageGoal(**{**g.__dict__, "command_id": command_id})
