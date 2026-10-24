"""The episode's goal as the language model understood it: command -> target + weighted hypotheses.

``parse_command`` runs the backend's goal parser once per command and scores the result against
the lexicon (privileged, for the language columns of the metrics row; the policy only receives the
hypotheses). Parsing at temperature 0 is deterministic, so ``GoalCache`` reuses the parse of a
command already seen in the run; the call's cost is still charged to every episode that uses it
(``parse_record``), as a deployed robot would parse each command it is given. A transport failure (server
down, timeout after retries) is raised, not turned into a failed parse: it says nothing about the model.

``GoalSettings`` (config group ``goal``) decides what reaches the detector, the value map and the
verifier. ``multi`` is the method; the others are ablations (all but ``mt_word`` keep the same parse
prompt):

* ``top1`` (D-038): only the highest-weighted hypothesis, weight 1.0;
* ``mt_command`` (D-039): the translator turns the whole command into English first, and the
  pipeline runs in English (parse, detector queries, verification question);
* ``mt_word`` (D-039): VLFM with a translator in front. The goal is the object's name in the command
  language, given with the episode as the category is in standard ObjectNav (not parsed from the
  command, so no VLM call); the translator turns it into English, and that one phrase is the only
  query (weight 1.0). Verification, when on, asks about that name in the command language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import BackendConfigError, BackendError, CallRecord, CallSink, ListSink, TransportError
from .lexicon import Lexicon
from .translation import Translator, create_translator

TRANSLATION_MODES = ("none", "command", "word")


@dataclass(frozen=True)
class GoalSettings:
    name: str = "multi"
    max_hypotheses: Optional[int] = None    # None: all parsed hypotheses
    translation: str = "none"               # none | command | word

    def __post_init__(self) -> None:
        if self.translation not in TRANSLATION_MODES:
            raise BackendConfigError(f"goal.translation must be one of {TRANSLATION_MODES}, got {self.translation!r}")
        if self.max_hypotheses is not None and self.max_hypotheses < 1:
            raise BackendConfigError(f"goal.max_hypotheses must be >= 1 or null, got {self.max_hypotheses}")

    @classmethod
    def from_config(cls, goal: Mapping[str, Any]) -> "GoalSettings":
        mh = goal.get("max_hypotheses")
        st = cls(name=str(goal["name"]), max_hypotheses=None if mh is None else int(mh),
                 translation=str(goal.get("translation", "none")))
        tr = goal.get("translator")
        has_tr = bool(tr) and str(tr.get("type", "none")) != "none"
        if st.translation != "none" and not has_tr:
            raise BackendConfigError(f"goal {st.name!r}: translation={st.translation!r} needs goal.translator")
        if st.translation == "none" and has_tr:
            raise BackendConfigError(f"goal {st.name!r}: a translator is configured but translation is 'none'")
        return st


def with_target_name(texts: Sequence[str], weights: Sequence[float], name: str) -> Tuple[List[str], List[float]]:
    """D-072: the model's own short English name of the target (parse_goal v4 ``target_en``) as one more detector query,
    weighted like its most likely hypothesis, weights renormalised to 1. Nothing changes when the name is empty or is
    already one of the hypotheses. The name comes from the model, never from the episode's category."""
    texts, weights = list(texts), [float(w) for w in weights]
    if not name or not texts or any(t.casefold() == name.casefold() for t in texts):
        return texts, weights
    weights = weights + [max(weights)]
    total = sum(weights)
    return texts + [name], [w / total for w in weights]


def select_hypotheses(texts: Sequence[str], weights: Sequence[float],
                      max_n: Optional[int]) -> Tuple[List[str], List[float]]:
    """The ``max_n`` highest-weighted hypotheses (ties keep the parser's order), weights renormalised to 1."""
    texts, weights = list(texts), [float(w) for w in weights]
    if max_n is None or max_n >= len(texts):
        return texts, weights
    order = sorted(range(len(texts)), key=lambda i: -weights[i])[:max_n]
    order.sort()
    kept_t, kept_w = [texts[i] for i in order], [weights[i] for i in order]
    total = sum(kept_w)
    kept_w = [w / total for w in kept_w] if total > 0 else [1.0 / len(kept_w)] * len(kept_w)
    ranked = sorted(zip(kept_t, kept_w), key=lambda tw: -tw[1])
    return [t for t, _ in ranked], [w for _, w in ranked]


@dataclass
class LanguageGoal:
    command_id: str
    text: str
    lang: str                               # language of the command as given
    category: str
    word_type: str
    parse_ok: bool
    parse_attempts: int
    target: str = ""
    hypotheses: List[str] = field(default_factory=list)       # what the policy uses
    weights: List[float] = field(default_factory=list)
    rooms: List[str] = field(default_factory=list)
    target_match: Optional[bool] = None
    hypothesis_match: Optional[bool] = None                    # over ``hypotheses`` (the ones used)
    error: str = ""
    parse_record: Optional[CallRecord] = None
    goal_mode: str = "multi"
    pipeline_lang: str = ""                 # language of the parse and the verification question
    translation: str = ""                   # translator output (the command or the object name), "" if none
    parsed_hypotheses: List[str] = field(default_factory=list)  # everything the parser wrote
    translation_record: Optional[CallRecord] = None
    target_en: str = ""                     # the model's own plain English name of the target (parse_goal v3, D-071)

    @property
    def effective_lang(self) -> str:
        return self.pipeline_lang or self.lang

    def summary(self) -> Dict[str, Any]:
        return {"command_id": self.command_id, "text": self.text, "lang": self.lang, "category": self.category,
                "parse_ok": self.parse_ok, "target": self.target, "hypotheses": self.hypotheses,
                "weights": self.weights, "rooms": self.rooms, "target_match": self.target_match,
                "hypothesis_match": self.hypothesis_match, "error": self.error, "goal_mode": self.goal_mode,
                "pipeline_lang": self.effective_lang, "translation": self.translation,
                "parsed_hypotheses": self.parsed_hypotheses, "target_en": self.target_en}


def parse_command(backend: Any, calls: ListSink, *, command_id: str, text: str, lang: str, category: str,
                  word_type: str, lexicon: Optional[Lexicon], settings: Optional[GoalSettings] = None,
                  translator: Optional[Translator] = None, word: Optional[str] = None,
                  word_lang: Optional[str] = None) -> LanguageGoal:
    """``calls`` must be (or feed) the backend's sink: the parse call's record is taken from it.
    ``word`` / ``word_lang``: the object's name and its language, for ``translation == "word"`` only."""
    st = settings or GoalSettings()
    if st.translation != "none" and translator is None:
        raise BackendConfigError(f"goal {st.name!r}: translation={st.translation!r} needs a translator")
    base = dict(command_id=command_id, text=text, lang=lang, category=category, word_type=word_type)
    if st.translation == "word":
        return _translate_word(base, st, translator, word, word_lang, lexicon)
    parse_text, parse_lang, translated, tr_rec = text, lang, "", None
    n0 = len(calls.records)

    def failed(attempts: int, err: str, rec: Optional[CallRecord]) -> LanguageGoal:
        return LanguageGoal(**base, parse_ok=False, parse_attempts=attempts, error=err[:300], parse_record=rec,
                            goal_mode=st.name, pipeline_lang=parse_lang, translation=translated,
                            translation_record=translator.last_record if translator is not None else None)

    try:
        if st.translation == "command":
            translated, tr_rec = translator.translate(text, lang)
            parse_text, parse_lang = translated, translator.target_lang
        n0 = len(calls.records)
        g = backend.parse_goal(parse_text, parse_lang)
    except TransportError:
        raise          # the server, not the model: an infrastructure failure of the episode (D-046), never cached
    except BackendError as e:
        rec = calls.records[-1] if len(calls.records) > n0 else None
        if rec is not None and rec.method != "parse_goal":
            rec = None
        return failed(sum(1 for a in rec.attempts if a.kind == "json") if rec else 0, str(e), rec)
    rec = calls.records[-1] if len(calls.records) > n0 else None
    attempts = sum(1 for a in rec.attempts if a.kind == "json") if rec else 1
    texts, weights = select_hypotheses(g.texts, g.weights, st.max_hypotheses)
    tm = hm = None
    if lexicon is not None and category in lexicon.categories:
        tm = lexicon.match_target(category, g.target, parse_lang)
        hm = lexicon.match_any_en(category, texts)
    return LanguageGoal(**base, parse_ok=True, parse_attempts=attempts, target=g.target, hypotheses=texts,
                        weights=weights, rooms=list(g.rooms), target_match=tm, hypothesis_match=hm,
                        parse_record=rec, goal_mode=st.name, pipeline_lang=parse_lang, translation=translated,
                        parsed_hypotheses=list(g.texts), translation_record=tr_rec, target_en=g.target_en)


def _translate_word(base: Dict[str, Any], st: GoalSettings, translator: Translator, word: Optional[str],
                    word_lang: Optional[str], lexicon: Optional[Lexicon]) -> LanguageGoal:
    if not word or not word_lang:
        raise BackendConfigError(f"goal {st.name!r}: translation='word' needs the object name and its language")
    if word_lang == translator.target_lang:
        raise BackendConfigError(f"goal {st.name!r}: the object name is already in {word_lang!r}")
    common = dict(goal_mode=st.name, pipeline_lang=base["lang"], target=word)
    try:
        translated, tr_rec = translator.translate(word, word_lang)
    except TransportError:
        raise
    except BackendError as e:
        return LanguageGoal(**base, **common, parse_ok=False, parse_attempts=0, error=f"translation: {e}"[:300],
                            translation_record=translator.last_record)
    hm = None
    if lexicon is not None and base["category"] in lexicon.categories:
        hm = lexicon.match_any_en(base["category"], [translated])
    return LanguageGoal(**base, **common, parse_ok=True, parse_attempts=0, hypotheses=[translated], weights=[1.0],
                        hypothesis_match=hm, translation=translated, translation_record=tr_rec, target_en=translated)


class GoalCache:
    def __init__(self, backend: Any, calls: ListSink, lexicon: Optional[Lexicon],
                 settings: Optional[GoalSettings] = None, translator: Optional[Translator] = None):
        self.backend, self.calls, self.lexicon = backend, calls, lexicon
        self.settings = settings or GoalSettings()
        self.translator = translator
        self._cache: Dict[tuple, LanguageGoal] = {}

    def get(self, *, command_id: str, text: str, lang: str, category: str, word_type: str,
            word: Optional[str] = None, word_lang: Optional[str] = None) -> LanguageGoal:
        key = (text, lang, category, word, word_lang)
        if key not in self._cache:
            self._cache[key] = parse_command(self.backend, self.calls, command_id=command_id, text=text, lang=lang,
                                             category=category, word_type=word_type, lexicon=self.lexicon,
                                             settings=self.settings, translator=self.translator, word=word,
                                             word_lang=word_lang)
        g = self._cache[key]
        return LanguageGoal(**{**g.__dict__, "command_id": command_id})


def make_goal_cache(goal_cfg: Mapping[str, Any], backend: Any, calls: ListSink, lexicon: Optional[Lexicon],
                    translator_sink: Optional[CallSink] = None) -> GoalCache:
    """Goal settings + translator from the ``goal`` config group. The translator logs to ``translator_sink``
    (the run's call log) and shares the backend's call context (run id, episode id)."""
    settings = GoalSettings.from_config(goal_cfg)
    translator = create_translator(goal_cfg.get("translator"), sink=translator_sink)
    if translator is not None:
        translator.context = backend.context
    return GoalCache(backend, calls, lexicon, settings, translator)
