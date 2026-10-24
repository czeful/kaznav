"""Commands given to the robot: human-written natural commands (the benchmark) or developer templates (debug).

Natural command bank (``format: kznav.natural_commands/v1``, D-048; authoring rules in docs/command_authoring.md):
one entry per command intent, written by a native speaker in every language of the study (parallel texts), so a
language contrast changes the language and nothing else::

    format: kznav.natural_commands/v1
    dataset: hm3d_objectnav_v1
    lexicon: lexicon/hm3d_objectnav.yaml
    languages: [kk, kk_ru, en]
    min_reviews: 1                     # approvals by native speakers other than the author
    commands:
      - id: chair-a01-01
        category: chair
        author: A01                    # anonymised; the key that links to the consent form, never a name
        split: test                    # dev (prompt and threshold development) | test (reported runs)
        style: direct                  # direct | need | indirect | descriptive (docs/command_authoring.md)
        texts: {kk: "...", kk_ru: "...", en: "..."}
        reviews: [{reviewer: R02, verdict: ok, note: ""}]    # verdict: ok | fix | reject
        status: approved               # draft | approved | rejected

Which command an episode gets depends on the episode id only (``stable_index``), and is the same in every
language and every arm. A benchmark-role run uses approved test commands only and refuses a category without one;
a dev-role run uses dev commands (drafts allowed). Template files (``templates:``) are the developer's drafts for
the debug sets; a template command is chosen per episode and language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from ..sim.objectnav_dataset import stable_index
from .lexicon import Lexicon, normalize

FORMAT = "kznav.natural_commands/v1"
SPLITS = ("dev", "test")
STATUSES = ("draft", "approved", "rejected")
VERDICTS = ("ok", "fix", "reject")
STYLES = ("direct", "need", "indirect", "descriptive")
# Language of the object name in a command language (kk_ru commands name objects in Russian, D-039).
WORD_LANG = {"kk": "kk", "kk_ru": "ru", "en": "en"}
_CYR = re.compile(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")
_KK_ONLY = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")
_LAT = re.compile(r"[A-Za-z]")


@dataclass
class NaturalCommand:
    id: str
    category: str
    author: str
    split: str
    texts: Dict[str, str]
    style: str = "direct"
    reviews: List[Dict[str, str]] = field(default_factory=list)
    status: str = "draft"
    notes: str = ""

    def approvals(self) -> int:
        return sum(1 for r in self.reviews if r.get("verdict") == "ok" and r.get("reviewer") != self.author)


@dataclass
class Issue:
    level: str                  # error | warning
    command_id: str
    message: str


def _name_in(text: str, names: Sequence[str]) -> bool:
    """A lexicon form occurs in the text, allowing suffixes on the last token (Kazakh case endings)."""
    toks = normalize(text).split()
    for form in names:
        ft = normalize(form).split()
        n = len(ft)
        for i in range(len(toks) - n + 1):
            if toks[i : i + n - 1] == ft[:-1] and toks[i + n - 1].startswith(ft[-1]):
                return True
    return False


class NaturalCommands:
    """A natural command bank; ``role`` (benchmark | dev | debug) decides which commands are usable."""

    kind = "natural"

    def __init__(self, path: Path, repo: Path, role: str = "dev"):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: format must be {FORMAT!r}, got {data.get('format')!r}")
        self.path, self.role = str(path), role
        self.dataset = str(data.get("dataset", ""))
        self.languages = [str(x) for x in data.get("languages", ["kk", "kk_ru", "en"])]
        self.min_reviews = int(data.get("min_reviews", 1))
        self.lexicon = Lexicon.load(Path(repo) / data["lexicon"])
        self.commands = [NaturalCommand(
            id=str(c["id"]), category=str(c["category"]), author=str(c.get("author", "")), split=str(c.get("split", "")),
            texts={str(k): str(v) for k, v in (c.get("texts") or {}).items()}, style=str(c.get("style", "direct")),
            reviews=[{str(k): str(v) for k, v in r.items()} for r in (c.get("reviews") or [])],
            status=str(c.get("status", "draft")), notes=str(c.get("notes", ""))) for c in data.get("commands") or []]
        self.by_id = {c.id: c for c in self.commands}
        self._usable: Dict[str, List[NaturalCommand]] = {}
        for c in self.commands:
            if self.usable(c):
                self._usable.setdefault(c.category, []).append(c)
        for v in self._usable.values():
            v.sort(key=lambda c: c.id)                  # file order must not matter

    @property
    def review_status(self) -> str:
        used = [c for v in self._usable.values() for c in v]
        if used and all(c.status == "approved" for c in used):
            return "approved"
        return "draft_needs_native_review"

    def usable(self, c: NaturalCommand) -> bool:
        if self.role == "benchmark":
            return c.split == "test" and c.status == "approved" and c.approvals() >= self.min_reviews
        if self.role == "dev":
            return c.split == "dev" and c.status != "rejected"
        return c.status != "rejected"                       # debug: everything not rejected

    def candidates(self, category: str) -> List[NaturalCommand]:
        if category not in self._usable:
            raise KeyError(f"{self.path}: no usable {self.role}-role command for category {category!r} "
                           f"(benchmark runs need approved test commands)")
        return self._usable[category]

    def pick(self, episode_id: str, category: str) -> NaturalCommand:
        cands = self.candidates(category)
        return cands[stable_index(f"{episode_id}|command", len(cands))]

    def command(self, episode_id: str, lang: str, category: str) -> Tuple[str, str]:
        """(command id, text): the same command intent for every language of the episode."""
        c = self.pick(episode_id, category)
        if not c.texts.get(lang, "").strip():
            raise KeyError(f"{self.path}: command {c.id} has no {lang!r} text")
        return c.id, c.texts[lang].strip()

    def object_name(self, lang: str, category: str) -> Tuple[str, str]:
        wl = WORD_LANG[lang]
        return (self.lexicon.categories[category].names.get(wl) or [category])[0], wl

    # ------------------------------------------------------------------ checks
    def validate(self, categories: Optional[Sequence[str]] = None, min_test_per_category: int = 0) -> List[Issue]:
        """Schema and content checks. Errors make the bank unusable for a benchmark run; warnings are reported."""
        out: List[Issue] = []
        err = lambda cid, m: out.append(Issue("error", cid, m))           # noqa: E731
        warn = lambda cid, m: out.append(Issue("warning", cid, m))        # noqa: E731
        seen_ids, seen_texts = set(), {}
        for c in self.commands:
            if c.id in seen_ids:
                err(c.id, "duplicate id")
            seen_ids.add(c.id)
            if c.category not in self.lexicon.categories:
                err(c.id, f"category {c.category!r} is not in the lexicon")
            if c.split not in SPLITS:
                err(c.id, f"split must be one of {SPLITS}")
            if c.status not in STATUSES:
                err(c.id, f"status must be one of {STATUSES}")
            if c.style not in STYLES:
                warn(c.id, f"style {c.style!r} is not one of {STYLES}")
            if not c.author:
                err(c.id, "author id missing")
            for r in c.reviews:
                if r.get("verdict") not in VERDICTS:
                    err(c.id, f"review verdict must be one of {VERDICTS}")
                if r.get("reviewer") == c.author:
                    err(c.id, "an author cannot review their own command")
            if c.status == "approved":
                if c.approvals() < self.min_reviews:
                    err(c.id, f"approved with {c.approvals()} approval(s) by others; needs {self.min_reviews}")
                if any(r.get("verdict") in ("fix", "reject") for r in c.reviews[-1:]):
                    err(c.id, "approved, but the last review asks for a fix or rejects it")
            for lang in self.languages:
                t = c.texts.get(lang, "").strip()
                if not t:
                    err(c.id, f"no {lang} text")
                    continue
                key = (lang, normalize(t))
                if key in seen_texts:
                    warn(c.id, f"{lang} text duplicates {seen_texts[key]}")
                seen_texts[key] = c.id
                if lang in ("kk", "kk_ru") and not _CYR.search(t):
                    err(c.id, f"{lang} text has no Cyrillic letters")
                if lang == "kk" and _LAT.search(t):
                    warn(c.id, "kk text contains Latin letters")
                if lang == "en" and _CYR.search(t):
                    err(c.id, "en text contains Cyrillic letters")
            kk, kkru = c.texts.get("kk", ""), c.texts.get("kk_ru", "")
            if kk and kkru and normalize(kk) == normalize(kkru):
                warn(c.id, "kk and kk_ru texts are identical (kk_ru should mix in Russian)")
            if kkru and not _KK_ONLY.search(kkru) and not re.search(r"\b(тауып|бар|қайда|маған|керек|әкел)", kkru.lower()):
                warn(c.id, "kk_ru text looks fully Russian (no Kazakh letters or common Kazakh words)")
        for cat in categories or []:
            n = sum(1 for c in self.commands if c.category == cat and c.split == "test" and c.status == "approved")
            if n < min_test_per_category:
                out.append(Issue("error" if n == 0 else "warning", f"[{cat}]",
                                 f"{n} approved test commands; the plan asks for >= {min_test_per_category}"))
        return out

    def coverage(self) -> List[Dict[str, Any]]:
        """Per category: commands by split and status, authors, and how often each language names the object with
        a lexicon form (the rest refer to it indirectly - the hard cases)."""
        rows = []
        for cat in sorted({c.category for c in self.commands}):
            cs = [c for c in self.commands if c.category == cat]
            row: Dict[str, Any] = {"category": cat, "word_type": self.lexicon.categories[cat].word_type
                                   if cat in self.lexicon.categories else "?"}
            for sp in SPLITS:
                row[f"{sp} approved"] = sum(1 for c in cs if c.split == sp and c.status == "approved")
                row[f"{sp} draft"] = sum(1 for c in cs if c.split == sp and c.status == "draft")
            row["authors"] = len({c.author for c in cs})
            names = self.lexicon.categories[cat].names if cat in self.lexicon.categories else {}
            for lang in self.languages:
                forms = names.get(WORD_LANG.get(lang, lang), []) + (names.get("kk", []) if lang == "kk_ru" else [])
                texts = [c.texts.get(lang, "") for c in cs if c.texts.get(lang)]
                row[f"{lang} names object"] = f"{sum(_name_in(t, forms) for t in texts)}/{len(texts)}"
            rows.append(row)
        return rows


class TemplateCommands:
    """Developer templates (commands/<dataset>.yaml with ``templates:``): debug sets and drafts only."""

    kind = "templates"

    def __init__(self, path: Path, repo: Path, role: str = "dev"):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.path, self.role = str(path), role
        self.templates: Dict[str, List[Dict[str, str]]] = data["templates"]
        self.review_status = str(data.get("review_status", "unspecified"))
        self.lexicon = Lexicon.load(Path(repo) / data["lexicon"])

    def command(self, episode_id: str, lang: str, category: str) -> Tuple[str, str]:
        """(template id, text); the template is chosen from the episode id, so it never depends on the subset."""
        ts = self.templates[lang]
        t = ts[stable_index(f"{episode_id}|{lang}", len(ts))]
        names = self.lexicon.categories[category].names
        text = t["text"].format(**{k: (names.get(k) or [category])[0] for k in ("kk", "ru", "en")})
        return t["id"], text[0].upper() + text[1:]

    def object_name(self, lang: str, category: str) -> Tuple[str, str]:
        wl = WORD_LANG[lang]
        return (self.lexicon.categories[category].names.get(wl) or [category])[0], wl


def load_commands(path: Path, repo: Path, role: str = "dev"):
    """NaturalCommands or TemplateCommands, by the file's content."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, Mapping) and data.get("format") == FORMAT:
        return NaturalCommands(path, repo, role)
    if isinstance(data, Mapping) and "templates" in data:
        return TemplateCommands(path, repo, role)
    raise ValueError(f"{path}: neither a natural command bank ({FORMAT}) nor a template file")
