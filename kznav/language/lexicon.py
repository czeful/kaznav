"""Category lexicon: accepted synonyms per language and word type.

Replaces opaque string equality from the pilot with explicit, reviewable
synonym lists. Matching rules (tested in tests/test_lexicon.py):

* normalisation: casefold, underscores/hyphens -> spaces, punctuation stripped,
  whitespace collapsed; English tokens get a light plural strip
  ("chairs" -> "chair", "couches" -> "couch"); applied to BOTH sides.
* ``match_en``: a synonym matches a hypothesis if the synonym's token sequence
  occurs contiguously in the hypothesis ("large cast-iron cauldron" matches
  "cauldron"). Used for "does any English hypothesis name the category?".
* ``match_target``: the parsed target equals one of the accepted forms of the
  command language after normalisation (no substring matching: inflected forms
  must be listed explicitly, which keeps the metric transparent).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _plural_strip(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es") and tok[:-2].endswith(("s", "x", "ch", "sh")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def normalize(text: str, *, english: bool = False) -> str:
    t = str(text).casefold().replace("_", " ").replace("-", " ")
    t = _PUNCT.sub(" ", t)
    toks = t.split()
    if english:
        toks = [_plural_strip(x) for x in toks]
    return " ".join(toks)


def _contains_phrase(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    return n > 0 and any(list(haystack[i : i + n]) == list(needle) for i in range(len(haystack) - n + 1))


@dataclass
class Category:
    name: str
    word_type: str                              # universal | cultural
    names: Dict[str, List[str]]                 # lang -> accepted forms (lang in en/kk/ru)
    review_status: str = "unspecified"
    dataset_labels: List[str] = field(default_factory=list)

    def en(self) -> List[str]:
        return self.names.get("en", [])


class Lexicon:
    def __init__(self, categories: Iterable[Category], source: str = ""):
        self.categories: Dict[str, Category] = {}
        for c in categories:
            if c.name in self.categories:
                raise ValueError(f"duplicate category {c.name!r} in lexicon {source}")
            if c.word_type not in ("universal", "cultural"):
                raise ValueError(f"{c.name}: word_type must be universal|cultural, got {c.word_type!r}")
            self.categories[c.name] = c
        self.source = source

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cats = [
            Category(
                name=str(c["name"]),
                word_type=str(c["word_type"]),
                names={str(k): [str(x) for x in v] for k, v in (c.get("names") or {}).items()},
                review_status=str(c.get("review_status", data.get("review_status", "unspecified"))),
                dataset_labels=[str(x) for x in c.get("dataset_labels", [])],
            )
            for c in data["categories"]
        ]
        return cls(cats, source=str(path))

    def __getitem__(self, name: str) -> Category:
        return self.categories[name]

    def match_en(self, category: str, hypothesis: str) -> bool:
        hyp = normalize(hypothesis, english=True).split()
        return any(_contains_phrase(hyp, normalize(s, english=True).split()) for s in self[category].en())

    def match_any_en(self, category: str, hypotheses: Iterable[str]) -> bool:
        return any(self.match_en(category, h) for h in hypotheses)

    def match_target(self, category: str, target: str, lang: str) -> Optional[bool]:
        """None if the lexicon has no accepted forms for this language (metric undefined, not False)."""
        forms = self._forms_for(category, lang)
        if not forms:
            return None
        t = normalize(target, english=(lang == "en"))
        return any(t == normalize(f, english=(lang == "en")) for f in forms)

    def _forms_for(self, category: str, lang: str) -> List[str]:
        names = self[category].names
        if lang == "kk_ru":  # mixed commands: the target may be named in either language
            return names.get("kk", []) + names.get("ru", [])
        return names.get(lang, [])

    def find_in_text(self, text: str, langs: Sequence[str] = ("kk", "ru", "en")) -> Optional[Mapping[str, str]]:
        """First category whose accepted form occurs in ``text``, allowing suffixes on each token
        ("орындықты" contains "орындық"). A crude stemmer for the MOCK backend only — never a metric."""
        toks = normalize(text).split()
        for c in self.categories.values():
            for lang in langs:
                for form in c.names.get(lang, []):
                    ft = normalize(form).split()
                    n = len(ft)
                    for i in range(len(toks) - n + 1):
                        if all(toks[i + j].startswith(ft[j]) for j in range(n)):
                            return {"category": c.name, "form": form, "lang": lang}
        return None
