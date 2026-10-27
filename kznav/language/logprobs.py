"""Turn first-token top-logprobs into probabilities over answer options.

Byte-level BPE tokenizers (Qwen3, and therefore Qolda) split Cyrillic answers
into single letters: "иә" -> "и", "ә". The first generated token is thus often
just the first letter, so a token matches an option when it is a prefix of one
of the option's surface forms (or the surface form is a prefix of it, for
tokens like "yes."). Prompt files must use options whose surface forms start
with different letters; ``assert_options_distinguishable`` enforces that.

The fraction of probability mass that fell on known options (``covered``) is
reported for every call: a low value means the model started its answer with
something else (a quote, a space, another word) and P(target) rests on a
small part of the distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

_STRIP_CHARS = " \t\n\r\"'«»“”„`.,:;!?()[]"


def normalize_token(tok: str) -> str:
    """Strip quotes, spaces and punctuation from a token and casefold it."""
    return tok.strip(_STRIP_CHARS).casefold()


def assert_options_distinguishable(options: Mapping[str, Sequence[str]]) -> None:
    """Check that answer options start with different letters, so a one-letter first token is unambiguous.

    Raises:
        ValueError: If two options share a first letter or an option has no (or a blank) surface form.
    """
    firsts: Dict[str, str] = {}
    for label, surfaces in options.items():
        if not surfaces:
            raise ValueError(f"option {label!r} has no surface forms")
        for s in surfaces:
            n = normalize_token(s)
            if not n:
                raise ValueError(f"option {label!r} has a blank surface form")
            other = firsts.get(n[0])
            if other is not None and other != label:
                raise ValueError(
                    f"options {other!r} and {label!r} share the first letter {n[0]!r}; "
                    "single-letter first tokens would be ambiguous"
                )
            firsts[n[0]] = label


@dataclass(frozen=True)
class OptionProbabilities:
    """Probability of each answer option from the first token's top log-probabilities.

    Attributes:
        probs: Probabilities renormalised over the options; empty if no token matched an option.
        mass: Raw probability mass per option.
        covered: Total mass that fell on known options.
        ambiguous: Mass of tokens that matched more than one option.
        matched: The tokens counted for each option.
    """
    probs: Dict[str, float]          # renormalised over options; empty if nothing matched
    mass: Dict[str, float]           # raw probability mass per option
    covered: float                   # total mass on options
    ambiguous: float                 # mass of tokens that matched more than one option
    matched: Dict[str, List[str]] = field(default_factory=dict)

    def p(self, label: str) -> Optional[float]:
        """Probability of option ``label``, or None if no token matched any option."""
        return self.probs.get(label) if self.probs else None


def option_probabilities(top_logprobs: Mapping[str, float], options: Mapping[str, Sequence[str]]) -> OptionProbabilities:
    """Sum the first-token probability mass per answer option and renormalise over the options.

    Args:
        top_logprobs: ``{token: logprob}`` of the first generated token.
        options: ``{label: [surface forms]}``, e.g. ``{"yes": ["yes"], "no": ["no"]}``.

    Returns:
        Per-option probabilities with the coverage diagnostics.
    """
    surf = {lab: [normalize_token(s) for s in ss] for lab, ss in options.items()}
    mass = {lab: 0.0 for lab in options}
    matched: Dict[str, List[str]] = {lab: [] for lab in options}
    ambiguous = 0.0
    for tok, lp in top_logprobs.items():
        n = normalize_token(tok)
        if not n:
            continue
        hits = [lab for lab, ss in surf.items() if any(s.startswith(n) or n.startswith(s) for s in ss)]
        p = math.exp(lp)
        if len(hits) == 1:
            mass[hits[0]] += p
            matched[hits[0]].append(tok)
        elif len(hits) > 1:
            ambiguous += p
    covered = sum(mass.values())
    probs = {lab: m / covered for lab, m in mass.items()} if covered > 0 else {}
    return OptionProbabilities(probs, mass, covered, ambiguous, matched)


def label_from_text(text: str, options: Mapping[str, Sequence[str]]) -> Optional[str]:
    """Option whose surface form the answer text starts with, else None."""
    n = normalize_token(text.split()[0]) if text.split() else ""
    if not n:
        return None
    for lab, ss in options.items():
        if any(n.startswith(normalize_token(s)) for s in ss):
            return lab
    return None
