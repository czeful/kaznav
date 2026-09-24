"""Structured outputs of the language layer.

``GoalHypotheses`` is the contract between the goal parser (any VLM) and the
rest of the pipeline. Validation is strict where the downstream consumer
would break (English-only hypotheses for the detector, weights in [0, 1]) and
lenient where it would not (extra keys are ignored but visible in the raw log).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def _clean_text(v: str) -> str:
    return " ".join(str(v).split())


class VisualHypothesis(BaseModel):
    en: str = Field(description="English visual description for an open-vocabulary detector")
    w: float = Field(ge=0.0, le=1.0, description="Prior weight; normalised by the pipeline")

    @field_validator("en")
    @classmethod
    def _english_only(cls, v: str) -> str:
        v = _clean_text(v)
        if not v:
            raise ValueError("hypothesis text is blank")
        if _CYRILLIC.search(v):
            raise ValueError(f"hypothesis must be English (Latin script), got {v!r}")
        return v


class TargetAttributes(BaseModel):
    material: List[str] = Field(default_factory=list)
    color: List[str] = Field(default_factory=list)
    size: List[str] = Field(default_factory=list)

    @field_validator("material", "color", "size")
    @classmethod
    def _strip(cls, v: List[str]) -> List[str]:
        return [s for s in (_clean_text(x) for x in v) if s]


class GoalHypotheses(BaseModel):
    target: str = Field(description="Target concept as named in the command, in the command language")
    hypotheses: List[VisualHypothesis] = Field(min_length=1, max_length=8)
    attributes: TargetAttributes = Field(default_factory=TargetAttributes)
    rooms: List[str] = Field(default_factory=list)
    landmarks: List[str] = Field(default_factory=list)

    @field_validator("target")
    @classmethod
    def _target_not_blank(cls, v: str) -> str:
        v = _clean_text(v)
        if not v:
            raise ValueError("target is blank")
        return v

    @field_validator("rooms", "landmarks")
    @classmethod
    def _strip_list(cls, v: List[str]) -> List[str]:
        return [s for s in (_clean_text(x) for x in v) if s]

    def normalized(self) -> "GoalHypotheses":
        """Merge case-insensitive duplicates, normalise weights to sum 1, sort by weight (stable)."""
        merged: Dict[str, VisualHypothesis] = {}
        order: List[str] = []
        for h in self.hypotheses:
            key = h.en.casefold()
            if key in merged:
                merged[key] = VisualHypothesis(en=merged[key].en, w=min(1.0, merged[key].w + h.w))
            else:
                merged[key] = h
                order.append(key)
        hyps = [merged[k] for k in order]
        total = sum(h.w for h in hyps)
        if total <= 0:
            weights = [1.0 / len(hyps)] * len(hyps)
        else:
            weights = [h.w / total for h in hyps]
        ranked = sorted(zip(hyps, weights), key=lambda hw: -hw[1])
        return self.model_copy(update={"hypotheses": [VisualHypothesis(en=h.en, w=w) for h, w in ranked]})

    @property
    def texts(self) -> List[str]:
        return [h.en for h in self.hypotheses]

    @property
    def weights(self) -> List[float]:
        return [h.w for h in self.hypotheses]


@dataclass(frozen=True)
class VerifyQuery:
    """What the verifier is asked about one image crop."""

    target: str   # target concept in the command language (from GoalHypotheses.target)
    lang: str     # command language code: en | kk | kk_ru | ru


@dataclass(frozen=True)
class VerificationResult:
    #: P(crop shows the target), uncalibrated, or None if the backend could not provide it.
    p_target: Optional[float]
    #: logprobs | unavailable | mock | replay
    source: str
    #: Probability mass of the first answer token that fell on a known option (diagnostic).
    coverage: Optional[float]
    answer_text: str
    #: Label read from the greedy text (True/False) or None if it matched no option.
    label_from_text: Optional[bool]

    def as_dict(self) -> dict:
        return asdict(self)
