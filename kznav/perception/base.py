"""Interfaces for the vision models of the pipeline.

``Detector``      open-vocabulary detection for a list of text queries (GroundingDINO, ...)
``Segmenter``     mask for one box (SAM2 / MobileSAM, ...)
``ValueEncoder``  image-text relevance for the value map (BLIP-2 ITM, SigLIP 2, ...)

All take multiple queries at once because the goal is a *set* of weighted
English hypotheses; fusion by weight happens in the value map / policy, not in
the model wrappers. Implementations register in the registries below and are
selected by name in ``configs/detector``, ``configs/value_encoder``, ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

import numpy as np

from ..registry import Registry


@dataclass(frozen=True)
class Detection:
    box_xyxy: "tuple[float, float, float, float]"  # pixels, (x1, y1, x2, y2)
    score: float
    query_index: int       # index into the query list passed to ``detect``
    phrase: str            # phrase the model grounded (may differ from the query)


class Detector(ABC):
    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    @abstractmethod
    def detect(self, rgb: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        """Detections for all queries in one image; score threshold is a config value."""

    def describe_config(self) -> Mapping[str, Any]:
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "spec": self.spec}


class Segmenter(ABC):
    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    @abstractmethod
    def segment(self, rgb: np.ndarray, box_xyxy: Sequence[float]) -> np.ndarray:
        """Boolean HxW mask of the object inside ``box_xyxy``."""

    def describe_config(self) -> Mapping[str, Any]:
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "spec": self.spec}


class ValueEncoder(ABC):
    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    @abstractmethod
    def score(self, rgb: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        """One relevance score per text for the whole image (shape [len(texts)])."""

    def describe_config(self) -> Mapping[str, Any]:
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "spec": self.spec}


DETECTORS: Registry[Detector] = Registry("detector", builtin_modules=["kznav.perception.mock"])
SEGMENTERS: Registry[Segmenter] = Registry("segmenter", builtin_modules=["kznav.perception.mock"])
VALUE_ENCODERS: Registry[ValueEncoder] = Registry("value encoder", builtin_modules=["kznav.perception.mock"])


def create_detector(spec: Mapping[str, Any]) -> Detector:
    return DETECTORS.resolve(spec)(spec)  # type: ignore[call-arg]


def create_segmenter(spec: Mapping[str, Any]) -> Segmenter:
    return SEGMENTERS.resolve(spec)(spec)  # type: ignore[call-arg]


def create_value_encoder(spec: Mapping[str, Any]) -> ValueEncoder:
    return VALUE_ENCODERS.resolve(spec)(spec)  # type: ignore[call-arg]
