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
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from ..registry import Registry


@dataclass(frozen=True)
class Detection:
    """One detected box for one query.

    Attributes:
        box_xyxy: Box in pixels, ``(x1, y1, x2, y2)``.
        score: Detector confidence.
        query_index: Index of the query (hypothesis) in the list passed to ``detect``.
        phrase: The phrase the model grounded; may differ from the query.
        mask: Optional instance mask (H x W bool) from detectors that produce one; otherwise the pipeline's
            segmenter makes it from the box.
    """
    box_xyxy: "tuple[float, float, float, float]"  # pixels, (x1, y1, x2, y2)
    score: float
    query_index: int       # index into the query list passed to ``detect``
    phrase: str            # phrase the model grounded (may differ from the query)
    #: Optional instance mask (H x W bool) from detectors that produce one; otherwise the
    #: pipeline's Segmenter makes it from the box.
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


class Detector(ABC):
    """Open-vocabulary detector interface: boxes for a list of text queries in one image.

    Args:
        spec: The detector config block (``configs/detector/<name>.yaml``).
    """
    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    @abstractmethod
    def detect(self, rgb: np.ndarray, queries: Sequence[str], aux: Optional[Mapping[str, Any]] = None) -> List[Detection]:
        """Detections for all queries in one image; score threshold is a config value.

        ``aux`` carries ground-truth side channels for oracle / diagnostic detectors only
        (e.g. the semantic image); real detectors must ignore it.
        """

    def describe_config(self) -> Mapping[str, Any]:
        """Class and config of the detector, for the run manifest."""
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "spec": self.spec}


class Segmenter(ABC):
    #: True when masks hold only the object (no background): the object memory then skips the D-054 depth band.
    """Segmenter interface: the object's mask inside a detector box.

    Args:
        spec: The segmenter config block (``configs/segmenter/<name>.yaml``).
    """
    precise = False

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    @abstractmethod
    def segment(self, rgb: np.ndarray, box_xyxy: Sequence[float]) -> Optional[np.ndarray]:
        """Boolean HxW mask of the object inside ``box_xyxy``; None when the segmenter finds nothing there."""

    def describe_config(self) -> Mapping[str, Any]:
        """Class and config of the segmenter, for the run manifest."""
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "spec": self.spec}


class ValueEncoder(ABC):
    """Image-text relevance interface for the semantic value map.

    Args:
        spec: The value-encoder config block (``configs/value_encoder/<name>.yaml``).
    """
    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)

    @abstractmethod
    def score(self, rgb: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        """One relevance score per text for the whole image (shape [len(texts)])."""

    def describe_config(self) -> Mapping[str, Any]:
        """Class and config of the encoder, for the run manifest."""
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "spec": self.spec}


DETECTORS: Registry[Detector] = Registry("detector", builtin_modules=["kznav.perception.mock", "kznav.perception.oracle",
                                                                  "kznav.perception.grounding_dino",
                                                                  "kznav.perception.coco_detector"])
SEGMENTERS: Registry[Segmenter] = Registry("segmenter", builtin_modules=["kznav.perception.mock", "kznav.perception.sam"])
VALUE_ENCODERS: Registry[ValueEncoder] = Registry("value encoder", builtin_modules=["kznav.perception.mock", "kznav.perception.siglip"])


def create_detector(spec: Mapping[str, Any]) -> Detector:
    """Create the detector selected by a config block (``type`` or ``class_path``)."""
    return DETECTORS.resolve(spec)(spec)  # type: ignore[call-arg]


def create_segmenter(spec: Mapping[str, Any]) -> Segmenter:
    """Create the segmenter selected by a config block (``type`` or ``class_path``)."""
    return SEGMENTERS.resolve(spec)(spec)  # type: ignore[call-arg]


def create_value_encoder(spec: Mapping[str, Any]) -> ValueEncoder:
    """Create the value encoder selected by a config block (``type`` or ``class_path``)."""
    return VALUE_ENCODERS.resolve(spec)(spec)  # type: ignore[call-arg]
