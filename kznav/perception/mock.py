"""GPU-free stand-ins for the vision models (tests and smoke runs only)."""
from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

from .base import DETECTORS, SEGMENTERS, VALUE_ENCODERS, Detection, Detector, Segmenter, ValueEncoder


@DETECTORS.register("none")
class NoDetector(Detector):
    def detect(self, rgb: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        return []


@SEGMENTERS.register("box")
class BoxSegmenter(Segmenter):
    """Mask = the box itself. Also a legitimate ablation ("no segmentation")."""

    def segment(self, rgb: np.ndarray, box_xyxy: Sequence[float]) -> np.ndarray:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
        return mask


@VALUE_ENCODERS.register("hash")
class HashValueEncoder(ValueEncoder):
    """Deterministic pseudo-scores in [0, 1] from (image bytes, text)."""

    def score(self, rgb: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        img_h = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        out = []
        for t in texts:
            d = hashlib.sha256(f"{img_h}|{t}".encode("utf-8")).digest()
            out.append(int.from_bytes(d[:8], "big") / 2**64)
        return np.asarray(out, dtype=float)
