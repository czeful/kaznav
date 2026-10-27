"""Ground-truth detector from the semantic sensor — an upper-bound ABLATION, never the method.

Returns one detection (score 1.0) per goal instance whose semantic id covers at least
``min_pixels`` pixels; the box is the bounding box of those pixels and the mask is exact.
The runner passes ``aux = {"semantic": image, "goal_ids": {query_index: [semantic ids]}}``.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from .base import DETECTORS, Detection, Detector


@DETECTORS.register("oracle_semantic")
class OracleSemanticDetector(Detector):
    """Ground-truth detector from the semantic sensor: the navigation ceiling, never the method.

    Args:
        spec: Detector config; ``min_pixels`` is the smallest visible goal area reported.
    """
    def __init__(self, spec: Mapping[str, Any]):
        super().__init__(spec)
        self.min_pixels = int(spec.get("min_pixels", 50))

    def detect(self, rgb: np.ndarray, queries: Sequence[str], aux: Optional[Mapping[str, Any]] = None) -> List[Detection]:
        """One detection per visible goal instance (score 1, exact mask).

        Raises:
            ValueError: If ``aux`` lacks the semantic image or the goal ids.
        """
        if not aux or aux.get("semantic") is None:
            raise ValueError("oracle_semantic needs aux['semantic'] and aux['goal_ids']")
        sem = np.asarray(aux["semantic"])
        out: List[Detection] = []
        for qi, ids in dict(aux["goal_ids"]).items():
            for sid in ids:
                mask = sem == sid
                if mask.sum() < self.min_pixels:
                    continue
                rr, cc = np.nonzero(mask)
                out.append(Detection((float(cc.min()), float(rr.min()), float(cc.max() + 1), float(rr.max() + 1)),
                                     1.0, int(qi), queries[int(qi)], mask=mask))
        return out
