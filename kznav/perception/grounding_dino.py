"""GroundingDINO open-vocabulary detector (HF transformers implementation, D-009).

All queries (the English visual hypotheses) go into one caption "q1 . q2 . q3 ." as in the
original GroundingDINO usage; each returned phrase is mapped back to the query it overlaps most
(token Jaccard). Phrases that match no query are dropped and counted.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .base import DETECTORS, Detection, Detector

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return set(_TOK.findall(s.casefold()))


def match_query(phrase: str, queries: Sequence[str]) -> Optional[int]:
    """Index of the query sharing most tokens with ``phrase`` (Jaccard), None if nothing overlaps."""
    pt = _tokens(phrase)
    if not pt:
        return None
    best, best_score = None, 0.0
    for i, q in enumerate(queries):
        qt = _tokens(q)
        if not qt:
            continue
        score = len(pt & qt) / len(pt | qt)
        if score > best_score:
            best, best_score = i, score
    return best


@DETECTORS.register("grounding_dino")
class GroundingDinoDetector(Detector):
    def __init__(self, spec: Mapping[str, Any]):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = str(spec.get("device", "cuda"))
        self.model_id = str(spec["model_id"])
        self.box_threshold = float(spec["box_threshold"])
        self.text_threshold = float(spec["text_threshold"])
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        self.unmatched_phrases = 0
        self.last_latency_s = float("nan")

    def detect(self, rgb: np.ndarray, queries: Sequence[str], aux: Optional[Mapping[str, Any]] = None) -> List[Detection]:
        if not queries:
            return []
        from PIL import Image

        caption = " . ".join(q.strip().lower() for q in queries) + " ."
        t0 = time.perf_counter()
        inputs = self.processor(images=Image.fromarray(rgb), text=caption, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        h, w = rgb.shape[:2]
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(h, w)])[0]
        self.last_latency_s = time.perf_counter() - t0
        labels = res.get("text_labels", res.get("labels"))
        out: List[Detection] = []
        for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            qi = match_query(str(label), queries)
            if qi is None:
                self.unmatched_phrases += 1
                continue
            out.append(Detection(tuple(float(v) for v in box), float(score), qi, str(label)))
        return out

    def describe_config(self) -> Dict[str, Any]:
        return {**super().describe_config(), "model_id": self.model_id}
