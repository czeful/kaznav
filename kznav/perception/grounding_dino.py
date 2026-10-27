"""GroundingDINO open-vocabulary detector (HF transformers implementation, D-009).

``query_mode`` (D-058):

* ``separate`` (default): every query (an English visual hypothesis) is its own caption "q ." with its own
  forward pass (the image is preprocessed once); a box belongs to the query of its caption. On the robot's camera the joint caption below
  separates goal-visible from goal-absent frames no better than chance (dev frames, AUROC 0.55 vs 0.61 for the
  same hypotheses one at a time; runs/diag_hyp_detect), and almost nothing passes the D-054 threshold.
* ``joint``: all queries in one caption "q1 . q2 . q3 ." as in the original GroundingDINO usage; each
  returned phrase is mapped back to the query it overlaps most (token Jaccard). Phrases that match no
  query are dropped and counted. With a single query both modes are the same computation.

``input_resize`` (D-059): ``default`` = the processor's resize (shortest side 800: a 640x480 frame becomes 1066x800);
``native`` = the frame at its own size, ~1.7x faster on the 640x480 cameras and as accurate on dev frames.

``max_box_area_frac`` (D-054): a box covering more of the image than this is dropped. From a camera 0.15 m above
the floor, close-up views of a wall or a toy fill the frame, and GroundingDINO then "finds" the queried object in
the whole image (0.6-0.8 confidence on val_mini). 1.0 = off; the value is chosen on dev with the score threshold.
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


def box_area_frac(box: Sequence[float], width: int, height: int) -> float:
    """Share of the image a (x1, y1, x2, y2) box covers, clipped to the image."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw = max(0.0, min(float(width), x2) - max(0.0, x1))
    bh = max(0.0, min(float(height), y2) - max(0.0, y1))
    return bw * bh / float(width * height)


@DETECTORS.register("grounding_dino")
class GroundingDinoDetector(Detector):
    """GroundingDINO / MM-GroundingDINO through HF transformers (see the module docstring for the modes).

    Args:
        spec: Detector config: ``model_id``, ``box_threshold``, ``text_threshold``, ``query_mode``,
            ``input_resize``, ``max_box_area_frac`` and ``device``.

    Raises:
        ValueError: If ``query_mode`` or ``input_resize`` has an unknown value.
    """
    def __init__(self, spec: Mapping[str, Any]):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = str(spec.get("device", "cuda"))
        self.model_id = str(spec["model_id"])
        self.box_threshold = float(spec["box_threshold"])
        self.text_threshold = float(spec["text_threshold"])
        self.max_box_area_frac = float(spec.get("max_box_area_frac", 1.0))
        self.query_mode = str(spec.get("query_mode", "separate"))
        if self.query_mode not in ("separate", "joint"):
            raise ValueError(f"detector.query_mode must be separate | joint, got {self.query_mode!r}")
        self.input_resize = str(spec.get("input_resize", "default"))
        if self.input_resize not in ("native", "default"):
            raise ValueError(f"detector.input_resize must be native | default, got {self.input_resize!r}")
        self.dropped_large_boxes = 0
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        self.unmatched_phrases = 0
        self.last_latency_s = float("nan")

    def detect(self, rgb: np.ndarray, queries: Sequence[str], aux: Optional[Mapping[str, Any]] = None) -> List[Detection]:
        """Detect every query in one image.

        Args:
            rgb: HxWx3 uint8 image.
            queries: English visual hypotheses.
            aux: Ignored (ground-truth side channels are for oracle detectors only).

        Returns:
            Detections at or above ``box_threshold``, each assigned to the query of its caption (``separate``) or
            to the query its phrase overlaps most (``joint``); boxes larger than ``max_box_area_frac`` are dropped.
        """
        if not queries:
            return []
        from PIL import Image

        qs = [q.strip().lower() for q in queries]
        captions = [q + " ." for q in qs] if self.query_mode == "separate" else [" . ".join(qs) + " ."]
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        # The image is preprocessed once; one forward pass per caption. Not a padded batch: padding the shorter
        # captions moves scores by up to 0.25 (dev frames) and saves little time.
        size = {"size": {"shortest_edge": min(h, w), "longest_edge": max(h, w)}} if self.input_resize == "native" else {}
        pixels = self.processor.image_processor(images=Image.fromarray(rgb), return_tensors="pt", **size).to(self.device)
        results = []
        for caption in captions:
            text = self.processor.tokenizer(caption, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**pixels, **text)
            results.append(self.processor.post_process_grounded_object_detection(
                outputs, text.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)])[0])
        self.last_latency_s = time.perf_counter() - t0
        out: List[Detection] = []
        for ci, res in enumerate(results):
            labels = res.get("text_labels", res.get("labels"))
            for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
                if box_area_frac(box, w, h) > self.max_box_area_frac:
                    self.dropped_large_boxes += 1
                    continue
                if self.query_mode == "separate":      # a phrase sharing no token with the caption is dropped,
                    qi = ci if match_query(str(label), [queries[ci]]) is not None else None   # as in joint mode
                else:
                    qi = match_query(str(label), queries)
                if qi is None:
                    self.unmatched_phrases += 1
                    continue
                out.append(Detection(tuple(float(v) for v in box), float(score), qi, str(label)))
        return out

    def describe_config(self) -> Dict[str, Any]:
        """Class, config and counters (unmatched phrases, dropped large boxes) for the manifest."""
        return {**super().describe_config(), "model_id": self.model_id}
