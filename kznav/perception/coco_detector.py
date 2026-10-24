"""Closed-set COCO detector for the VLFM reference row (D-062): VLFM detects its COCO target classes with YOLOv7 at
confidence 0.8 (Grounding DINO only for non-COCO classes). Here RT-DETR (R101, Objects365 + COCO; 56.2 COCO AP, the
YOLOv7-E6E class at 56.8) from transformers - no extra packages.

A closed-set detector knows its 80 classes, not language: each query is mapped to a COCO class through ``aliases``
(the benchmark's category names) or by its own name; a query that maps to no class gets no detections. So this
detector belongs to the baseline rows only - it would bypass the language step of the method (D-060).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .base import DETECTORS, Detection, Detector
from .grounding_dino import box_area_frac


@DETECTORS.register("coco_closed_set")
class CocoClosedSetDetector(Detector):
    def __init__(self, spec: Mapping[str, Any]):
        super().__init__(spec)
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch = torch
        self.device = str(spec.get("device", "cuda"))
        self.model_id = str(spec["model_id"])
        self.box_threshold = float(spec["box_threshold"])
        self.max_box_area_frac = float(spec.get("max_box_area_frac", 1.0))
        self.aliases: Dict[str, str] = {str(k).lower(): str(v).lower() for k, v in (spec.get("aliases") or {}).items()}
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        self.class_ids = {name.lower(): int(i) for i, name in self.model.config.id2label.items()}
        self.unmapped_queries = 0
        self.dropped_large_boxes = 0
        self.last_latency_s = float("nan")

    def coco_class(self, query: str) -> Optional[int]:
        q = query.strip().lower()
        return self.class_ids.get(self.aliases.get(q, q))

    def detect(self, rgb: np.ndarray, queries: Sequence[str], aux: Optional[Mapping[str, Any]] = None) -> List[Detection]:
        wanted = {}
        for qi, q in enumerate(queries):
            cid = self.coco_class(q)
            if cid is None:
                self.unmapped_queries += 1
            else:
                wanted.setdefault(cid, qi)
        if not wanted:
            return []
        from PIL import Image

        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(outputs, threshold=self.box_threshold,
                                                           target_sizes=[(h, w)])[0]
        self.last_latency_s = time.perf_counter() - t0
        out: List[Detection] = []
        for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
            qi = wanted.get(int(label))
            if qi is None:
                continue
            if box_area_frac(box, w, h) > self.max_box_area_frac:
                self.dropped_large_boxes += 1
                continue
            out.append(Detection(tuple(float(v) for v in box), float(score), qi, self.model.config.id2label[int(label)]))
        return out

    def describe_config(self) -> Mapping[str, Any]:
        return {**super().describe_config(), "model_id": self.model_id}
