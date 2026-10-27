"""SAM 2 box-prompted masks (D-060): the object's contour inside a detector box, as VLFM / ApexNav do with Mobile-SAM.

The image is encoded once per frame (the policy segments all of a frame's detections in a row, same array); each box
is a prompt to the mask decoder. The mask is clipped to the box. A mask below ``min_pixels`` returns None, and the
policy falls back to the box itself (with the D-054 point filter). ``precise = True`` tells the object memory that
the mask has no background, so the front-layer depth band (a stand-in for a segmenter) is not applied.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .base import SEGMENTERS, Segmenter


@SEGMENTERS.register("sam2")
class Sam2Segmenter(Segmenter):
    """SAM 2 box-prompted segmenter (D-060); see the module docstring.

    Args:
        spec: Segmenter config: ``model_id``, ``min_pixels``, ``device``.
    """
    precise = True

    def __init__(self, spec: Mapping[str, Any]):
        super().__init__(spec)
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = str(spec.get("device", "cuda"))
        self.model_id = str(spec["model_id"])
        self.min_pixels = int(spec.get("min_pixels", 20))
        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()
        self._frame: Optional[np.ndarray] = None
        self._embeddings = None
        self._sizes = None
        self.last_latency_s = float("nan")
        self.empty_masks = 0

    def _encode(self, rgb: np.ndarray) -> None:
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            self._embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        self._sizes = inputs["original_sizes"]
        self._frame = rgb

    def segment(self, rgb: np.ndarray, box_xyxy: Sequence[float]) -> Optional[np.ndarray]:
        """Mask of the object inside ``box_xyxy``, clipped to the box; None if it has fewer than ``min_pixels``."""
        t0 = time.perf_counter()
        if rgb is not self._frame:
            self._encode(rgb)
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        boxes = self.processor(input_boxes=[[[x1, y1, x2, y2]]], original_sizes=self._sizes.tolist(),
                               return_tensors="pt")["input_boxes"].to(self.device)
        with self.torch.inference_mode():
            out = self.model(input_boxes=boxes, image_embeddings=self._embeddings, multimask_output=False)
        mask = self.processor.post_process_masks(out.pred_masks.cpu(), self._sizes.cpu())[0]
        mask = np.asarray(mask).reshape(-1, h, w)[0].astype(bool)
        clip = np.zeros((h, w), bool)
        clip[max(int(y1), 0):max(int(np.ceil(y2)), 0), max(int(x1), 0):max(int(np.ceil(x2)), 0)] = True
        mask &= clip
        self.last_latency_s = time.perf_counter() - t0
        if mask.sum() < self.min_pixels:
            self.empty_masks += 1
            return None
        return mask

    def describe_config(self) -> Mapping[str, Any]:
        """Class, config and timing for the manifest."""
        return {**super().describe_config(), "model_id": self.model_id}
