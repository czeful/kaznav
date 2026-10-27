"""SigLIP 2 image-text relevance for the value map (HF transformers).

``score`` returns the cosine similarity between the image embedding and the embedding of
``prompt_template.format(text)`` for each text (VLFM uses the analogous BLIP-2 ITM cosine).
Text embeddings are cached: the hypotheses stay fixed within an episode.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .base import VALUE_ENCODERS, ValueEncoder


@VALUE_ENCODERS.register("siglip2")
class Siglip2Encoder(ValueEncoder):
    """SigLIP 2 relevance of the whole image to each goal hypothesis (cosine similarity).

    Args:
        spec: Encoder config: ``model_id``, ``prompt_template`` (e.g. ``"a photo of a {}."``), ``dtype``, ``device``.
    """
    def __init__(self, spec: Mapping[str, Any]):
        super().__init__(spec)
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = str(spec.get("device", "cuda"))
        self.dtype = torch.float16 if spec.get("dtype", "fp16") == "fp16" else torch.float32
        self.model_id = str(spec["model_id"])
        self.template = str(spec["prompt_template"])
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
        self._text_cache: Dict[str, Any] = {}

    def _text_embeddings(self, texts: Sequence[str]):
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            prompts = [self.template.format(t) for t in missing]
            inputs = self.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            for t, e in zip(missing, emb):
                self._text_cache[t] = e
        return self.torch.stack([self._text_cache[t] for t in texts])

    def score(self, rgb: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        """Cosine similarity between the image embedding and each text's template embedding (shape ``[len(texts)]``)."""
        if not texts:
            return np.zeros(0)
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with self.torch.inference_mode():
            img = self.model.get_image_features(**inputs)
        img = img / img.norm(dim=-1, keepdim=True)
        txt = self._text_embeddings(texts)
        return (txt @ img[0]).float().cpu().numpy()
