"""A verifier backend without a language model (D-075): P(the crop shows the named object) from SigLIP 2 features.

The verifier (kznav.policy.verification) asks its backend ``verify_candidate(crop, VerifyQuery)``; this backend
answers it locally, so calibration, the stop threshold, accumulation of views, the hypothesis belief and the last
metre stay exactly as with a VLM. The name asked about is the query's target - with ``question_lang: en`` the model's
own short English name of the target (parse_goal v4 ``target_en``, D-071); the pipeline never supplies the category.

Two modes:
* ``siglip_head`` - a small MLP on [image, text, image * text] embeddings, trained on robot-camera crops of train
  scenes used nowhere else (tools/make_verifier_data.py, tools/train_verifier_head.py);
* ``siglip_zero`` - SigLIP's own sigmoid for "a photo of a <name>." (no training).

The SigLIP model is the policy's value-map encoder (already on the GPU): no second copy is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..language.schemas import VerificationResult, VerifyQuery

MODES = ("siglip_head", "siglip_zero")


class SiglipVerifierBackend:
    """Verifier backend that answers locally with SigLIP 2 instead of a language model (D-075).

    Args:
        encoder: The policy's SigLIP value encoder (its model and text cache are reused).
        mode: ``siglip_zero`` (SigLIP's own sigmoid) or ``siglip_head`` (a trained MLP head).
        head_file: Checkpoint of the head (``siglip_head`` only).

    Raises:
        ValueError: If the mode is unknown, the encoder is missing, or the head was trained for another encoder or
            prompt template.
    """
    def __init__(self, encoder: Any, mode: str, head_file: Optional[str] = None):
        if mode not in MODES:
            raise ValueError(f"local verifier mode must be one of {MODES}, not {mode!r}")
        if encoder is None or not hasattr(encoder, "model"):
            raise ValueError("the local verifier needs the SigLIP value encoder (value_encoder=siglip2_base)")
        self.name, self.enc = mode, encoder
        self.torch = encoder.torch
        self.head = None
        if mode == "siglip_head":
            if not head_file:
                raise ValueError("siglip_head needs policy.verification.head_file")
            ck = self.torch.load(Path(head_file), map_location="cpu")
            if ck["model_id"] != encoder.model_id:
                raise ValueError(f"head {head_file} was trained on {ck['model_id']}, the encoder is {encoder.model_id}")
            if ck["template"] != encoder.template:
                raise ValueError(f"head template {ck['template']!r} differs from the encoder's {encoder.template!r}")
            nn = self.torch.nn
            self.head = nn.Sequential(nn.Linear(int(ck["in_dim"]), int(ck["hidden"])), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(int(ck["hidden"]), 1))
            self.head.load_state_dict(ck["state_dict"])
            self.head.to(encoder.device).eval()
        model = encoder.model
        self.scale = float(model.logit_scale.exp())
        self.bias = float(model.logit_bias)

    def _image(self, rgb: np.ndarray):
        from PIL import Image

        enc = self.enc
        inputs = enc.processor(images=Image.fromarray(np.ascontiguousarray(rgb)), return_tensors="pt").to(enc.device)
        with self.torch.inference_mode():
            e = enc.model.get_image_features(pixel_values=inputs["pixel_values"].to(enc.dtype))
        return (e / e.norm(dim=-1, keepdim=True)).float()[0]

    def p_target(self, rgb: np.ndarray, name: str) -> float:
        """P(the image shows ``name``) from SigLIP: the trained head, or SigLIP's own sigmoid on the cosine."""
        img = self._image(rgb)
        txt = self.enc._text_embeddings([name])[0].float()
        with self.torch.inference_mode():
            if self.head is not None:
                x = self.torch.cat([img, txt, img * txt]).unsqueeze(0)
                return float(self.torch.sigmoid(self.head(x))[0, 0])
            return float(self.torch.sigmoid(self.torch.tensor(self.scale * float(img @ txt) + self.bias)))

    def verify_candidate(self, image: np.ndarray, query: VerifyQuery) -> VerificationResult:
        """Answer the verifier's question about a crop, in the same form as a VLM backend."""
        p = self.p_target(image, query.target)
        return VerificationResult(p_target=p, source=self.name, coverage=1.0, answer_text=f"{p:.4f}",
                                  label_from_text=None)
