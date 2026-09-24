"""Image encoding for model requests and for the audit log."""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image


def encode_image(image: np.ndarray, fmt: str = "png", max_side: Optional[int] = None) -> bytes:
    """Encode an HxWx3 uint8 array. PNG by default: lossless, so the logged file is exactly the input."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 uint8 image, got {image.dtype} {image.shape}")
    img = Image.fromarray(image[..., :3])
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    return buf.getvalue()


def data_uri(data: bytes, fmt: str) -> str:
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(data).decode('ascii')}"
