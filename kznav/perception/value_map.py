"""Semantic value map (VLFM-style): how promising each observed cell looks for the goal.

Every frame the value encoder scores the whole image against the goal hypotheses; the score
v = sum_k w_k * s_k (hypothesis weights w_k from the belief) is written into the cells seen in
that frame with a confidence that falls off towards the image border:
    c = cos^2( (angle from the optical axis) / (hfov/2) * pi/2 )
and fused with the previous estimate by confidence-weighted averaging (Yokoyama et al., VLFM):
    v <- (c_old v_old + c v) / (c_old + c),   c <- (c_old^2 + c^2) / (c_old + c).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class ValueMap:
    def __init__(self, shape: Tuple[int, int], res: float, x0: float, y0: float, hfov_deg: float):
        self.value = np.zeros(shape, dtype=np.float32)
        self.conf = np.zeros(shape, dtype=np.float32)
        self.res, self.x0, self.y0 = res, x0, y0
        self.half_fov = math.radians(hfov_deg) / 2
        self.n_updates = 0

    def update(self, cells: np.ndarray, cam_xy: Tuple[float, float], cam_yaw: float, v: float) -> None:
        """``cells``: (N, 2) (row, col) observed in this frame."""
        if len(cells) == 0:
            return
        r, c = cells[:, 0], cells[:, 1]
        x = self.x0 + (c + 0.5) * self.res
        y = self.y0 + (r + 0.5) * self.res
        ang = np.arctan2(y - cam_xy[1], x - cam_xy[0]) - cam_yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        cn = np.cos(np.clip(np.abs(ang) / self.half_fov, 0.0, 1.0) * np.pi / 2) ** 2
        keep = cn > 1e-4
        r, c, cn = r[keep], c[keep], cn[keep].astype(np.float32)
        co, vo = self.conf[r, c], self.value[r, c]
        tot = co + cn
        self.value[r, c] = (co * vo + cn * np.float32(v)) / tot
        self.conf[r, c] = (co * co + cn * cn) / tot
        self.n_updates += 1

    def value_at(self, cell: Tuple[int, int], radius_cells: int = 0) -> float:
        if radius_cells <= 0:
            return float(self.value[cell])
        r, c = cell
        sl = (slice(max(0, r - radius_cells), r + radius_cells + 1), slice(max(0, c - radius_cells), c + radius_cells + 1))
        seen = self.conf[sl] > 0
        return float(self.value[sl][seen].max()) if seen.any() else 0.0

    def normalized(self) -> Optional[np.ndarray]:
        seen = self.conf > 0
        if not seen.any():
            return None
        lo, hi = float(self.value[seen].min()), float(self.value[seen].max())
        out = np.zeros_like(self.value)
        if hi > lo:
            out[seen] = (self.value[seen] - lo) / (hi - lo)
        return out
