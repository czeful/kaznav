"""Frontier extraction: known-free cells bordering unknown space, grouped into regions.

Only frontiers the robot can reach are returned: the region connected to the robot's cell over
cells that are not (inflated) obstacles. Each region is represented by the member cell closest
to the region's centroid, so the target itself lies on the frontier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Frontier:
    cell: Tuple[int, int]        # representative (row, col)
    size: int                    # number of frontier cells in the region
    centroid: Tuple[float, float]


def reachable_mask(blocked: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    labels, _ = ndimage.label(~blocked, structure=_EIGHT)
    lab = labels[start]
    return labels == lab if lab > 0 else np.zeros_like(blocked)


def find_frontiers(free: np.ndarray, unknown: np.ndarray, blocked: np.ndarray, start: Tuple[int, int],
                   min_size: int) -> List[Frontier]:
    """Frontier regions reachable from ``start``; ``blocked`` = inflated obstacles."""
    edge = free & ndimage.binary_dilation(unknown, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    edge &= reachable_mask(blocked, start)
    labels, n = ndimage.label(edge, structure=_EIGHT)
    out: List[Frontier] = []
    for lab in range(1, n + 1):
        rr, cc = np.nonzero(labels == lab)
        if len(rr) < min_size:
            continue
        cy, cx = rr.mean(), cc.mean()
        i = int(np.argmin((rr - cy) ** 2 + (cc - cx) ** 2))
        out.append(Frontier((int(rr[i]), int(cc[i])), int(len(rr)), (float(cy), float(cx))))
    return out
