"""Calibration of probabilistic decisions: ECE, MCE, Brier, reliability diagrams.

Two ECE variants exist in the literature and they are NOT interchangeable:

* ``ece_binary`` — bins on P(positive) over [0, 1] and compares the mean
  prediction with the observed positive rate per bin. This is the one that
  matters for a stop threshold on P(target), and the default in reports.
* ``ece_top_label`` — bins on the confidence of the predicted class
  (max(p, 1 - p) in [0.5, 1]). This is what the Qolda pilot reported
  (``qolda check/qolda_pilot/metrics.py``); kept to reproduce those numbers.

Binning convention (both): equal-width bins, bin i covers (lo, hi], the first
bin also includes its left edge — identical to the pilot implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _prep(probs: Sequence[float], labels: Sequence[bool]) -> "tuple[np.ndarray, np.ndarray]":
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("empty input")
    if p.shape != y.shape:
        raise ValueError(f"probs and labels differ in length: {p.size} vs {y.size}")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary")
    return p, y


def _bin_index(values: np.ndarray, n_bins: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    edges = np.linspace(lo, hi, n_bins + 1)
    # (lo, hi] per bin; values equal to the global left edge go to bin 0.
    idx = np.searchsorted(edges, values, side="left") - 1
    return np.clip(idx, 0, n_bins - 1)


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Per-bin counts, mean predicted probability and observed positive rate of a reliability diagram.

    Attributes:
        edges: Bin edges over [0, 1], length ``n_bins + 1``.
        counts: Samples per bin.
        mean_pred: Mean predicted probability per bin (NaN for empty bins).
        frac_pos: Fraction of positive labels per bin (NaN for empty bins).
    """
    edges: np.ndarray
    counts: np.ndarray
    mean_pred: np.ndarray  # NaN for empty bins
    frac_pos: np.ndarray   # NaN for empty bins

    def rows(self) -> "list[dict]":
        """Return the diagram as one dict per bin (bin edges, count, mean prediction, positive rate)."""
        out = []
        for i in range(len(self.counts)):
            out.append(
                {
                    "bin_lo": float(self.edges[i]),
                    "bin_hi": float(self.edges[i + 1]),
                    "n": int(self.counts[i]),
                    "mean_pred": float(self.mean_pred[i]),
                    "frac_pos": float(self.frac_pos[i]),
                }
            )
        return out


def reliability_diagram(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10) -> ReliabilityDiagram:
    """Bin predicted probabilities into equal-width bins and compare them with the observed outcomes.

    Args:
        probs: Predicted probabilities in [0, 1].
        labels: Binary ground-truth labels, aligned with ``probs``.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        The reliability diagram.

    Raises:
        ValueError: If the probabilities are not finite and within [0, 1] or the labels are not binary.
    """
    p, y = _prep(probs, labels)
    idx = _bin_index(p, n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.bincount(idx, weights=p, minlength=n_bins) / counts
        frac_pos = np.bincount(idx, weights=y, minlength=n_bins) / counts
    return ReliabilityDiagram(np.linspace(0, 1, n_bins + 1), counts, mean_pred, frac_pos)


def ece_binary(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error of P(positive): the count-weighted mean |observed rate - mean prediction| over bins.

    Args:
        probs: Predicted probabilities of the positive class.
        labels: Binary ground-truth labels.
        n_bins: Number of equal-width bins.

    Returns:
        The ECE in [0, 1].
    """
    rd = reliability_diagram(probs, labels, n_bins)
    mask = rd.counts > 0
    gaps = np.abs(rd.frac_pos[mask] - rd.mean_pred[mask])
    return float(np.sum(rd.counts[mask] / rd.counts.sum() * gaps))


def mce_binary(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10) -> float:
    """Maximum calibration error over non-empty bins."""
    rd = reliability_diagram(probs, labels, n_bins)
    mask = rd.counts > 0
    return float(np.max(np.abs(rd.frac_pos[mask] - rd.mean_pred[mask])))


def ece_top_label(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10, threshold: float = 0.5) -> float:
    """Top-label ECE as computed in the pilot: confidence of the predicted class, bins over [0, 1]."""
    p, y = _prep(probs, labels)
    pred = (p >= threshold).astype(float)
    correct = (pred == y).astype(float)
    conf = np.where(pred == 1.0, p, 1.0 - p)
    idx = _bin_index(conf, n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    mask = counts > 0
    acc = np.bincount(idx, weights=correct, minlength=n_bins)[mask] / counts[mask]
    mconf = np.bincount(idx, weights=conf, minlength=n_bins)[mask] / counts[mask]
    return float(np.sum(counts[mask] / p.size * np.abs(acc - mconf)))


def brier_score(probs: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared difference between predicted probabilities and binary labels (lower is better)."""
    p, y = _prep(probs, labels)
    return float(np.mean((p - y) ** 2))
