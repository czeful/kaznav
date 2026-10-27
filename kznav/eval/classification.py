"""Quality of the candidate-verification decision (binary classification)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    """Confusion counts and the derived rates of a binary decision.

    Precision, recall and F1 are NaN when undefined (no predicted positives, no positives), so that an empty case is
    never reported as 0 or 1.
    """
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float  # NaN when nothing was predicted positive
    recall: float     # NaN when there are no positives
    f1: float         # NaN when precision or recall is undefined
    false_positive_rate: float

    def as_dict(self) -> dict:
        """Return all fields as a plain dict (for tables and JSON)."""
        return asdict(self)


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def binary_metrics(predicted: Sequence[bool], labels: Sequence[bool]) -> BinaryMetrics:
    """Confusion counts, accuracy, precision, recall, F1 and false-positive rate of binary predictions.

    Args:
        predicted: Predicted labels.
        labels: Ground-truth labels, aligned with ``predicted``.

    Returns:
        The metrics.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    pr = np.asarray(predicted, dtype=bool).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if pr.shape != y.shape:
        raise ValueError("predicted and labels differ in length")
    if pr.size == 0:
        raise ValueError("empty input")
    tp = int(np.sum(pr & y))
    fp = int(np.sum(pr & ~y))
    tn = int(np.sum(~pr & ~y))
    fn = int(np.sum(~pr & y))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return BinaryMetrics(
        n=int(pr.size), tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=(tp + tn) / pr.size,
        precision=precision, recall=recall, f1=f1,
        false_positive_rate=_ratio(fp, fp + tn),
    )


def threshold_metrics(probs: Sequence[float], labels: Sequence[bool], threshold: float) -> BinaryMetrics:
    """Binary metrics of the decision ``probs >= threshold`` against ``labels``."""
    p = np.asarray(probs, dtype=float)
    return binary_metrics(p >= threshold, labels)
