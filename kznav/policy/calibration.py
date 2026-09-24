"""Calibrators for the verifier probability and the stop threshold.

Calibrators are fitted on the dev split ONLY and saved as JSON artifacts with
provenance (which backend, prompt, split and episodes). The runner refuses a
calibration artifact whose dev episodes overlap the evaluation episodes.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..eval.classification import threshold_metrics


def _logit_arr(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class Calibrator:
    kind = "base"

    def __call__(self, p: Sequence[float]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class IdentityCalibrator(Calibrator):
    kind = "identity"

    def __call__(self, p: Sequence[float]) -> np.ndarray:
        return np.asarray(p, dtype=float)

    def to_dict(self) -> dict:
        return {"kind": self.kind}


class PlattCalibrator(Calibrator):
    """sigmoid(a * logit(p) + b), fitted by Newton's method with Platt's smoothed targets."""

    kind = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = float(a), float(b)

    @classmethod
    def fit(cls, p: Sequence[float], y: Sequence[bool], max_iter: int = 100, tol: float = 1e-10) -> "PlattCalibrator":
        x = _logit_arr(np.asarray(p, dtype=float))
        yy = np.asarray(y, dtype=float)
        n_pos, n_neg = yy.sum(), (1 - yy).sum()
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Platt scaling needs both classes in the calibration set")
        t = np.where(yy == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        a, b = 1.0, 0.0
        for _ in range(max_iter):
            z = a * x + b
            q = 1 / (1 + np.exp(-z))
            g = np.array([np.sum((q - t) * x), np.sum(q - t)])
            wq = q * (1 - q)
            h = np.array([[np.sum(wq * x * x), np.sum(wq * x)], [np.sum(wq * x), np.sum(wq)]])
            h += 1e-12 * np.eye(2)
            step = np.linalg.solve(h, g)
            a, b = a - step[0], b - step[1]
            if np.max(np.abs(step)) < tol:
                break
        return cls(a, b)

    def __call__(self, p: Sequence[float]) -> np.ndarray:
        z = self.a * _logit_arr(np.asarray(p, dtype=float)) + self.b
        return 1 / (1 + np.exp(-z))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "a": self.a, "b": self.b}


class IsotonicCalibrator(Calibrator):
    """Monotone step function fitted with pool-adjacent-violators; linear interpolation between knots."""

    kind = "isotonic"

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    @classmethod
    def fit(cls, p: Sequence[float], y: Sequence[bool]) -> "IsotonicCalibrator":
        order = np.argsort(np.asarray(p, dtype=float), kind="mergesort")
        xs = np.asarray(p, dtype=float)[order]
        ys = np.asarray(y, dtype=float)[order]
        # Pool adjacent violators over blocks (value, weight, x-range).
        vals: List[float] = []
        wts: List[float] = []
        xlo: List[float] = []
        xhi: List[float] = []
        for xi, yi in zip(xs, ys):
            vals.append(yi); wts.append(1.0); xlo.append(xi); xhi.append(xi)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, lo2, hi2 = vals.pop(), wts.pop(), xlo.pop(), xhi.pop()
                v1, w1 = vals[-1], wts[-1]
                vals[-1] = (v1 * w1 + v2 * w2) / (w1 + w2)
                wts[-1] = w1 + w2
                xhi[-1] = hi2
        knots_x: List[float] = []
        knots_y: List[float] = []
        for v, lo, hi in zip(vals, xlo, xhi):
            knots_x.extend([lo, hi]); knots_y.extend([v, v])
        return cls(knots_x, knots_y)

    def __call__(self, p: Sequence[float]) -> np.ndarray:
        return np.interp(np.asarray(p, dtype=float), self.x, self.y)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x": self.x.tolist(), "y": self.y.tolist()}


def calibrator_from_dict(d: dict) -> Calibrator:
    kind = d["kind"]
    if kind == "identity":
        return IdentityCalibrator()
    if kind == "platt":
        return PlattCalibrator(d["a"], d["b"])
    if kind == "isotonic":
        return IsotonicCalibrator(d["x"], d["y"])
    raise ValueError(f"unknown calibrator kind {kind!r}")


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    objective: str
    precision: float
    recall: float
    f1: float
    n: int


def choose_threshold(
    p_calibrated: Sequence[float],
    y: Sequence[bool],
    *,
    objective: str = "max_f1",
    min_precision: Optional[float] = None,
) -> ThresholdChoice:
    """Pick the stop threshold on dev data.

    ``max_f1``: maximise F1. ``min_precision``: the lowest threshold whose
    precision >= ``min_precision`` (highest recall at bounded false stops).
    Candidate thresholds are the observed probabilities.
    """
    p = np.asarray(p_calibrated, dtype=float)
    labels = np.asarray(y, dtype=bool)
    cands = np.unique(p)
    best: Optional[ThresholdChoice] = None
    for thr in cands:
        m = threshold_metrics(p, labels, float(thr))
        prec, rec, f1 = m.precision, m.recall, m.f1
        if objective == "max_f1":
            score = -1.0 if math.isnan(f1) else f1
            if best is None or score > (best.f1 if not math.isnan(best.f1) else -1.0):
                best = ThresholdChoice(float(thr), objective, prec, rec, f1, int(p.size))
        elif objective == "min_precision":
            if min_precision is None:
                raise ValueError("min_precision objective needs min_precision")
            if not math.isnan(prec) and prec >= min_precision:
                if best is None or rec > best.recall:
                    best = ThresholdChoice(float(thr), objective, prec, rec, f1, int(p.size))
        else:
            raise ValueError(f"unknown objective {objective!r}")
    if best is None:
        raise ValueError("no threshold satisfies the objective on this data")
    return best


@dataclass
class CalibrationArtifact:
    """What gets saved after fitting on dev; loaded by the stop criterion at test time."""

    calibrator: dict
    threshold: dict
    backend: str
    model_id: str
    prompt_sha256: str
    dev_split: str
    dev_episode_ids: List[str]
    n_samples: int
    base_rate: float
    created_utc: str
    notes: str = ""
    extra: Dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibrationArtifact":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def assert_disjoint_from(self, eval_episode_ids: Sequence[str]) -> None:
        overlap = set(self.dev_episode_ids) & set(eval_episode_ids)
        if overlap:
            raise ValueError(
                f"calibration was fitted on {len(overlap)} of the evaluation episodes "
                f"(e.g. {sorted(overlap)[:3]}); thresholds must come from a disjoint dev split"
            )
