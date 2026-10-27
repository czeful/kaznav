"""Calibrators for the verifier probability and the stop threshold.

Calibrators are fitted on the dev split ONLY and saved as JSON artifacts with
provenance (which backend, prompt, split and episodes). The runner refuses a
calibration artifact whose dev episodes overlap the evaluation episodes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import expit

from ..eval.classification import threshold_metrics


def _logit_arr(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw verifier probabilities to calibrated ones; serialisable to a dict."""
    kind = "base"

    def __call__(self, p: Sequence[float]) -> np.ndarray:  # pragma: no cover - interface
        """Calibrate a sequence of raw probabilities."""
        raise NotImplementedError

    def to_dict(self) -> dict:  # pragma: no cover - interface
        """Serialise the fitted calibrator."""
        raise NotImplementedError


class IdentityCalibrator(Calibrator):
    """No calibration: returns the raw probabilities (debug runs without an artifact)."""
    kind = "identity"

    def __call__(self, p: Sequence[float]) -> np.ndarray:
        """Return the raw probabilities unchanged."""
        return np.asarray(p, dtype=float)

    def to_dict(self) -> dict:
        """Serialise as ``{"kind": "identity"}``."""
        return {"kind": self.kind}


class PlattCalibrator(Calibrator):
    """sigmoid(a * logit(p) + b), fitted by Newton's method with Platt's smoothed targets."""

    kind = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = float(a), float(b)

    @classmethod
    def fit(cls, p: Sequence[float], y: Sequence[bool], max_iter: int = 100, tol: float = 1e-10) -> "PlattCalibrator":
        """Newton's method with backtracking on the smoothed-target log loss (Lin, Lin & Weng 2007). Without the
        line search a verifier that only says ~0 or ~1 (Qwen on the pilot crops) sends the plain Newton step to
        a ~ -1e12."""
        x = _logit_arr(np.asarray(p, dtype=float))
        yy = np.asarray(y, dtype=float)
        n_pos, n_neg = yy.sum(), (1 - yy).sum()
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Platt scaling needs both classes in the calibration set")
        t = np.where(yy == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

        def loss(a: float, b: float) -> float:
            z = a * x + b                         # t*log(1+e^-z) + (1-t)*log(1+e^z), overflow-free
            return float(np.sum(t * np.logaddexp(0.0, -z) + (1 - t) * np.logaddexp(0.0, z)))

        a, b = 1.0, 0.0
        f = loss(a, b)
        for _ in range(max_iter):
            q = expit(a * x + b)
            g = np.array([np.sum((q - t) * x), np.sum(q - t)])
            wq = q * (1 - q)
            h = np.array([[np.sum(wq * x * x), np.sum(wq * x)], [np.sum(wq * x), np.sum(wq)]]) + 1e-12 * np.eye(2)
            step = np.linalg.solve(h, g)
            lr = 1.0
            while lr > 1e-12:
                na, nb = a - lr * step[0], b - lr * step[1]
                nf = loss(na, nb)
                if nf <= f + 1e-4 * lr * float(g @ -step):
                    break
                lr /= 2
            moved = max(abs(na - a), abs(nb - b))
            a, b, f = na, nb, nf
            if moved < tol:
                break
        return cls(a, b)

    def __call__(self, p: Sequence[float]) -> np.ndarray:
        """Apply ``sigmoid(a * logit(p) + b)``."""
        return expit(self.a * _logit_arr(np.asarray(p, dtype=float)) + self.b)

    def to_dict(self) -> dict:
        """Serialise the fitted slope ``a`` and intercept ``b``."""
        return {"kind": self.kind, "a": self.a, "b": self.b}


class IsotonicCalibrator(Calibrator):
    """Monotone step function fitted with pool-adjacent-violators; linear interpolation between knots."""

    kind = "isotonic"

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    @classmethod
    def fit(cls, p: Sequence[float], y: Sequence[bool]) -> "IsotonicCalibrator":
        """Fit a non-decreasing step function by pool-adjacent-violators.

        Args:
            p: Raw probabilities.
            y: Binary labels.

        Returns:
            The calibrator with two knots (left and right edge) per pooled block.
        """
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
        """Interpolate the fitted step function linearly between knots."""
        return np.interp(np.asarray(p, dtype=float), self.x, self.y)

    def to_dict(self) -> dict:
        """Serialise the knots."""
        return {"kind": self.kind, "x": self.x.tolist(), "y": self.y.tolist()}


def calibrator_from_dict(d: dict) -> Calibrator:
    """Rebuild a calibrator from its serialised form.

    Raises:
        ValueError: If the kind is unknown.
    """
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
    """The chosen stop threshold and its precision, recall and F1 on the fitting data."""
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
        """Write the artifact as JSON (parent directories are created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibrationArtifact":
        """Read an artifact written by ``save``."""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def assert_disjoint_from(self, eval_episode_ids: Sequence[str]) -> None:
        """Refuse evaluation on episodes the calibration was fitted on.

        Raises:
            ValueError: If any evaluation episode id is among the artifact's dev episode ids.
        """
        overlap = set(self.dev_episode_ids) & set(eval_episode_ids)
        if overlap:
            raise ValueError(
                f"calibration was fitted on {len(overlap)} of the evaluation episodes "
                f"(e.g. {sorted(overlap)[:3]}); thresholds must come from a disjoint dev split"
            )


# ---------------------------------------------------------------------------- fitting the stop rule (D-042)
FITTERS = {"identity": lambda p, y: IdentityCalibrator(), "platt": PlattCalibrator.fit, "isotonic": IsotonicCalibrator.fit}


@dataclass
class StopRuleFit:
    """A fitted stop rule: the calibrator, the threshold, the base rate and the fit diagnostics."""
    calibrator: Calibrator
    threshold: ThresholdChoice
    base_rate: float
    metrics: Dict[str, float]


def _fold_of(group: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(group.encode("utf-8")).digest()[:4], "big") % folds


def fit_stop_rule(p_raw: Sequence[float], y: Sequence[bool], groups: Sequence[str], *, method: str = "platt",
                  objective: str = "max_f1", min_precision: Optional[float] = None, n_bins: int = 10,
                  folds: int = 5) -> StopRuleFit:
    """Calibrator + stop threshold on dev samples (one per verifier call; ``groups`` = episode ids).

    Fitting and judging on the same calls is optimistic, so ``metrics`` also holds out-of-fold numbers: folds
    split by episode (calls of one episode are correlated), each fold calibrated and thresholded by a rule
    fitted on the other folds. Folds whose training part lacks a class are skipped and counted."""
    from ..eval.calibration import brier_score, ece_binary

    p = np.asarray(p_raw, dtype=float)
    yy = np.asarray(y, dtype=bool)
    g = np.asarray([_fold_of(str(s), folds) for s in groups])
    if method not in FITTERS:
        raise ValueError(f"unknown calibration method {method!r}; one of {sorted(FITTERS)}")
    cal = FITTERS[method](p, yy)
    pc = cal(p)
    thr = choose_threshold(pc, yy, objective=objective, min_precision=min_precision)
    oof = np.full(p.shape, np.nan)
    oof_stop = np.zeros(p.shape, dtype=bool)
    skipped = 0
    for k in range(folds):
        test, train = g == k, g != k
        if not test.any():
            continue
        if yy[train].all() or not yy[train].any():
            skipped += 1
            continue
        c = FITTERS[method](p[train], yy[train])
        t = choose_threshold(c(p[train]), yy[train], objective=objective, min_precision=min_precision)
        oof[test] = c(p[test])
        oof_stop[test] = oof[test] >= t.threshold
    have = ~np.isnan(oof)
    m_oof = threshold_metrics(oof_stop[have].astype(float), yy[have], 0.5) if have.any() else None
    metrics = {
        "n": int(p.size), "n_pos": int(yy.sum()), "ece_raw": ece_binary(p, yy, n_bins), "brier_raw": brier_score(p, yy),
        "ece_fit": ece_binary(pc, yy, n_bins), "brier_fit": brier_score(pc, yy),
        "oof_n": int(have.sum()), "oof_folds_skipped": skipped,
        "oof_ece": ece_binary(oof[have], yy[have], n_bins) if have.any() else float("nan"),
        "oof_precision": m_oof.precision if m_oof else float("nan"),
        "oof_recall": m_oof.recall if m_oof else float("nan"),
    }
    return StopRuleFit(cal, thr, float(yy.mean()), metrics)
