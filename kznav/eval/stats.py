"""Aggregate statistics: bootstrap CIs, Wilson CI, Cohen's d, McNemar.

All comparisons in this project are *paired* (same episodes, same seeds for
every model), so the primary contrast statistics are the paired ones:
``paired_bootstrap_diff`` (CI of the mean difference), ``cohens_d_paired``
(d_z) and ``mcnemar_exact`` for binary success. ``cohens_d_independent`` is for
contrasts between disjoint episode sets (e.g. cultural vs universal words).

Non-finite inputs raise instead of being dropped silently; the caller decides
explicitly what to exclude and reports the exclusion count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    n: int
    level: float
    method: str

    def fmt(self, digits: int = 3) -> str:
        return f"{self.estimate:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def _as_finite_1d(values: Sequence[float], name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite entries; filter them explicitly and report how many")
    return arr


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], np.ndarray] = lambda x: x.mean(axis=-1),
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    chunk: int = 1_000,
) -> ConfidenceInterval:
    """Percentile bootstrap CI.

    ``statistic`` maps a (B, n) array of resamples to B values (vectorised);
    the default is the mean. Resampling is chunked to bound memory at full
    val-split sizes. Deterministic for a given ``seed``.
    """
    x = _as_finite_1d(values)
    n = x.size
    point = float(statistic(x[None, :])[0])
    if np.all(x == x[0]):
        return ConfidenceInterval(point, point, point, n, level, "percentile-bootstrap")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        stats[start : start + b] = statistic(x[idx])
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return ConfidenceInterval(point, float(low), float(high), n, level, "percentile-bootstrap")


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """CI of mean(a - b) resampling *episodes* (pairs), not the two arms independently."""
    xa, xb = _as_finite_1d(a, "a"), _as_finite_1d(b, "b")
    if xa.shape != xb.shape:
        raise ValueError(f"paired arrays differ in length: {xa.size} vs {xb.size}")
    return bootstrap_ci(xa - xb, n_boot=n_boot, level=level, seed=seed)


def wilson_ci(successes: int, n: int, level: float = 0.95) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = _z_for_level(level)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ConfidenceInterval(p, (centre - adj) / denom, (centre + adj) / denom, n, level, "wilson")


def _z_for_level(level: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + level / 2.0)


def cohens_d_independent(a: Sequence[float], b: Sequence[float], *, hedges: bool = False) -> float:
    """Standardised mean difference (mean(a) - mean(b)) / pooled SD.

    With ``hedges=True`` applies the small-sample correction J = 1 - 3 / (4 df - 1).
    Returns NaN when the pooled SD is zero.
    """
    xa, xb = _as_finite_1d(a, "a"), _as_finite_1d(b, "b")
    n1, n2 = xa.size, xb.size
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations")
    pooled = math.sqrt(((n1 - 1) * xa.var(ddof=1) + (n2 - 1) * xb.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0.0:
        return math.nan
    d = (xa.mean() - xb.mean()) / pooled
    if hedges:
        df = n1 + n2 - 2
        d *= 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """d_z = mean(a - b) / SD(a - b) for paired observations. NaN if SD(a - b) is zero."""
    xa, xb = _as_finite_1d(a, "a"), _as_finite_1d(b, "b")
    if xa.shape != xb.shape:
        raise ValueError(f"paired arrays differ in length: {xa.size} vs {xb.size}")
    diff = xa - xb
    if diff.size < 2:
        raise ValueError("need at least 2 pairs")
    sd = diff.std(ddof=1)
    if sd == 0.0:
        return math.nan
    return float(diff.mean() / sd)


@dataclass(frozen=True)
class McNemarResult:
    only_a: int
    only_b: int
    p_value: float


def mcnemar_exact(success_a: Sequence[bool], success_b: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar test on paired binary outcomes (binomial on discordant pairs)."""
    sa = np.asarray(success_a, dtype=bool)
    sb = np.asarray(success_b, dtype=bool)
    if sa.shape != sb.shape:
        raise ValueError("paired arrays differ in length")
    only_a = int(np.sum(sa & ~sb))
    only_b = int(np.sum(~sa & sb))
    n = only_a + only_b
    if n == 0:
        return McNemarResult(0, 0, 1.0)
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return McNemarResult(only_a, only_b, min(1.0, 2.0 * tail))


def mean_ci(values: Sequence[float], *, n_boot: int = 10_000, level: float = 0.95, seed: int = 0,
            binary: Optional[bool] = None) -> ConfidenceInterval:
    """Mean with a 95% CI: Wilson for 0/1 data (better at extreme rates), bootstrap otherwise."""
    x = _as_finite_1d(values)
    is_binary = bool(np.all((x == 0) | (x == 1))) if binary is None else binary
    if is_binary:
        return wilson_ci(int(x.sum()), int(x.size), level)
    return bootstrap_ci(x, n_boot=n_boot, level=level, seed=seed)
