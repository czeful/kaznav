"""Aggregate statistics: bootstrap CIs, Wilson CI, Cohen's d, McNemar, scene-clustered inference.

All comparisons in this project are *paired* (same episodes, same seeds for
every model), so the primary contrast statistics are the paired ones:
``paired_bootstrap_diff`` (CI of the mean difference), ``cohens_d_paired``
(d_z) and ``mcnemar_exact`` for binary success. ``cohens_d_independent`` is for
contrasts between disjoint episode sets (e.g. cultural vs universal words).

Episodes of one scene are not independent (same layout, same object instances),
and a benchmark has few scenes (HM3D v1 val: 20). The paper's primary inference
(D-047, docs/analysis_plan.md) therefore treats the scene as the unit:
``cluster_bootstrap_mean`` resamples scenes, ``cluster_sign_flip_test`` flips
the sign of whole scenes' paired differences (exact for up to
``exact_max_clusters`` scenes). Seeds are averaged per episode before either.
McNemar on pooled episode-seed pairs is reported as a secondary test.

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


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error over all tests of one table); NaNs stay NaN."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# ---------------------------------------------------------------------------- scene-clustered inference
def _clusters(values: Sequence[float], clusters: Sequence, name: str = "values"):
    x = _as_finite_1d(values, name)
    c = np.asarray(clusters).ravel()
    if c.shape != x.shape:
        raise ValueError(f"{name} and clusters differ in length: {x.size} vs {c.size}")
    keys, inv = np.unique(c.astype(str), return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=keys.size)
    counts = np.bincount(inv, minlength=keys.size).astype(float)
    return x, sums, counts


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence, *, n_boot: int = 10_000, level: float = 0.95,
                           seed: int = 0, chunk: int = 2_000) -> ConfidenceInterval:
    """Percentile CI of the mean over units, resampling whole clusters (scenes) with replacement.

    The resampled mean is sum / count over the drawn clusters' units (unequal cluster sizes are handled as a ratio
    estimator). With a single cluster the CI collapses to the point estimate - there is nothing to resample, and
    the caller must not report it as a CI (``n`` in the result is the number of clusters)."""
    x, sums, counts = _clusters(values, clusters)
    k = sums.size
    point = float(x.mean())
    if k < 2 or np.all(x == x[0]):
        return ConfidenceInterval(point, point, point, k, level, "scene-cluster-bootstrap")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, k, size=(b, k))
        stats[start : start + b] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return ConfidenceInterval(point, float(low), float(high), k, level, "scene-cluster-bootstrap")


@dataclass(frozen=True)
class PermutationResult:
    statistic: float          # mean paired difference over units
    p_value: float            # two-sided
    n_clusters: int
    n_permutations: int       # 2^k when exact
    method: str               # exact | monte-carlo


def cluster_sign_flip_test(diffs: Sequence[float], clusters: Sequence, *, n_perm: int = 20_000, seed: int = 0,
                           exact_max_clusters: int = 16) -> PermutationResult:
    """Two-sided sign-flip permutation test of mean(diffs) = 0 with the cluster (scene) as the exchangeable unit.

    Under the null the two arms are exchangeable within each episode, so flipping the sign of all paired
    differences of a scene at once leaves the null distribution unchanged, whatever the dependence between
    episodes of that scene. Exact over all 2^k sign vectors for k <= ``exact_max_clusters``; otherwise Monte
    Carlo with the observed assignment counted, p = (1 + #{|T*| >= |T|}) / (1 + n_perm). The smallest attainable
    p is 2 / 2^k: with few scenes no contrast can be significant, which is the honest answer."""
    x, sums, counts = _clusters(diffs, clusters, "diffs")
    k, n = sums.size, counts.sum()
    t_obs = float(sums.sum() / n)
    tol = 1e-12 * max(1.0, abs(t_obs))
    if np.all(sums == 0):
        return PermutationResult(t_obs, 1.0, k, 0, "exact")
    if k <= exact_max_clusters:
        signs = 1.0 - 2.0 * ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
        t = signs @ sums / n
        p = float(np.mean(np.abs(t) >= abs(t_obs) - tol))
        return PermutationResult(t_obs, min(1.0, p), k, 2 ** k, "exact")
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_perm, 4_096):
        b = min(4_096, n_perm - start)
        signs = rng.choice((-1.0, 1.0), size=(b, k))
        hits += int(np.sum(np.abs(signs @ sums / n) >= abs(t_obs) - tol))
    return PermutationResult(t_obs, (1 + hits) / (1 + n_perm), k, n_perm, "monte-carlo")
