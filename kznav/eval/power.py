"""Sample size for paired success-rate contrasts (D-049): analytic McNemar and Monte-Carlo power of the
analysis plan's primary test (scene sign-flip permutation on seed-averaged episode differences, D-047).

Generative model of one contrast (arm b against arm a on the same episodes), parameterised by quantities a pilot
or dev run can estimate:

* ``k_scenes`` scenes with ``n`` episodes split evenly; scene k has its own success rate
  p_k = p_a + sigma_scene * z_k and its own effect delta_k = delta + tau * z'_k (z ~ N(0, 1), clipped so that the
  cell probabilities stay valid);
* each episode's pair (A, B) is drawn from the 2x2 table with marginals p_k, p_k + delta_k and discordance
  ``psi`` = P(A != B): P(1, 0) = (psi - delta_k) / 2, P(0, 1) = (psi + delta_k) / 2;
* every seed repeats the episode's outcome, flipped with probability ``seed_flip`` (seed noise; our pipeline is
  nearly deterministic per episode, D-036), and seeds are averaged per episode.

The test is two-sided at ``alpha / m`` (Bonferroni for ``m`` primary contrasts: Holm is at least as powerful, so
the result is a lower bound on power under Holm).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PowerParams:
    p_a: float                   # success rate of the reference arm
    delta: float                 # p_b - p_a (absolute, e.g. 0.10 = 10 points)
    psi: float                   # discordance P(A != B); must be >= |delta|
    n: int                       # episodes (pairs) per arm
    k_scenes: int = 20           # HM3D ObjectNav v1 val: 20 scenes
    seeds: int = 1
    seed_flip: float = 0.0
    sigma_scene: float = 0.10    # SD of scene success rates
    tau: float = 0.05            # SD of the effect across scenes
    alpha: float = 0.05
    m: int = 1                   # number of primary contrasts (Bonferroni bound for Holm)

    def __post_init__(self) -> None:
        if not 0 < self.p_a < 1 or not 0 < self.p_a + self.delta < 1:
            raise ValueError("p_a and p_a + delta must be in (0, 1)")
        if self.psi < abs(self.delta) or self.psi > 1:
            raise ValueError(f"discordance psi={self.psi} must be in [|delta|, 1]")
        if self.n < self.k_scenes:
            raise ValueError("need at least one episode per scene")


def mcnemar_n(p_a: float, delta: float, psi: float, *, alpha: float = 0.05, power: float = 0.8, m: int = 1,
              design_effect: float = 1.0) -> int:
    """Pairs needed by the two-sided McNemar test (Connor 1987 normal approximation) at alpha / m, inflated by a
    design effect for clustering (1 + (cluster size - 1) * ICC of the paired differences)."""
    if psi < abs(delta) or delta == 0:
        raise ValueError("need delta != 0 and psi >= |delta|")
    z_a = NormalDist().inv_cdf(1 - alpha / (2 * m))
    z_b = NormalDist().inv_cdf(power)
    n = (z_a * math.sqrt(psi) + z_b * math.sqrt(psi - delta * delta)) ** 2 / (delta * delta)
    return int(math.ceil(n * design_effect))


def _cells(p: np.ndarray, d: np.ndarray, psi: float) -> np.ndarray:
    """(…, 4) probabilities of (1,1), (1,0), (0,1), (0,0); clipped to a valid table."""
    p10 = np.clip((psi - d) / 2, 0, None)
    p01 = np.clip((psi + d) / 2, 0, None)
    p11 = np.clip(p - p10, 0, None)
    p00 = np.clip(1 - p11 - p10 - p01, 0, None)
    c = np.stack([p11, p10, p01, p00], axis=-1)
    return c / c.sum(axis=-1, keepdims=True)


def simulate_pairs(pp: PowerParams, rng: np.random.Generator, n_sims: int) -> np.ndarray:
    """(n_sims, n) seed-averaged differences B - A (episode order: scene-major; scene = index // per_scene)."""
    k, n = pp.k_scenes, pp.n
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    scene_of = np.repeat(np.arange(k), sizes)
    p_k = np.clip(pp.p_a + pp.sigma_scene * rng.standard_normal((n_sims, k)), 0.02, 0.98)
    d_k = pp.delta + pp.tau * rng.standard_normal((n_sims, k))
    d_k = np.clip(d_k, -p_k + 0.01, 0.99 - p_k)
    cells = _cells(p_k, d_k, pp.psi)[:, scene_of, :]                     # (S, n, 4)
    u = rng.random((n_sims, n, 1))
    idx = (u > np.cumsum(cells, axis=-1)).sum(axis=-1)                  # 0..3
    a = np.isin(idx, (0, 1)).astype(float)
    b = np.isin(idx, (0, 2)).astype(float)
    if pp.seeds > 1 or pp.seed_flip > 0:
        flip_a = rng.random((n_sims, n, pp.seeds)) < pp.seed_flip
        flip_b = rng.random((n_sims, n, pp.seeds)) < pp.seed_flip
        a = np.abs(a[..., None] - flip_a).mean(axis=-1)
        b = np.abs(b[..., None] - flip_b).mean(axis=-1)
    return b - a


def sign_flip_pvalues(diffs: np.ndarray, scene_of: np.ndarray, k: int, rng: np.random.Generator,
                      n_perm: int = 4000) -> np.ndarray:
    """Vectorised scene sign-flip p-values for many simulated data sets (rows of ``diffs``)."""
    sums = np.stack([np.bincount(scene_of, weights=row, minlength=k) for row in diffs])     # (S, k)
    t_obs = np.abs(sums.sum(axis=1))
    signs = rng.choice((-1.0, 1.0), size=(n_perm, k))
    t_perm = np.abs(sums @ signs.T)                                                          # (S, n_perm)
    hits = (t_perm >= t_obs[:, None] - 1e-9).sum(axis=1)
    return (1 + hits) / (1 + n_perm)


def simulate_power(pp: PowerParams, *, n_sims: int = 1000, n_perm: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo power of the primary test, and of McNemar on the same simulated data for comparison."""
    rng = np.random.default_rng(seed)
    d = simulate_pairs(pp, rng, n_sims)
    sizes = np.full(pp.k_scenes, pp.n // pp.k_scenes)
    sizes[: pp.n % pp.k_scenes] += 1
    scene_of = np.repeat(np.arange(pp.k_scenes), sizes)
    p = sign_flip_pvalues(d, scene_of, pp.k_scenes, rng, n_perm)
    level = pp.alpha / pp.m
    # McNemar (exact, pooled seed-0 outcomes are not kept; use the sign of the episode difference)
    from .stats import mcnemar_exact

    mc = []
    for row in d:
        b_only, a_only = int((row > 0).sum()), int((row < 0).sum())
        mc.append(mcnemar_exact([True] * a_only + [False] * b_only, [False] * a_only + [True] * b_only).p_value)
    se = math.sqrt(max(1e-12, np.mean(p <= level) * (1 - np.mean(p <= level)) / n_sims))
    return {"power": float(np.mean(p <= level)), "power_se": se, "power_mcnemar": float(np.mean(np.array(mc) <= level)),
            "mean_diff": float(d.mean()), "level": level}


def required_n(pp: PowerParams, target: float = 0.8, *, n_grid=None, n_sims: int = 600, n_perm: int = 3000,
               seed: int = 0) -> Optional[int]:
    """Smallest n on the grid whose simulated power reaches ``target`` (None if none does)."""
    grid = n_grid or [100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000]
    for n in grid:
        if n < pp.k_scenes:
            continue
        r = simulate_power(PowerParams(**{**asdict(pp), "n": n}), n_sims=n_sims, n_perm=n_perm, seed=seed)
        if r["power"] >= target:
            return n
    return None


def estimate_from_pairs(a: np.ndarray, b: np.ndarray, scenes: np.ndarray) -> Dict[str, float]:
    """Pilot estimates for the model from paired 0/1 outcomes (one seed): p_a, delta, psi, and - with >= 5 scenes -
    sigma_scene (SD of scene SRs of arm a, sampling noise removed) and tau (same for scene differences)."""
    a, b, scenes = np.asarray(a, float), np.asarray(b, float), np.asarray(scenes).astype(str)
    out = {"n": int(a.size), "p_a": float(a.mean()), "delta": float((b - a).mean()), "psi": float((a != b).mean()),
           "k_scenes": int(len(set(scenes)))}
    if out["k_scenes"] >= 5:
        keys = sorted(set(scenes))
        sr = np.array([a[scenes == s].mean() for s in keys])
        dd = np.array([(b - a)[scenes == s].mean() for s in keys])
        m = np.array([(scenes == s).sum() for s in keys], float)
        noise_sr = np.mean(sr * (1 - sr) / m)
        noise_d = np.mean([np.var((b - a)[scenes == s], ddof=0) / (scenes == s).sum() for s in keys])
        out["sigma_scene"] = float(math.sqrt(max(0.0, sr.var(ddof=1) - noise_sr)))
        out["tau"] = float(math.sqrt(max(0.0, dd.var(ddof=1) - noise_d)))
    return out
