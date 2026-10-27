"""Bayesian bookkeeping for goal hypotheses and for individual candidates.

Model
-----
The goal parser returns K English visual descriptions h_1..h_K of one target
concept, with prior weights w_k. Exactly one description is assumed to be the
"right" way to look for the target.

A verification event concerns one candidate c. It carries
  * m_k in [0, 1] — how strongly the detector associated c with h_k
    (e.g. the detection score for query h_k; 0 if h_k did not fire), and
  * q = P(c is the target | crop) from the verifier, already calibrated, with
    base rate pi (the positive rate of the calibration set).

If h_k is the right description, a candidate matched to it with strength m_k
is the target with probability rho_k = rho_wrong + (rho_right - rho_wrong) * m_k.
The crop enters as virtual evidence with likelihood ratio
lambda = odds(q) / odds(pi) for "target" vs "not target", so

    P(crop | h_k) ∝ rho_k * lambda + (1 - rho_k)

and w_k ← w_k * P(crop | h_k)^beta, renormalised. Properties (tested):
q = pi gives no update; equal m_k for all k gives no update; updates commute
(without the floor); ``beta`` < 1 tempers overconfident verifiers; a weight
floor keeps one confident mistake from permanently eliminating a hypothesis —
the pilot showed the verifier is confidently wrong on cultural items.

Candidate evidence accumulates in log-odds around the same base rate:
logit P = logit(pi) + gamma * sum_i (logit(q_i) - logit(pi)); gamma < 1
discounts correlated views of the same object. With ``first_view_full`` (D-069) the first view counts in full and
gamma applies to the repeated views only: one view gives P = q_1, the quantity the stop threshold was fitted on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


def logit(p: float, eps: float = 1e-6) -> float:
    """Log-odds of ``p``, with ``p`` clipped to [eps, 1 - eps] so that 0 and 1 stay finite."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Logistic function, computed without overflow for large negative inputs."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class BeliefParams:
    """Parameters of the hypothesis belief and of candidate evidence (``policy.belief``).

    Attributes:
        rho_right: P(candidate is the target | fully matched to the right hypothesis).
        rho_wrong: P(candidate is the target | matched to a wrong hypothesis).
        base_rate: Positive rate of the verifier's calibration set (pi).
        evidence_power: Exponent beta in (0, 1] that tempers each verification's likelihood.
        weight_floor: Minimum normalised weight per hypothesis.
        view_discount: Discount gamma in (0, 1] for repeated views of one candidate.

    Raises:
        ValueError: If a parameter is outside its range.
    """
    rho_right: float      # P(candidate is target | fully matched to the right hypothesis)
    rho_wrong: float      # P(candidate is target | matched to a wrong hypothesis)
    base_rate: float      # verifier calibration-set positive rate (pi)
    evidence_power: float = 1.0   # beta in (0, 1]
    weight_floor: float = 1e-3    # minimum normalised weight per hypothesis
    view_discount: float = 1.0    # gamma in (0, 1] for repeated views of a candidate

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho_wrong < self.rho_right <= 1.0:
            raise ValueError("need 0 <= rho_wrong < rho_right <= 1")
        if not 0.0 < self.base_rate < 1.0:
            raise ValueError("base_rate must be in (0, 1)")
        if not 0.0 < self.evidence_power <= 1.0:
            raise ValueError("evidence_power must be in (0, 1]")
        if not 0.0 < self.view_discount <= 1.0:
            raise ValueError("view_discount must be in (0, 1]")


class HypothesisBelief:
    """Posterior weights over the K visual hypotheses of one episode goal."""

    def __init__(self, names: Sequence[str], prior_weights: Sequence[float], params: BeliefParams):
        w = np.asarray(prior_weights, dtype=float)
        if len(names) != w.size or w.size == 0:
            raise ValueError("names and prior_weights must be non-empty and equal length")
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("prior weights must be finite and non-negative")
        if params.weight_floor * w.size >= 1.0:
            raise ValueError("weight_floor too large for this many hypotheses")
        w = np.full(w.size, 1.0 / w.size) if w.sum() == 0 else w / w.sum()
        self.names = tuple(names)
        self.params = params
        self._w = self._apply_floor(w)
        self.n_updates = 0

    @property
    def weights(self) -> np.ndarray:
        """Current normalised hypothesis weights (a copy)."""
        return self._w.copy()

    def _apply_floor(self, w: np.ndarray) -> np.ndarray:
        """Clamp weights below the floor to the floor and rescale the rest so the sum stays 1."""
        f = self.params.weight_floor
        w0 = w / w.sum()
        if f <= 0:
            return w0
        clamp = w0 < f
        while True:
            # Terminates with a non-empty unclamped set because f * K < 1.
            scale = (1.0 - f * clamp.sum()) / w0[~clamp].sum()
            out = np.where(clamp, f, w0 * scale)
            new_clamp = clamp | (out < f)
            if np.array_equal(new_clamp, clamp):
                return out
            clamp = new_clamp

    def likelihoods(self, match: Sequence[float], p_target: float) -> np.ndarray:
        """Likelihood of one verification answer under each hypothesis, ``rho_k * lambda + (1 - rho_k)``.

        Args:
            match: Per-hypothesis association strengths ``m_k`` in [0, 1] of the verified candidate.
            p_target: Calibrated verifier probability that the crop shows the target.

        Returns:
            One likelihood per hypothesis (unnormalised).

        Raises:
            ValueError: If ``match`` does not have one entry per hypothesis.
        """
        m = np.clip(np.asarray(match, dtype=float), 0.0, 1.0)
        if m.size != len(self.names):
            raise ValueError("match vector length differs from number of hypotheses")
        p = self.params
        lam = math.exp(logit(p_target) - logit(p.base_rate))
        rho = p.rho_wrong + (p.rho_right - p.rho_wrong) * m
        return rho * lam + (1.0 - rho)

    def update(self, match: Sequence[float], p_target: float) -> np.ndarray:
        """Apply one verification event; returns the new weights."""
        like = self.likelihoods(match, p_target)
        logw = np.log(self._w) + self.params.evidence_power * np.log(like)
        logw -= logw.max()
        self._w = self._apply_floor(np.exp(logw))
        self.n_updates += 1
        return self.weights


@dataclass
class CandidateEvidence:
    """Accumulated P(candidate is the target) over repeated verifications."""

    base_rate: float
    view_discount: float = 1.0
    observations: List[float] = field(default_factory=list)
    first_view_full: bool = False      # D-069: the discount applies to repeated views, not to the first one

    def add(self, p_target: float) -> float:
        """Record one calibrated verifier answer and return the updated P(candidate is the target)."""
        self.observations.append(float(p_target))
        return self.probability()

    def probability(self) -> float:
        """P(candidate is the target) from all answers so far; the base rate before the first answer."""
        if not self.observations:
            return self.base_rate
        prior = logit(self.base_rate)
        diffs = [logit(q) - prior for q in self.observations]
        if self.first_view_full:
            total = prior + diffs[0] + self.view_discount * sum(diffs[1:])
        else:
            total = prior + self.view_discount * sum(diffs)
        return sigmoid(total)
