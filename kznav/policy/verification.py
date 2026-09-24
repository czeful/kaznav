"""Verify-before-stop: the VLM checks a crop of the candidate the robot is about to stop at.

When the policy is within stop range of its best candidate and sees it in the current frame, it
asks ``Verifier.check``. A call is made for the first view and again only when the crop grew by
``regrow_factor`` (a closer, more informative view), at most ``max_calls`` times per candidate.
Each answer ``q = P(yes)`` (first-token logprobs, calibrated on dev) is

* accumulated per candidate (``CandidateEvidence``: log-odds around the calibration base rate,
  correlated views discounted), and
* applied to the hypothesis weights (``HypothesisBelief``, with the candidate's per-query detector
  scores as the match vector), which then weight the value map and the candidate ranking.

The candidate is accepted (stop) when its probability reaches the stop threshold, rejected when it
falls below ``reject_below`` or the call budget is used up, and otherwise the robot keeps
approaching for a closer view; with no new view for ``max_wait_steps`` steps it is rejected.
Rejected candidates are never approached again in the episode.

Without a calibration artifact (dev split) the verifier runs with the identity calibrator and a
DEBUG threshold; runs made that way are debug-only (their manifest says so).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..language.base import BackendError
from ..language.schemas import VerifyQuery
from .belief import BeliefParams, CandidateEvidence, HypothesisBelief
from .calibration import CalibrationArtifact, Calibrator, IdentityCalibrator, calibrator_from_dict


@dataclass(frozen=True)
class VerifyParams:
    stop_threshold: float
    reject_below: float
    base_rate: float
    max_calls: int
    regrow_factor: float
    max_wait_steps: int
    crop_padding: float
    min_crop_px: int
    calibrated: bool            # False: identity calibrator + debug threshold (debug-only runs)


@dataclass
class VerifyEvent:
    step: int
    cid: int
    p_raw: Optional[float]
    p_cal: Optional[float]
    p_candidate: float
    decision: str               # stop | reject | wait
    source: str                 # logprobs | unavailable | error | none (no call this step)
    answer: str
    latency_s: float
    crop_area_px: float
    weights: List[float]
    called: bool


@dataclass
class _CandState:
    evidence: CandidateEvidence
    calls: int = 0
    last_area: float = 0.0
    wait: int = 0


def crop_box(rgb: np.ndarray, box: Sequence[float], padding: float, min_px: int) -> Tuple[np.ndarray, float]:
    """Crop around a detection box with ``padding`` (fraction of the box size) of context per side."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = max(bw * (1 + 2 * padding), min_px) / 2
    hh = max(bh * (1 + 2 * padding), min_px) / 2
    a, b = int(max(0, np.floor(cx - hw))), int(min(w, np.ceil(cx + hw)))
    c, d = int(max(0, np.floor(cy - hh))), int(min(h, np.ceil(cy + hh)))
    return np.ascontiguousarray(rgb[c:d, a:b]), bw * bh


class Verifier:
    def __init__(self, backend: Any, query: VerifyQuery, params: VerifyParams, belief: Optional[HypothesisBelief],
                 calibrator: Optional[Calibrator] = None, view_discount: float = 1.0):
        self.backend = backend
        self.query = query
        self.p = params
        self.belief = belief
        self.calibrator = calibrator or IdentityCalibrator()
        self.view_discount = view_discount
        self.events: List[VerifyEvent] = []
        self._state: Dict[int, _CandState] = {}

    def probability(self, cid: int) -> Optional[float]:
        st = self._state.get(cid)
        return st.evidence.probability() if st is not None and st.calls else None

    def check(self, cand: Any, rgb: np.ndarray, step: int, n_queries: int) -> VerifyEvent:
        """One stop-range decision about ``cand`` (seen in this frame's ``rgb`` at ``cand.last_box``)."""
        st = self._state.setdefault(cand.cid, _CandState(CandidateEvidence(self.p.base_rate, self.view_discount)))
        crop, area = crop_box(rgb, cand.last_box, self.p.crop_padding, self.p.min_crop_px)
        can_call = st.calls < self.p.max_calls and (st.calls == 0 or area >= self.p.regrow_factor * st.last_area)
        p_raw = p_cal = None
        source, answer, latency = "none", "", 0.0
        if can_call:
            t0 = time.perf_counter()
            try:
                r = self.backend.verify_candidate(crop, self.query)
                p_raw, source, answer = r.p_target, r.source, r.answer_text
            except BackendError as e:          # the call is logged by the backend's sink
                source, answer = "error", str(e)[:120]
            latency = time.perf_counter() - t0
            st.calls += 1
            st.last_area = area
            st.wait = 0
            if p_raw is not None:
                p_cal = float(self.calibrator([p_raw])[0])
                cand.verification.append(p_cal)
                st.evidence.add(p_cal)
                if self.belief is not None:
                    self.belief.update(cand.match_vector(n_queries), p_cal)
        else:
            st.wait += 1
        P = st.evidence.probability()
        have = bool(st.evidence.observations)
        if have and P >= self.p.stop_threshold:
            decision = "stop"
        elif have and P < self.p.reject_below:
            decision = "reject"
        elif st.calls >= self.p.max_calls or st.wait > self.p.max_wait_steps:
            decision = "reject"
        else:
            decision = "wait"
        if decision == "reject":
            cand.rejected = True
        ev = VerifyEvent(step, cand.cid, p_raw, p_cal, float(P), decision, source, answer, latency, float(area),
                         [float(w) for w in self.belief.weights] if self.belief is not None else [], can_call)
        self.events.append(ev)
        return ev

    @property
    def weights(self) -> Optional[List[float]]:
        return [float(w) for w in self.belief.weights] if self.belief is not None else None


def event_dict(ev: VerifyEvent) -> Dict[str, Any]:
    return asdict(ev)


def load_stop_calibration(pol: Mapping[str, Any]) -> Tuple[Calibrator, float, float, bool]:
    """(calibrator, stop threshold, base rate, calibrated?) from ``policy.stop``."""
    st = pol["stop"]
    path = st.get("calibration_file")
    if path:
        art = CalibrationArtifact.load(Path(path))
        return calibrator_from_dict(art.calibrator), float(art.threshold["threshold"]), float(art.base_rate), True
    return IdentityCalibrator(), float(st["debug_threshold"]), float(st["debug_base_rate"]), False


def make_verifier(cfg: Mapping[str, Any], backend: Any, target: str, lang: str, hypotheses: Sequence[str],
                  prior_weights: Sequence[float]) -> Verifier:
    pol = cfg["policy"]
    v, b = pol["verification"], pol["belief"]
    cal, thr, base, calibrated = load_stop_calibration(pol)
    params = VerifyParams(stop_threshold=thr, reject_below=float(v["reject_below"]), base_rate=base,
                          max_calls=int(v["max_calls_per_candidate"]), regrow_factor=float(v["reverify_if_crop_area_grows"]),
                          max_wait_steps=int(v["max_wait_steps"]), crop_padding=float(v["crop_padding"]),
                          min_crop_px=int(v["min_crop_px"]), calibrated=calibrated)
    belief = HypothesisBelief(list(hypotheses), list(prior_weights), BeliefParams(
        rho_right=float(b["rho_right"]), rho_wrong=float(b["rho_wrong"]), base_rate=base,
        evidence_power=float(b["evidence_power"]), weight_floor=float(b["weight_floor"]),
        view_discount=float(b["view_discount"]))) if len(hypotheses) > 0 else None
    return Verifier(backend, VerifyQuery(target, lang), params, belief, cal, float(b["view_discount"]))
