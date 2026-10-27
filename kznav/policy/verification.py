"""Verify-before-stop: the VLM checks a crop of the candidate the robot is about to stop at.

When the policy is within stop range of its best candidate and sees it in the current frame, it
asks ``Verifier.check``. A call is made for the first view whose detection box is at least
``min_box_side_px`` on its shorter side (a smaller object is not judged: the robot approaches it,
D-032) and again only when the crop grew by ``regrow_factor`` (a closer, more informative view),
at most ``max_calls`` times per candidate.
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

from ..language.base import BackendError, TransportError
from ..language.schemas import VerifyQuery
from .belief import BeliefParams, CandidateEvidence, HypothesisBelief
from .calibration import CalibrationArtifact, Calibrator, IdentityCalibrator, calibrator_from_dict


@dataclass(frozen=True)
class VerifyParams:
    """Parameters of verify-before-stop (``policy.verification`` and the stop calibration)."""
    stop_threshold: float
    reject_below: float
    base_rate: float
    max_calls: int
    regrow_factor: float
    max_wait_steps: int
    crop_padding: float
    min_crop_px: int
    calibrated: bool            # False: identity calibrator + debug threshold (debug-only runs)
    min_box_side_px: int = 0    # boxes smaller than this on their shorter side get no call (D-032)
    new_view_deg: float = 0.0   # >0: a call is also made from this many degrees around the object (D-071 second look)


@dataclass
class VerifyEvent:
    """One verification decision about one candidate, logged to ``events.jsonl``."""
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
    box_xyxy: List[float] = field(default_factory=list)    # the candidate's detection box in this frame
    crop_xyxy: List[int] = field(default_factory=list)     # the pixels sent to the model (box + padding)
    dist_m: Optional[float] = None                         # robot to the candidate's nearest point (D-070)


@dataclass
class _CandState:
    evidence: CandidateEvidence
    calls: int = 0
    last_area: float = 0.0
    wait: int = 0
    last_bearing: Optional[float] = None     # robot's bearing around the object at the last call


def crop_bounds(shape: Sequence[int], box: Sequence[float], padding: float, min_px: int) -> Tuple[int, int, int, int]:
    """Pixel bounds (x1, y1, x2, y2) of the crop around a detection box: ``padding`` (fraction of the box size)
    of context per side, at least ``min_px`` per side, clamped to the image."""
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = max(bw * (1 + 2 * padding), min_px) / 2
    hh = max(bh * (1 + 2 * padding), min_px) / 2
    return (int(max(0, np.floor(cx - hw))), int(max(0, np.floor(cy - hh))),
            int(min(w, np.ceil(cx + hw))), int(min(h, np.ceil(cy + hh))))


def crop_box(rgb: np.ndarray, box: Sequence[float], padding: float, min_px: int) -> Tuple[np.ndarray, float]:
    """Crop around a detection box with ``padding`` (fraction of the box size) of context per side."""
    a, c, b, d = crop_bounds(rgb.shape, box, padding, min_px)
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.ascontiguousarray(rgb[c:d, a:b]), max(1.0, x2 - x1) * max(1.0, y2 - y1)


class Verifier:
    """Decides, candidate by candidate, whether the robot may stop: asks the backend about crops and accumulates
    the calibrated answers.

    Args:
        backend: VLM backend or local verifier with ``verify_candidate``.
        query: What is asked about each crop (name and language).
        params: Thresholds, call budget and crop settings.
        belief: Hypothesis belief updated by every answer, or None.
        calibrator: Maps raw P(yes) to calibrated probabilities; identity if None.
        view_discount: Discount of repeated views of one candidate.
        first_view_full: Count the first answer in full (D-069).
    """
    def __init__(self, backend: Any, query: VerifyQuery, params: VerifyParams, belief: Optional[HypothesisBelief],
                 calibrator: Optional[Calibrator] = None, view_discount: float = 1.0, first_view_full: bool = False):
        self.backend = backend
        self.query = query
        self.p = params
        self.belief = belief
        self.calibrator = calibrator or IdentityCalibrator()
        self.view_discount = view_discount
        self.first_view_full = first_view_full
        self.events: List[VerifyEvent] = []
        self._state: Dict[int, _CandState] = {}

    def probability(self, cid: int) -> Optional[float]:
        """Accumulated P(candidate is the target), or None if the candidate has not been asked about yet."""
        st = self._state.get(cid)
        return st.evidence.probability() if st is not None and st.calls else None

    def can_call(self, cand: Any, bearing: Optional[float] = None) -> bool:
        """Would ``check`` ask the model about ``cand`` now (first view, a view grown by ``regrow_factor``, or - with
        ``new_view_deg`` - a view from far enough around the object)? No side effects (D-070, D-071)."""
        st = self._state.get(cand.cid)
        x1, y1, x2, y2 = cand.last_box
        area = max(1.0, x2 - x1) * max(1.0, y2 - y1)
        if min(x2 - x1, y2 - y1) < self.p.min_box_side_px:
            return False
        if st is None or st.calls == 0:
            return True
        return st.calls < self.p.max_calls and (area >= self.p.regrow_factor * st.last_area or self._new_view(st, bearing))

    def _new_view(self, st: _CandState, bearing: Optional[float]) -> bool:
        if self.p.new_view_deg <= 0 or bearing is None or st.last_bearing is None:
            return False
        d = abs((bearing - st.last_bearing + np.pi) % (2 * np.pi) - np.pi)
        return bool(np.degrees(d) >= self.p.new_view_deg)

    def calls(self, cid: int) -> int:
        """Number of verifier calls made for a candidate."""
        st = self._state.get(cid)
        return st.calls if st is not None else 0

    def undecided(self, cid: int) -> bool:
        """Asked at least once and the candidate's probability between ``reject_below`` and the stop threshold."""
        P = self.probability(cid)
        return P is not None and self.p.reject_below <= P < self.p.stop_threshold

    def check(self, cand: Any, rgb: np.ndarray, step: int, n_queries: int, dist_m: Optional[float] = None,
              bearing: Optional[float] = None) -> VerifyEvent:
        """One stop-range decision about ``cand`` (seen in this frame's ``rgb`` at ``cand.last_box``)."""
        st = self._state.setdefault(cand.cid, _CandState(CandidateEvidence(
            self.p.base_rate, self.view_discount, first_view_full=self.first_view_full)))
        crop, area = crop_box(rgb, cand.last_box, self.p.crop_padding, self.p.min_crop_px)
        x1, y1, x2, y2 = cand.last_box
        big_enough = min(x2 - x1, y2 - y1) >= self.p.min_box_side_px
        can_call = big_enough and st.calls < self.p.max_calls and \
            (st.calls == 0 or area >= self.p.regrow_factor * st.last_area or self._new_view(st, bearing))
        p_raw = p_cal = None
        source, answer, latency = "none", "", 0.0
        if can_call:
            t0 = time.perf_counter()
            try:
                r = self.backend.verify_candidate(crop, self.query)
                p_raw, source, answer = r.p_target, r.source, r.answer_text
            except TransportError:
                raise                          # server down: the episode fails as infrastructure (D-046)
            except BackendError as e:          # invalid output after retries; the call is logged by the sink
                source, answer = "error", str(e)[:120]
            latency = time.perf_counter() - t0
            st.calls += 1
            st.last_area = area
            st.last_bearing = bearing
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
                         [float(w) for w in self.belief.weights] if self.belief is not None else [], can_call,
                         box_xyxy=[float(v) for v in cand.last_box],
                         crop_xyxy=list(crop_bounds(rgb.shape, cand.last_box, self.p.crop_padding, self.p.min_crop_px)),
                         dist_m=None if dist_m is None else float(dist_m))
        self.events.append(ev)
        return ev

    def check_crop(self, cand: Any, crop: np.ndarray, step: int, n_queries: int) -> VerifyEvent:
        """D-072: ask about a crop stored when the candidate was detected, without driving there. Only a confident
        "no" acts (the candidate is rejected, the robot does not drive to it); otherwise the answer is evidence and the
        robot verifies on the spot as usual (its first real view counts as a new view)."""
        st = self._state.setdefault(cand.cid, _CandState(CandidateEvidence(
            self.p.base_rate, self.view_discount, first_view_full=self.first_view_full)))
        p_raw = p_cal = None
        source, answer, latency, called = "none", "", 0.0, False
        if st.calls < self.p.max_calls:
            t0 = time.perf_counter()
            try:
                r = self.backend.verify_candidate(np.ascontiguousarray(crop), self.query)
                p_raw, source, answer = r.p_target, r.source, r.answer_text
            except TransportError:
                raise
            except BackendError as e:
                source, answer = "error", str(e)[:120]
            latency = time.perf_counter() - t0
            called = True
            st.calls += 1
            st.last_area = 0.0                 # any real view later is a new one
            if p_raw is not None:
                p_cal = float(self.calibrator([p_raw])[0])
                cand.verification.append(p_cal)
                st.evidence.add(p_cal)
                if self.belief is not None:
                    self.belief.update(cand.match_vector(n_queries), p_cal)
        P = st.evidence.probability()
        decision = "reject" if st.evidence.observations and P < self.p.reject_below else "wait"
        if decision == "reject":
            cand.rejected = True
        ev = VerifyEvent(step, cand.cid, p_raw, p_cal, float(P), decision, source, answer, latency,
                         float(crop.shape[0] * crop.shape[1]), self.weights or [], called)
        self.events.append(ev)
        return ev

    @property
    def weights(self) -> Optional[List[float]]:
        """Current hypothesis weights, or None without a belief."""
        return [float(w) for w in self.belief.weights] if self.belief is not None else None


def event_dict(ev: VerifyEvent) -> Dict[str, Any]:
    """A verification event as a JSON-ready dict."""
    return asdict(ev)


def load_stop_calibration(pol: Mapping[str, Any]) -> Tuple[Calibrator, float, float, bool]:
    """(calibrator, stop threshold, base rate, calibrated?) from ``policy.stop``."""
    st = pol["stop"]
    path = st.get("calibration_file")
    if path:
        art = CalibrationArtifact.load(Path(path))
        return calibrator_from_dict(art.calibrator), float(art.threshold["threshold"]), float(art.base_rate), True
    return IdentityCalibrator(), float(st["debug_threshold"]), float(st["debug_base_rate"]), False


def question(cfg: Mapping[str, Any], target: str, lang: str, target_en: str = "",
             hypotheses: Sequence[str] = ()) -> VerifyQuery:
    """What the verifier asks about each crop. ``question_lang: command`` (default): the target as named in the
    command, in the command's language. ``en`` (D-070 stage 2, D-071): the same question in English about the model's
    own English name of the target (parse_goal v3), falling back to its most likely hypothesis; the command is still
    understood only by the model, never by the pipeline."""
    mode = str(cfg["policy"]["verification"].get("question_lang", "command"))
    if mode == "command":
        return VerifyQuery(target, lang)
    if mode != "en":
        raise ValueError(f"policy.verification.question_lang must be command or en, not {mode!r}")
    name = target_en or (hypotheses[0] if hypotheses else "") or target
    return VerifyQuery(name, "en")


def make_verifier(cfg: Mapping[str, Any], backend: Any, target: str, lang: str, hypotheses: Sequence[str],
                  prior_weights: Sequence[float], target_en: str = "") -> Verifier:
    """Build the verifier of one episode from the config: question, calibration, thresholds and belief.

    Args:
        cfg: The run config.
        backend: VLM backend or local verifier.
        target: The target as named in the command.
        lang: Language of the command (or the pipeline language after translation).
        hypotheses: The goal's English hypotheses.
        prior_weights: Their parsed weights.
        target_en: The model's own English name of the target, used by the English question.

    Returns:
        The verifier.
    """
    pol = cfg["policy"]
    v, b = pol["verification"], pol["belief"]
    cal, thr, base, calibrated = load_stop_calibration(pol)
    params = VerifyParams(stop_threshold=thr, reject_below=float(v["reject_below"]), base_rate=base,
                          max_calls=int(v["max_calls_per_candidate"]), regrow_factor=float(v["reverify_if_crop_area_grows"]),
                          max_wait_steps=int(v["max_wait_steps"]), crop_padding=float(v["crop_padding"]),
                          min_crop_px=int(v["min_crop_px"]), calibrated=calibrated,
                          min_box_side_px=int(v.get("min_box_side_px", 0)),
                          new_view_deg=float(v.get("second_look_deg", 0.0)))
    belief = HypothesisBelief(list(hypotheses), list(prior_weights), BeliefParams(
        rho_right=float(b["rho_right"]), rho_wrong=float(b["rho_wrong"]), base_rate=base,
        evidence_power=float(b["evidence_power"]), weight_floor=float(b["weight_floor"]),
        view_discount=float(b["view_discount"]))) if len(hypotheses) > 0 else None
    # D-069: read with a default, so configs without the key (every run before D-069) keep their hash and behaviour
    return Verifier(backend, question(cfg, target, lang, target_en, hypotheses), params, belief, cal,
                    float(b["view_discount"]),
                    first_view_full=bool(b.get("first_view_full", False)))
