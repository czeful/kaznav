"""Episode outcome taxonomy and failure attribution.

The runner records *why the episode ended* (``Termination``) plus a few
ground-truth facts; ``classify_episode`` maps them to exactly one
``Outcome``. The funnel columns (goal ever in view -> ever detected -> ever
verified positive) are kept separately in the episode CSV so the failure
classes can be split further post hoc without re-running anything.

``attribute_failure`` assigns every failed episode to exactly one pipeline
``Stage`` - the first stage, in pipeline order, that broke (D-045). Because the
stages partition the failures, a success-rate gap between two arms decomposes
exactly into per-stage differences of failure rates (the H1 test, see
docs/analysis_plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


class Termination(str, Enum):
    """Why an episode ended, as reported by the episode runner."""
    STOP_CALLED = "stop_called"
    STEP_LIMIT = "step_limit"              # decision-step / sim-time budget exhausted
    NO_FRONTIERS = "no_frontiers"          # nothing left to explore, no stop called
    STUCK_ABORT = "stuck_abort"            # stuck longer than the configured limit
    COLLISION_ABORT = "collision_abort"    # only if a collision limit is configured
    FELL = "fell"                          # robot tipped over / fell through a hole or down a step
    MODEL_ERROR = "model_error"            # VLM failed irrecoverably (after retries)
    SIM_ERROR = "sim_error"                # simulator / infrastructure failure


class Outcome(str, Enum):
    """Exactly one outcome class per episode, used in the stage tables and the paper (D-045)."""
    SUCCESS = "success"
    FALSE_STOP = "false_stop"              # stopped at an object that is not a goal instance
    STOP_TOO_FAR = "stop_too_far"          # stopped at a goal instance but outside the success region
    STOP_UNLABELED = "stop_unlabeled"      # stopped, ground truth of the stop candidate unknown
    STUCK = "stuck"
    COLLISIONS = "collisions"
    STEP_LIMIT = "step_limit"
    NOT_FOUND = "not_found"                # exploration exhausted without stopping
    FELL = "fell"
    MODEL_FAILURE = "model_failure"
    SIM_FAILURE = "sim_failure"


@dataclass(frozen=True)
class OutcomeInputs:
    """The facts about a finished episode that ``classify_episode`` needs."""
    termination: Termination
    success: bool
    #: Ground truth for the candidate the robot stopped at: True if it is an
    #: instance of the goal category, False if not, None if no stop / unknown.
    stop_candidate_is_goal: Optional[bool] = None


def classify_episode(x: OutcomeInputs) -> Outcome:
    """Assign the single outcome class of a finished episode.

    Args:
        x: Termination reason, success flag and the ground truth of the stop candidate.

    Returns:
        The outcome; a stop is split into success, false stop, stop too far or unlabeled stop.

    Raises:
        ValueError: If the inputs are inconsistent (e.g. success without a stop call).
    """
    if x.success:
        if x.termination is not Termination.STOP_CALLED:
            raise ValueError(f"success without a stop call (termination={x.termination.value})")
        return Outcome.SUCCESS
    t = x.termination
    if t is Termination.STOP_CALLED:
        if x.stop_candidate_is_goal is None:
            return Outcome.STOP_UNLABELED
        return Outcome.STOP_TOO_FAR if x.stop_candidate_is_goal else Outcome.FALSE_STOP
    mapping = {
        Termination.STEP_LIMIT: Outcome.STEP_LIMIT,
        Termination.NO_FRONTIERS: Outcome.NOT_FOUND,
        Termination.STUCK_ABORT: Outcome.STUCK,
        Termination.COLLISION_ABORT: Outcome.COLLISIONS,
        Termination.FELL: Outcome.FELL,
        Termination.MODEL_ERROR: Outcome.MODEL_FAILURE,
        Termination.SIM_ERROR: Outcome.SIM_FAILURE,
    }
    return mapping[t]


class Stage(str, Enum):
    """Where a failed episode broke. ``NONE`` = success."""

    NONE = "none"
    INFRASTRUCTURE = "infrastructure"      # simulator / model-server failure, not the method (re-run, never reported)
    QUERY = "query"                        # the goal parse failed, or none of its queries names the goal category
    FALSE_STOP = "false_stop"              # stopped at an object that is not a goal instance
    STOP_TOO_FAR = "stop_too_far"          # stopped at a goal instance, outside the success region
    STOP_UNLABELED = "stop_unlabeled"      # stopped; no label for the stop candidate
    STUCK = "stuck"                        # stuck / fell / collision limit (the embodiment)
    NOT_SEEN = "not_seen"                  # the goal was never in view (exploration)
    NOT_DETECTED = "not_detected"          # in view, the detector never fired on it with the episode's queries
    VERIFY_REJECT = "verify_reject"        # detected, the verifier was asked about a goal instance and said no
    NOT_REACHED = "not_reached"            # detected, never verified or stopped at (budget, planning, given up)


#: Stage -> group. H1 compares "language" + "grounding" (the language->vision interface) with "navigation".
STAGE_GROUP: Mapping[Stage, str] = {
    Stage.NONE: "success", Stage.INFRASTRUCTURE: "infrastructure",
    Stage.QUERY: "language",
    Stage.NOT_DETECTED: "grounding", Stage.VERIFY_REJECT: "grounding", Stage.FALSE_STOP: "grounding",
    Stage.NOT_SEEN: "navigation", Stage.NOT_REACHED: "navigation", Stage.STUCK: "navigation",
    Stage.STOP_TOO_FAR: "navigation", Stage.STOP_UNLABELED: "unlabeled",
}

#: Report order (pipeline order).
STAGE_ORDER = (Stage.QUERY, Stage.NOT_SEEN, Stage.NOT_DETECTED, Stage.VERIFY_REJECT, Stage.FALSE_STOP,
               Stage.NOT_REACHED, Stage.STOP_TOO_FAR, Stage.STUCK, Stage.STOP_UNLABELED, Stage.INFRASTRUCTURE)


@dataclass(frozen=True)
class AttributionInputs:
    """The facts about a finished episode that ``attribute_failure`` needs to place it at one failure stage."""
    success: bool
    termination: Termination
    stop_candidate_is_goal: Optional[bool] = None
    parse_ok: bool = True
    hypothesis_match: Optional[bool] = None    # None: no language model (the query is the category name)
    goal_ever_in_view: bool = False
    goal_ever_detected: bool = False
    goal_ever_verify_rejected: bool = False    # a goal instance was shown to the verifier and rejected


_STUCK = (Termination.STUCK_ABORT, Termination.FELL, Termination.COLLISION_ABORT)
ORDERS = ("pipeline", "navigation_first")


def attribute_failure(x: AttributionInputs, order: str = "pipeline") -> Stage:
    """Exactly one stage per failed episode; the first matching rule wins (``order="pipeline"``, the default):

    1. success -> NONE;  2. simulator error, or a model error after a successful parse -> INFRASTRUCTURE;
    3. parse failed or no query names the goal -> QUERY;  4. stop called -> FALSE_STOP / STOP_TOO_FAR /
    STOP_UNLABELED by the stop candidate's label;  5. stuck, fell, collision limit -> STUCK;
    6. goal never in view -> NOT_SEEN;  7. never detected -> NOT_DETECTED;  8. a goal instance rejected by the
    verifier -> VERIFY_REJECT;  9. otherwise (detected, not rejected, never reached) -> NOT_REACHED.

    ``order="navigation_first"`` is the sensitivity analysis of H1: the navigation rules come first (after 1-2:
    STUCK, then NOT_SEEN for any episode in which the goal was never in view, stop or not), then 3, 4, 7, 8, 9.
    It attributes to navigation every failure that navigation alone would explain, so a language/grounding
    share that survives it is not an artefact of the rule order (docs/analysis_plan.md).
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    t = x.termination
    if x.success:
        return Stage.NONE
    if t is Termination.SIM_ERROR or (t is Termination.MODEL_ERROR and x.parse_ok):
        return Stage.INFRASTRUCTURE
    if order == "navigation_first" and x.parse_ok:
        if t in _STUCK:
            return Stage.STUCK
        if not x.goal_ever_in_view:
            return Stage.NOT_SEEN
    if not x.parse_ok or x.hypothesis_match is False:
        return Stage.QUERY
    if t is Termination.STOP_CALLED:
        if x.stop_candidate_is_goal is None:
            return Stage.STOP_UNLABELED
        return Stage.STOP_TOO_FAR if x.stop_candidate_is_goal else Stage.FALSE_STOP
    if t in _STUCK:
        return Stage.STUCK
    if not x.goal_ever_in_view:
        return Stage.NOT_SEEN
    if not x.goal_ever_detected:
        return Stage.NOT_DETECTED
    if x.goal_ever_verify_rejected:
        return Stage.VERIFY_REJECT
    return Stage.NOT_REACHED


def _opt_bool(v) -> Optional[bool]:
    if v is None or (isinstance(v, float) and v != v) or v == "":
        return None
    if isinstance(v, str):
        return {"true": True, "false": False}.get(v.strip().lower())
    return bool(v)


def attribute_row(row: Mapping, order: str = "pipeline") -> Stage:
    """``attribute_failure`` from an episodes.csv row (dict or pandas row); rows before schema v4 have no
    verifier-rejection column and are attributed without rule 8 (their rejections count as NOT_REACHED)."""
    return attribute_failure(AttributionInputs(
        success=bool(_opt_bool(row["success"])), termination=Termination(row["termination"]),
        stop_candidate_is_goal=_opt_bool(row.get("stop_candidate_is_goal")),
        parse_ok=_opt_bool(row.get("parse_ok")) is not False,
        hypothesis_match=_opt_bool(row.get("hypothesis_match")),
        goal_ever_in_view=bool(_opt_bool(row.get("goal_ever_in_view"))),
        goal_ever_detected=bool(_opt_bool(row.get("goal_ever_detected"))),
        goal_ever_verify_rejected=bool(_opt_bool(row.get("goal_ever_verify_rejected")))), order)
