"""Episode outcome taxonomy.

The runner records *why the episode ended* (``Termination``) plus a few
ground-truth facts; ``classify_episode`` maps them to exactly one
``Outcome``. The funnel columns (goal ever in view -> ever detected -> ever
verified positive) are kept separately in the episode CSV so the failure
classes can be split further post hoc without re-running anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Termination(str, Enum):
    STOP_CALLED = "stop_called"
    STEP_LIMIT = "step_limit"              # decision-step / sim-time budget exhausted
    NO_FRONTIERS = "no_frontiers"          # nothing left to explore, no stop called
    STUCK_ABORT = "stuck_abort"            # stuck longer than the configured limit
    COLLISION_ABORT = "collision_abort"    # only if a collision limit is configured
    MODEL_ERROR = "model_error"            # VLM failed irrecoverably (after retries)
    SIM_ERROR = "sim_error"                # simulator / infrastructure failure


class Outcome(str, Enum):
    SUCCESS = "success"
    FALSE_STOP = "false_stop"              # stopped at an object that is not a goal instance
    STOP_TOO_FAR = "stop_too_far"          # stopped at a goal instance but outside the success region
    STOP_UNLABELED = "stop_unlabeled"      # stopped, ground truth of the stop candidate unknown
    STUCK = "stuck"
    COLLISIONS = "collisions"
    STEP_LIMIT = "step_limit"
    NOT_FOUND = "not_found"                # exploration exhausted without stopping
    MODEL_FAILURE = "model_failure"
    SIM_FAILURE = "sim_failure"


@dataclass(frozen=True)
class OutcomeInputs:
    termination: Termination
    success: bool
    #: Ground truth for the candidate the robot stopped at: True if it is an
    #: instance of the goal category, False if not, None if no stop / unknown.
    stop_candidate_is_goal: Optional[bool] = None


def classify_episode(x: OutcomeInputs) -> Outcome:
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
        Termination.MODEL_ERROR: Outcome.MODEL_FAILURE,
        Termination.SIM_ERROR: Outcome.SIM_FAILURE,
    }
    return mapping[t]
