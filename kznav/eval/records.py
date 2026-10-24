"""The per-episode CSV contract: one ``EpisodeResult`` = one row.

Every aggregate in the paper is computed from these rows, so the column set
is fixed here and versioned (``SCHEMA_VERSION``). Add columns, never rename
or repurpose them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import List, Optional

SCHEMA_VERSION = 4   # v2: + recoveries; v3: + goal_mode, pipeline_lang, translation, translation_latency_s;
                     # v4: + stop labels, verifier rejections, failure_stage, vlm_call_errors, localization
                     #     (D-045, D-046, D-051)


@dataclass
class EpisodeResult:
    # --- identity -------------------------------------------------------------
    run_id: str
    cell_id: str                 # matrix cell (backend x condition), "" for single runs
    dataset: str                 # e.g. hm3d_objectnav_v1 / hm3d_ovon / custom_room
    split: str                   # e.g. val / dev
    scene_id: str
    episode_id: str
    goal_category: str
    word_type: str               # universal | cultural
    command_lang: str            # en | kk | kk_ru | ru
    command_id: str
    command_text: str
    backend: str
    model_id: str
    embodiment: str
    seed: int
    # --- standard ObjectNav metrics (kznav.eval.metrics) ----------------------
    success: bool
    spl: float
    soft_spl: float
    dts: float
    distance_to_goal: float
    start_geodesic: float
    path_length: float
    num_decision_steps: int
    sim_time_s: float
    wall_time_s: float
    # --- outcome (kznav.eval.failures) ----------------------------------------
    termination: str
    outcome: str
    stop_candidate_is_goal: Optional[bool]
    # --- embodiment -----------------------------------------------------------
    collisions: int
    stuck_time_s: float
    # --- funnel (ground truth from the semantic sensor / annotations) ---------
    goal_ever_in_view: bool
    goal_ever_detected: bool
    goal_ever_verified_positive: bool
    # --- language -------------------------------------------------------------
    parse_ok: bool
    parse_attempts: int
    n_hypotheses: int
    target_match: Optional[bool]       # parsed target in the accepted synonym list
    hypothesis_match: Optional[bool]   # any English hypothesis in the category synonym list
    # --- cost -----------------------------------------------------------------
    vlm_calls: int
    vlm_calls_verify: int
    vlm_latency_total_s: float
    vlm_latency_median_s: float
    vlm_http_retries: int
    vlm_json_retries: int
    verify_logprobs_missing: int
    # --- recovery behaviour (v2) ----------------------------------------------
    recoveries: int = 0              # times the robot backed off after making no progress
    # --- goal ablations (v3, D-038 / D-039) -----------------------------------
    goal_mode: str = ""              # multi | top1 | mt_command | mt_word ("" without a language model)
    pipeline_lang: str = ""          # language of the parse and the verification question (en after mt_command)
    translation: str = ""            # translator output: the command (mt_command) or the object name (mt_word)
    translation_latency_s: float = 0.0   # the translation call, charged to every episode of the command
    # --- stop labels and failure attribution (v4, D-045 / D-046) ---------------
    stop_box_goal_fraction: Optional[float] = None   # goal pixels in the stop candidate's latest box (content label)
    stop_is_goal_by_position: Optional[bool] = None  # stop candidate within stop_candidate_radius_m of a goal AABB
    stop_label_source: str = ""                      # content | position | "" - what stop_candidate_is_goal uses
    goal_ever_verify_rejected: bool = False          # a goal instance (content label) shown to the verifier, rejected
    failure_stage: str = ""                          # kznav.eval.failures.Stage: one stage per failed episode
    vlm_call_errors: int = 0                         # model calls that failed after retries (invalid output)
    localization: str = "ground_truth"               # pose given to the policy: ground_truth | odometry (D-051)
    loc_error_final_m: float = 0.0                   # |estimated - true| position at the end of the episode
    loc_heading_error_final_deg: float = 0.0
    # --- provenance -----------------------------------------------------------
    schema_version: int = SCHEMA_VERSION
    notes: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def column_names() -> List[str]:
    return [f.name for f in fields(EpisodeResult)]
