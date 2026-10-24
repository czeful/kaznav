"""Failure attribution (D-045), content stop labels (D-046), episode failures in the runner (D-046) and
scene-clustered inference (D-047)."""
import csv

import numpy as np
import pytest

from kznav.eval.failures import (STAGE_GROUP, AttributionInputs, Stage, Termination, attribute_failure,
                                 attribute_row)
from kznav.eval.stats import bootstrap_ci, cluster_bootstrap_mean, cluster_sign_flip_test
from kznav.language.base import InvalidOutputError, ListSink, TransportError
from kznav.language.goal import GoalCache, parse_command
from kznav.perception.base import create_segmenter
from kznav.perception.oracle import OracleSemanticDetector
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.run import benchmark as bm
from kznav.run.benchmark import RunAborted
from kznav.run.episode import Executor, run_episode
from kznav.run.logging import RunLogger
from kznav.sim.mock_env import MockEnv, room_with_boxes

from test_benchmark import CHAIR, TABLE, agent_view_points, dataset_episode, goal, runner_for, rows
from test_smoke_pipeline import GOAL, episode_for, small_cfg

T = Termination


# ---------------------------------------------------------------- attribution rules
@pytest.mark.parametrize("kw,stage", [
    (dict(success=True, termination=T.STOP_CALLED, stop_candidate_is_goal=True), Stage.NONE),
    (dict(success=False, termination=T.SIM_ERROR), Stage.INFRASTRUCTURE),
    (dict(success=False, termination=T.MODEL_ERROR, parse_ok=True), Stage.INFRASTRUCTURE),
    (dict(success=False, termination=T.MODEL_ERROR, parse_ok=False), Stage.QUERY),
    (dict(success=False, termination=T.STOP_CALLED, stop_candidate_is_goal=False, hypothesis_match=False), Stage.QUERY),
    (dict(success=False, termination=T.STOP_CALLED, stop_candidate_is_goal=False, hypothesis_match=True),
     Stage.FALSE_STOP),
    (dict(success=False, termination=T.STOP_CALLED, stop_candidate_is_goal=True), Stage.STOP_TOO_FAR),
    (dict(success=False, termination=T.STOP_CALLED), Stage.STOP_UNLABELED),
    (dict(success=False, termination=T.STUCK_ABORT, goal_ever_in_view=False), Stage.STUCK),
    (dict(success=False, termination=T.FELL, goal_ever_in_view=True, goal_ever_detected=True), Stage.STUCK),
    (dict(success=False, termination=T.STEP_LIMIT), Stage.NOT_SEEN),
    (dict(success=False, termination=T.NO_FRONTIERS, goal_ever_in_view=True), Stage.NOT_DETECTED),
    (dict(success=False, termination=T.STEP_LIMIT, goal_ever_in_view=True, goal_ever_detected=True,
          goal_ever_verify_rejected=True), Stage.VERIFY_REJECT),
    (dict(success=False, termination=T.STEP_LIMIT, goal_ever_in_view=True, goal_ever_detected=True), Stage.NOT_REACHED),
])
def test_attribution_rules(kw, stage):
    assert attribute_failure(AttributionInputs(**kw)) is stage


def test_every_stage_has_a_group_and_no_language_model_means_no_query_failure():
    assert set(STAGE_GROUP) == set(Stage)
    x = AttributionInputs(success=False, termination=T.STEP_LIMIT, hypothesis_match=None)
    assert attribute_failure(x) is Stage.NOT_SEEN


def test_attribution_from_csv_strings():
    row = {"success": "False", "termination": "stop_called", "stop_candidate_is_goal": "", "parse_ok": "True",
           "hypothesis_match": "", "goal_ever_in_view": "True", "goal_ever_detected": "True"}
    assert attribute_row(row) is Stage.STOP_UNLABELED
    assert attribute_row({**row, "stop_candidate_is_goal": "False"}) is Stage.FALSE_STOP
    assert attribute_row({**row, "hypothesis_match": "False"}) is Stage.QUERY


# ---------------------------------------------------------------- stop labels in the episode
class TableDetector(OracleSemanticDetector):
    """Fires on the table whatever it is asked for: a guaranteed false stop."""

    def detect(self, rgb, queries, aux=None):
        return super().detect(rgb, queries, {"semantic": aux["semantic"], "goal_ids": {0: [TABLE.semantic_id]}})


def test_stop_labels_by_content_and_position():
    cfg = small_cfg("habitat_agent")
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    ok = ObjectNavPolicy(cfg, OracleSemanticDetector({"min_pixels": 20}), create_segmenter(cfg["segmenter"]),
                         ["chair"], env.K)
    res, _ = run_episode(env, ok, Executor(env, cfg), ep, cfg, run_id="t", use_oracle_aux=True)
    assert res.success and res.failure_stage == "none" and res.stop_label_source == "content"
    assert res.stop_box_goal_fraction >= 0.3 and res.stop_candidate_is_goal is True
    bad = ObjectNavPolicy(cfg, TableDetector({"min_pixels": 20}), create_segmenter(cfg["segmenter"]), ["chair"], env.K)
    res, _ = run_episode(env, bad, Executor(env, cfg), ep, cfg, run_id="t", use_oracle_aux=True)
    assert res.termination == "stop_called" and res.outcome == "false_stop" and res.failure_stage == "false_stop"
    assert res.stop_box_goal_fraction == pytest.approx(0.0) and res.stop_is_goal_by_position is False


# ---------------------------------------------------------------- infrastructure errors are not results
class DownBackend:
    """parse_goal / verify_candidate fail like a server that is down (or like an invalid answer)."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def parse_goal(self, text, lang):
        self.calls += 1
        raise self.exc


def test_transport_errors_propagate_and_are_not_cached():
    b = DownBackend(TransportError("connection refused"))
    with pytest.raises(TransportError):
        parse_command(b, ListSink(), command_id="c", text="x", lang="kk", category="chair", word_type="universal",
                      lexicon=None)
    cache = GoalCache(b, ListSink(), None)
    for _ in range(2):
        with pytest.raises(TransportError):
            cache.get(command_id="c", text="x", lang="kk", category="chair", word_type="universal")
    assert b.calls == 3                                         # nothing was cached
    g = parse_command(DownBackend(InvalidOutputError("not json")), ListSink(), command_id="c", text="x", lang="kk",
                      category="chair", word_type="universal", lexicon=None)
    assert not g.parse_ok                                       # the model's fault: a failed parse, a result


# ---------------------------------------------------------------- the runner keeps going
def failing_run_episode(fail_ids, real=bm.run_episode):
    def run(env, policy, executor, ep, cfg, **kw):
        if ep.episode_id in fail_ids:
            raise RuntimeError(f"boom in {ep.episode_id}")
        return real(env, policy, executor, ep, cfg, **kw)
    return run


def test_runner_records_failures_retries_on_resume_then_gives_up(tmp_path, monkeypatch):
    cfg = small_cfg("habitat_agent")
    eps = dataset_episode(tmp_path, goals_list=[goal(vps=agent_view_points(cfg))],
                          starts=((-2.0, 0.0, 2.0), (-1.5, 0.0, 2.2), (-2.2, 0.0, 1.5)))
    monkeypatch.setattr(bm, "run_episode", failing_run_episode({"mockscene:1"}))
    r, logger = runner_for(tmp_path, cfg)
    built = []
    factory = r.env_factory
    r.env_factory = lambda s: built.append(s) or factory(s)
    assert r.run(eps) == {"run": 2, "excluded": 0, "skipped_done": 0, "failed": 1}
    logger.close()
    assert len(built) == 2                                      # the simulator was rebuilt after the failure
    f = rows(r.failures_csv)
    assert [(x["episode_id"], x["attempt"], x["kind"]) for x in f] == [("mockscene:1", "1", "exception")]
    assert (r.out / f[0]["traceback_file"]).read_text(encoding="utf-8").count("boom in mockscene:1")
    assert [d["episode_id"] for d in rows(r.episodes_csv)] == ["mockscene:0", "mockscene:2"]
    again, logger = runner_for(tmp_path, cfg)
    assert again.run(eps) == {"run": 0, "excluded": 0, "skipped_done": 2, "failed": 1}     # retried once more
    logger.close()
    third, logger = runner_for(tmp_path, cfg)
    assert third.run(eps)["skipped_done"] == 3                 # given up after max_episode_attempts = 2
    logger.close()
    monkeypatch.setattr(bm, "run_episode", failing_run_episode(set()))
    cfg_more = {**cfg, "run": {**cfg["run"], "max_episode_attempts": 3}}
    fixed, logger = runner_for(tmp_path, cfg_more)
    assert fixed.run(eps) == {"run": 1, "excluded": 0, "skipped_done": 2, "failed": 0}
    logger.close()
    done = rows(fixed.episodes_csv)
    assert done[-1]["episode_id"] == "mockscene:1" and '"attempt": 3' in done[-1]["notes"]


def test_consecutive_failures_abort_the_run(tmp_path, monkeypatch):
    cfg = small_cfg("habitat_agent")
    cfg = {**cfg, "run": {**cfg["run"], "max_consecutive_failures": 2}}
    eps = dataset_episode(tmp_path, goals_list=[goal(vps=agent_view_points(cfg))],
                          starts=((-2.0, 0.0, 2.0), (-1.5, 0.0, 2.2), (-2.2, 0.0, 1.5)))
    monkeypatch.setattr(bm, "run_episode", failing_run_episode({e.episode_id for e in eps}))
    r, logger = runner_for(tmp_path, cfg)
    with pytest.raises(RunAborted, match="2 episodes failed in a row"):
        r.run(eps)
    logger.close()
    assert len(rows(r.failures_csv)) == 2 and not r.episodes_csv.exists()


def test_events_carry_attempt_ids(tmp_path):
    cfg = small_cfg("habitat_agent")
    eps = dataset_episode(tmp_path, goals_list=[goal(vps=agent_view_points(cfg))])[:1]
    r, logger = runner_for(tmp_path, cfg)
    r.run(eps)
    logger.close()
    import json
    ev = [json.loads(l) for l in open(r.out / "events.jsonl", encoding="utf-8")]
    done = [e for e in ev if e["type"] == "episode_done"]
    assert len(done) == 1 and done[0]["attempt"] == "mockscene:0#1" and done[0]["failure_stage"] == "none"
    assert all(e.get("attempt") == "mockscene:0#1" for e in ev)


# ---------------------------------------------------------------- scene-clustered inference
def test_cluster_bootstrap_is_wider_than_iid_for_clustered_data():
    rng = np.random.default_rng(0)
    scene_effect = rng.normal(0, 0.4, size=8)
    clusters = np.repeat(np.arange(8), 25)
    x = (rng.random(200) < np.clip(0.4 + scene_effect[clusters], 0, 1)).astype(float)
    c = cluster_bootstrap_mean(x, clusters, n_boot=4000)
    iid = bootstrap_ci(x, n_boot=4000)
    assert c.estimate == pytest.approx(x.mean()) and c.n == 8
    assert (c.high - c.low) > 1.3 * (iid.high - iid.low)
    assert cluster_bootstrap_mean(x, clusters, n_boot=4000) == c              # deterministic
    one = cluster_bootstrap_mean(x, np.zeros(200))
    assert one.low == one.high == one.estimate                                  # one scene: nothing to resample


def test_sign_flip_test_exact_and_monte_carlo():
    d = np.array([1, 1, 0, 1, 1, 0, 1, 1.0])
    clusters = np.repeat(np.arange(4), 2)                                       # every scene positive
    r = cluster_sign_flip_test(d, clusters)
    assert r.method == "exact" and r.n_permutations == 16 and r.p_value == pytest.approx(2 / 16)
    assert cluster_sign_flip_test(d, np.zeros(8)).p_value == 1.0                # one scene: never significant
    assert cluster_sign_flip_test(np.zeros(6), np.arange(6)).p_value == 1.0
    big = cluster_sign_flip_test(np.ones(40), np.repeat(np.arange(20), 2), n_perm=5000)
    assert big.method == "monte-carlo" and big.p_value < 0.001
    rng = np.random.default_rng(3)
    null = [cluster_sign_flip_test(rng.normal(size=60), np.repeat(np.arange(12), 5)).p_value for _ in range(200)]
    assert 0.02 < np.mean(np.array(null) < 0.05) < 0.10                          # calibrated under the null


def test_navigation_first_order_moves_navigation_failures_out_of_the_language_stages():
    wrong_query_never_seen = AttributionInputs(success=False, termination=T.STEP_LIMIT, hypothesis_match=False)
    assert attribute_failure(wrong_query_never_seen) is Stage.QUERY
    assert attribute_failure(wrong_query_never_seen, "navigation_first") is Stage.NOT_SEEN
    false_stop_never_seen = AttributionInputs(success=False, termination=T.STOP_CALLED, stop_candidate_is_goal=False,
                                              hypothesis_match=True)
    assert attribute_failure(false_stop_never_seen, "navigation_first") is Stage.NOT_SEEN
    seen = AttributionInputs(success=False, termination=T.STEP_LIMIT, hypothesis_match=False, goal_ever_in_view=True)
    assert attribute_failure(seen, "navigation_first") is Stage.QUERY
    parse_failed = AttributionInputs(success=False, termination=T.MODEL_ERROR, parse_ok=False)
    assert attribute_failure(parse_failed, "navigation_first") is Stage.QUERY      # never drove: not navigation
    with pytest.raises(ValueError):
        attribute_failure(seen, "random")


def test_only_events_of_completed_attempts_are_read(tmp_path):
    import json

    from kznav.run.logging import completed_events

    lines = [{"type": "verification", "episode_id": "s:1", "attempt": "s:1#1", "p_raw": 0.9},      # crashed attempt
             {"type": "episode_failed", "episode_id": "s:1", "attempt": "s:1#1"},
             {"type": "verification", "episode_id": "s:1", "attempt": "s:1#2", "p_raw": 0.2},
             {"type": "episode_done", "episode_id": "s:1", "attempt": "s:1#2"},
             {"type": "verification", "episode_id": "dbg", "p_raw": 0.5}]                            # no attempt id
    f = tmp_path / "events.jsonl"
    f.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    assert [e["p_raw"] for e in completed_events(f, "verification")] == [0.2, 0.5]
