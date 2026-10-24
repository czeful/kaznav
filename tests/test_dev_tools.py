"""Pure parts of the dev tools: GroundingDINO cross-check matching (D-009) and the episode-budget curve (D-005)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from kznav.eval.records import column_names

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import dev_studies  # noqa: E402
import gdino_crosscheck as gx  # noqa: E402


def test_iou_and_one_to_one_matching():
    a = [{"box": [0, 0, 10, 10]}, {"box": [20, 20, 30, 30]}]
    b = [{"box": [1, 0, 11, 10]}, {"box": [0, 0, 10, 10]}, {"box": [50, 50, 60, 60]}]
    assert gx.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0 and gx.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    m = gx.match(a, b)
    assert [(i, j) for i, j, _ in m] == [(0, 1)]                 # best pair first, each box used once


def write(d: Path, name, obj):
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


@pytest.mark.parametrize("shift,met", [(0.0, True), (4.0, False)])
def test_compare_applies_the_fixed_criterion(tmp_path, shift, met):
    items = [{"image": f"i{k}.png", "kind": "sim_object", "queries": ["banana"]} for k in range(10)]
    hf = {it["image"]: [{"box": [0, 0, 10, 10], "score": 0.6, "phrase": "banana"}] for it in items}
    org = {it["image"]: [{"box": [shift, 0, 10 + shift, 10], "score": 0.62, "phrase": "banana"}] for it in items}
    write(tmp_path, "manifest.json", items)
    write(tmp_path, "detections_hf.json", hf)
    write(tmp_path, "detections_original.json", org)
    text = gx.compare(tmp_path)
    assert ("**met**" in text) == met and "| all | 10 | 10 | 10 |" in text


def test_budget_curve_recommends_where_sr_saturates(tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    steps = [100, 150, 200, 260, 900, 1000]
    rows = [{**{c: 0 for c in column_names()}, "episode_id": f"s:{i}", "success": i < 4, "num_decision_steps": s}
            for i, s in enumerate(steps)]
    pd.DataFrame(rows, columns=column_names()).to_csv(d / "episodes.csv", index=False)
    curve = dev_studies.budget_curve([d], tmp_path / "out")
    assert curve.SR.iloc[-1] == pytest.approx(4 / 6)
    rec = int(curve[curve.recommended].max_decision_steps.iloc[0])
    assert 260 <= rec < 1000 and (tmp_path / "out" / "budget_curve.png").exists()


def test_dev_studies_cli_parses_subcommands_with_overrides(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import dev_studies

    seen = {}
    monkeypatch.setattr(dev_studies, "pitch_study", lambda pitches, overrides, out: seen.update(p=pitches, o=overrides)
                        or __import__("pandas").DataFrame([{"pitch_deg": 0}]))
    monkeypatch.setattr(sys, "argv", ["dev_studies.py", "pitch", "--pitches", "0,10", "--out",
                                      str(Path(__import__("tempfile").mkdtemp())), "dataset=x", "embodiment=y"])
    dev_studies.main()
    assert seen == {"p": [0.0, 10.0], "o": ["dataset=x", "embodiment=y"]}


def test_pitch_select_counts_unrunnable_episodes_as_failures(tmp_path):
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import dev_studies

    def run(name, rows):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame(rows, columns=["episode_id", "goal_category", "success"]).to_csv(d / "episodes.csv", index=False)
        return str(d)

    low = run("p0", [("s:1", "chair", True), ("s:2", "chair", True)])                       # sees no TV
    high = run("p20", [("s:1", "chair", True), ("s:2", "chair", False), ("s:3", "tv", True)])
    t = dev_studies.pitch_select({"p0": low, "p20": high}).set_index("pitch")
    assert t.loc["p0", "SR over union"] == pytest.approx(2 / 3) and t.loc["p20", "SR over union"] == pytest.approx(2 / 3)
    assert t.loc["p0", "chosen"] and not t.loc["p20", "chosen"]          # tie: the smaller pitch
    assert t.loc["p0", "SR tv"] == "0/1" and t.loc["p0", "SR over run"] == 1.0


# ---------------------------------------------------------------- D-065: candidate log of "seen but not reached"
def test_candidate_log_records_every_step_and_finds_the_goal_candidate(tmp_path):
    import diag_candidates as dc
    from kznav.perception.base import create_detector, create_segmenter
    from kznav.policy.objectnav import ObjectNavPolicy
    from kznav.run.episode import Executor, run_episode
    from kznav.sim.mock_env import MockEnv, room_with_boxes

    sys.path.insert(0, str(REPO / "tests"))
    from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg

    cfg = small_cfg("wheeled_vacuum")
    cfg["perception"]["detection"]["fusion"]["enabled"] = True     # reliable / suspected flags in the log
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), ["chair"], env.K)
    log = dc.CandidateLog(tmp_path / "cands_mock.jsonl", ep, cfg["policy"], 0.3)
    res, trace = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="t", use_oracle_aux=True, on_frame=log)
    log.close()
    raw = [json.loads(s) for s in (tmp_path / "cands_mock.jsonl").read_text(encoding="utf-8").splitlines()]
    assert raw[0]["episode_id"] == "mock_000" and raw[0]["queries"] == ["chair"] and raw[0]["reliable_conf"] == 0.45
    assert len(raw) - 1 == res.num_decision_steps and [r["step"] for r in raw[1:]] == list(range(len(raw) - 1))
    row = dc.explain(raw[1:], raw[0], goal_radius=0.3, overlap=0.3)
    assert res.success and row["goal_det_steps"] > 0 and row["goal_cands"] >= 1 and row["goal_max_conf"] > 0.45


def _line(step, cands, kind="explore", cand=None, exhausted=False, goal_det=False, verify=None):
    return {"step": step, "kind": kind, "cand": cand, "exhausted": exhausted, "goal_in_view": 0, "goal_det": goal_det,
            "n_det": 0, "verify": verify, "cands": cands}


@pytest.mark.parametrize("lines,reason", [
    ([_line(0, [], goal_det=True)], "no_goal_candidate"),
    ([_line(0, [[1, 0.3, 0.6, 1, 0, 0, 0.1, 3.0, 0.9]])], "never_reliable"),
    ([_line(0, [[1, 0.3, 0.6, 1, 0, 0, 0.1, 3.0, None]], exhausted=True)], "suspected_too_late"),
    ([_line(0, [[1, 0.6, 0.6, 1, 0, 1, 0.1, 3.0, None], [2, 0.7, 0.7, 1, 0, 1, 2.0, 1.0, None]], "approach", 2)],
     "reliable_not_chosen"),
    ([_line(0, [[1, 0.6, 0.6, 1, 0, 1, 0.1, 3.0, None]], "approach", 1),
      _line(1, [[1, 0.6, 0.6, 1, 0, 5, 0.1, 1.0, None]])], "approached_lost"),
    ([_line(0, [[1, 0.6, 0.6, 1, 0, 1, 0.1, 0.5, 0.8]], "approach", 1, verify=[1, "wait", 0.4])],
     "approach_cut_by_budget"),
    ([_line(0, [[1, 0.6, 0.6, 1, 0, 3, 0.1, 0.5, 0.8]], "explore", verify=[1, "reject", 0.1])], "goal_rejected"),
    ([_line(0, [[1, 0.51, 0.51, 1, 0, 1, 0.1, 2.0, None]], "approach", 1),            # D-065: 0.51 -> 0.26
      _line(1, [[1, 0.26, 0.51, 1, 1, 0, 0.1, 1.8, None]])], "became_suspected"),
    ([_line(0, [[1, 0.6, 0.6, 1, 0, 1, 0.1, 2.0, None]], "approach", 1),
      _line(1, [[1, 0.6, 0.6, 1, 0, 1, 0.1, 2.0, None]])], "switched"),
])
def test_explain_names_why_the_goal_was_not_reached(lines, reason):
    import diag_candidates as dc

    header = {"episode_id": "e", "category": "bed", "stop_distance_m": 0.6}
    row = dc.explain(lines, header, goal_radius=0.3, overlap=0.3)
    assert row["reason"] == reason and reason in dc.REASONS
