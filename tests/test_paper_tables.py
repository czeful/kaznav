"""Paper tables (kznav.eval.paper): pooling seeds, paired contrasts with Holm, rendering."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kznav.eval import paper
from kznav.eval.records import column_names
from kznav.eval.stats import holm_adjust


def fake_run(root: Path, name: str, success, seed=0, verify=None, stages=None, scenes=1, failed=()):
    """``stages``: failure stage per failed episode (default false_stop); episode i is in scene i % scenes;
    ``failed``: episode indices that never completed (failures.csv, no row)."""
    d = root / name
    d.mkdir(parents=True)
    row = {c: 0 for c in column_names()}
    rows = []
    for i, s in enumerate(success):
        if i in failed:
            continue
        st = "none" if s else (stages[i] if stages else "false_stop")
        rows.append({**row, "dataset": "hm3d", "split": "val", "episode_id": f"sc{i % scenes}:{i}",
                     "scene_id": f"sc{i % scenes}", "seed": seed,
                     "goal_category": "chair" if i % 2 else "bed", "word_type": "universal", "success": bool(s),
                     "spl": 0.5 * s, "soft_spl": 0.5, "dts": 0.0 if s else 1.0, "outcome": "success" if s else "false_stop",
                     "termination": "stop_called", "failure_stage": st,
                     "goal_ever_in_view": True, "goal_ever_detected": bool(i % 3), "goal_ever_verified_positive": bool(s),
                     "vlm_calls": 3, "vlm_calls_verify": 2 if verify else 0, "vlm_latency_total_s": 4.0})
    pd.DataFrame(rows, columns=column_names()).to_csv(d / "episodes.csv", index=False)
    if failed:
        pd.DataFrame([{"episode_id": f"sc{i % scenes}:{i}", "attempt": 2, "kind": "exception"} for i in failed]
                     ).to_csv(d / "failures.csv", index=False)
    if verify:
        with open(d / "events.jsonl", "w", encoding="utf-8") as f:
            for p, y in verify:
                f.write(json.dumps({"type": "verification", "called": True, "p_raw": p, "is_goal_by_content": y,
                                    "is_goal_privileged": not y}) + "\n")
    return d


def test_holm_adjust():
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = holm_adjust([0.2, np.nan, 0.01])
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.2, 0.02])


def test_arms_pool_seeds_and_contrasts_are_paired(tmp_path):
    a = paper.Arm.load("a", "A", [fake_run(tmp_path, "a0", [1, 1, 0, 1]), fake_run(tmp_path, "a1", [1, 0, 0, 1], seed=1)])
    b = paper.Arm.load("b", "B", [fake_run(tmp_path, "b0", [0, 1, 0, 0]), fake_run(tmp_path, "b1", [0, 0, 0, 1], seed=1)])
    main = paper.main_table([a, b])
    assert list(main["n"]) == [8, 8] and list(main["seeds"]) == [2, 2] and main["SR, % [95% CI]"][0].startswith("62.5")
    assert "Wilson" in main.attrs["note"]                                  # one scene: no cluster bootstrap
    ct = paper.contrast_table([a, b], [("a", "b")])
    s = ct[ct.metric == "success"].iloc[0]
    assert (s.episodes, s.seeds, s.scenes) == (4, 2, 1) and s["diff"] == pytest.approx(0.375)
    assert (s.only_a, s.only_b) == (3, 0)                                  # McNemar on pooled episode-seed pairs
    assert s.p == 1.0 and s.p_holm == pytest.approx(s.p)                    # one scene cannot be significant
    c = paper.Arm.load("c", "C", [fake_run(tmp_path, "c0", [1, 0, 1])])
    with pytest.raises(ValueError, match="not paired"):
        paper.contrast_table([a, c], [("a", "c")])


def test_breakdown_funnel_verifier_and_rendering(tmp_path):
    v = [(0.9, True), (0.2, False), (0.7, False), (0.1, False)]
    a = paper.Arm.load("a", "Model_A & co", [fake_run(tmp_path, "a", [1, 0, 1, 0], verify=v)])
    bd = paper.breakdown([a], "goal_category").iloc[0]
    assert bd["bed"] == "2/2 (100%)" and bd["chair"] == "0/2 (0%)"           # even ids are beds
    fn = paper.funnel([a]).iloc[0]
    assert fn["goal in view"] == 1.0 and fn["success"] == 0.5
    vt = paper.verifier_table([a]).iloc[0]
    assert vt["label"] == "content" and vt["sensitivity"] == "1/1" and vt["specificity"] == "2/3"
    tex = paper.to_latex(paper.main_table([a]), "SR (%) − test", "tab:x")
    assert r"Model\_A \& co" in tex and r"SR (\%) - test" in tex and r"\toprule" in tex
    md = paper.to_markdown(pd.DataFrame({"x": [0.12345, np.nan], "s": ["95.0", "a"]}))
    assert "0.123" in md and "—" in md and "95.0" in md


def test_scene_level_inference_failures_stages_and_gap(tmp_path):
    n = 60
    rng = np.random.default_rng(0)
    sa = (rng.random(n) < 0.6).astype(int)
    sb = np.where(rng.random(n) < 0.5, 0, sa)                              # b loses about half of a's successes
    st_b = ["query" if i % 2 else "not_seen" for i in range(n)]
    a = paper.Arm.load("a", "A", [fake_run(tmp_path, "a", sa, scenes=10, stages=["not_seen"] * n)])
    b = paper.Arm.load("b", "B", [fake_run(tmp_path, "b", sb, scenes=10, stages=st_b, failed=(5,))])
    main = paper.main_table([a, b])
    assert "scene bootstrap" in main.attrs["note"] and list(main.failed) == [0, 1] and list(main.scenes) == [10, 10]
    ct = paper.contrast_table([a, b], [("a", "b")])
    s = ct[ct.metric == "success"].iloc[0]
    assert s.episodes == n - 1 and s.scenes == 10 and s.dropped_failed == 1    # the failed episode left both arms
    assert s.p < 0.05 and s.ci_low > 0
    st = paper.stage_table([a, b]).set_index("Arm")
    assert st.loc["A", "SR"] + st.loc["A", "not_seen"] == pytest.approx(1.0)
    assert st.loc["B", "[language]"] == pytest.approx(st.loc["B", "query"])
    gd = paper.gap_decomposition([a, b], "a", "b").set_index("stage")
    gap = gd.loc["SR gap (a − b)", "contribution"]
    assert gap == pytest.approx(s["diff"])
    parts = gd.loc[["query", "not_seen"], "contribution"].sum()
    assert parts == pytest.approx(gap)                                       # stages partition the failures
    assert gd.loc["[language + grounding]", "share_of_gap"] == pytest.approx(gd.loc["query", "contribution"] / gap)


def test_interaction_contrast_and_holm_family(tmp_path):
    n = 40
    rng = np.random.default_rng(1)
    base = (rng.random(n) < 0.5).astype(int)
    q_kk = base                                                            # native model, Kazakh
    g_kk = np.where(np.arange(n) % 4 == 0, 0, base)                        # general model loses some in Kazakh
    q_en = base
    g_en = base                                                            # ... and nothing in English
    arms = [paper.Arm.load(k, k.upper(), [fake_run(tmp_path, k, v, scenes=8)])
            for k, v in (("q_kk", q_kk), ("g_kk", g_kk), ("q_en", q_en), ("g_en", g_en))]
    ct = paper.contrast_table(arms, [("q_kk", "g_kk"), ("q_kk", "g_kk", "q_en", "g_en")], metrics=("success",))
    simple, inter = ct.iloc[0], ct.iloc[1]
    assert inter.a == "(Q_KK − G_KK)" and inter.b == "(Q_EN − G_EN)"
    assert inter["diff"] == pytest.approx(simple["diff"]) == pytest.approx((q_kk - g_kk).mean())
    assert pd.isna(inter.only_a) and not pd.isna(simple.only_a)            # McNemar only for a − b
    assert list(ct.p_holm) == list(holm_adjust(ct.p.to_numpy()))
    with pytest.raises(ValueError, match="2 arms"):
        paper.contrast_table(arms, [("q_kk", "g_kk", "q_en")])


def test_sensitivity_variants(tmp_path):
    n = 30
    a = paper.Arm.load("a", "A", [fake_run(tmp_path, "a", [1] * n, scenes=6)])
    st_b = ["false_stop"] * n
    b = paper.Arm.load("b", "B", [fake_run(tmp_path, "b", [i % 2 for i in range(n)], scenes=6, stages=st_b)])
    b.episodes["command_id"] = [f"c{i % 3}" for i in range(n)]
    a.episodes["command_id"] = b.episodes["command_id"].to_numpy()
    by_cmd = paper.contrast_table([a, b], [("a", "b")], metrics=("success",), cluster="command_id").iloc[0]
    assert by_cmd.clusters == "3 commands" and by_cmd.p == pytest.approx(0.25)   # 3 clusters: min p = 2/8
    b.episodes.loc[0:3, "vlm_call_errors"] = 1
    no_err = paper.contrast_table([a, b], [("a", "b")], metrics=("success",), drop_model_errors=True).iloc[0]
    assert no_err.episodes == n - 4 and no_err.dropped_failed == 8
    # the stop label decides false stop (grounding) vs stop too far (navigation)
    b.episodes["stop_label_source"] = "content"
    b.episodes["stop_is_goal_by_position"] = True
    b.episodes["stop_candidate_is_goal"] = False
    b.episodes["termination"] = "stop_called"
    b.episodes["parse_ok"], b.episodes["hypothesis_match"] = True, True
    content = paper.gap_decomposition([a, b], "a", "b").set_index("stage")
    pos = paper.gap_decomposition([a, b], "a", "b", stop_label="position").set_index("stage")
    assert "false_stop" in content.index and "stop_too_far" in pos.index and "false_stop" not in pos.index
