"""Fitting the stop rule on dev runs (D-042): content labels from the loop, the fit, the tool's guards, fixed subsets."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kznav.language.base import ListSink, TeeSink
from kznav.language.goal import parse_command
from kznav.language.lexicon import Lexicon
from kznav.language.registry import create_backend
from kznav.perception.base import create_detector, create_segmenter
from kznav.policy.calibration import CalibrationArtifact, fit_stop_rule
from kznav.policy.objectnav import ObjectNavPolicy
from kznav.policy.verification import crop_bounds, crop_box, make_verifier
from kznav.run.config import compose_config
from kznav.run.episode import Executor, run_episode
from kznav.run.logging import RunLogger, read_jsonl
from kznav.sim.mock_env import MockEnv, room_with_boxes
from kznav.sim.objectnav_dataset import balanced_subset, pick_scenes, random_subset

from test_benchmark import dataset_episode
from test_smoke_pipeline import GOAL, TABLE, episode_for, small_cfg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import fit_calibration as tool  # noqa: E402


def overconfident(n=600, seed=0):
    """A verifier that is right 80% of the time but always says 0.02 or 0.98 (like Qwen on the pilot crops)."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.2
    right = rng.random(n) < 0.8
    says_yes = np.where(right, y, ~y)
    p = np.where(says_yes, 0.98, 0.02)
    groups = [f"ep{i // 6}" for i in range(n)]                  # six calls per episode
    return p, y, groups


def test_fit_reduces_ece_of_an_overconfident_verifier():
    p, y, g = overconfident()
    fit = fit_stop_rule(p, y, g, method="platt")
    m = fit.metrics
    assert m["ece_raw"] > 0.1 and m["ece_fit"] < m["ece_raw"] / 3 and m["oof_ece"] < m["ece_raw"]
    assert fit.base_rate == pytest.approx(y.mean()) and 0 < fit.threshold.threshold < 1
    assert m["oof_n"] == len(p) and m["oof_folds_skipped"] == 0


def test_folds_never_split_an_episode():
    from kznav.policy.calibration import _fold_of

    assert len({_fold_of("run/ep7", 5) for _ in range(3)}) == 1
    p, y, g = overconfident(n=60)
    fit = fit_stop_rule(p, y, g, method="isotonic", folds=3)
    assert fit.metrics["oof_n"] + 0 <= len(p)
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_stop_rule(p, y, g, method="beta")


def test_crop_bounds_match_the_crop():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    a, c, b, d = crop_bounds(rgb.shape, (10, 20, 30, 50), 0.25, 16)
    crop, area = crop_box(rgb, (10, 20, 30, 50), 0.25, 16)
    assert crop.shape[:2] == (d - c, b - a) and area == 600.0


def test_loop_logs_content_labels_for_every_verifier_call(tmp_path):
    cfg = small_cfg("habitat_agent")
    cfg = {**cfg, "backend": compose_config(["backend=mock", "backend.mock.verify_mode=constant",
                                             "backend.mock.verify_p=0.95"])["backend"]}
    env = MockEnv(cfg, room_with_boxes(6.0, extra=[GOAL, TABLE]))
    ep = episode_for(env, cfg)
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=TeeSink(calls))
    goal = parse_command(backend, calls, command_id="c0", text="Орындық қайда? Тауып бер.", lang="kk", category="chair",
                         word_type="universal", lexicon=Lexicon.load(REPO / "lexicon" / "hm3d_objectnav.yaml"))
    policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]), goal.hypotheses, env.K)
    policy.verifier = make_verifier(cfg, backend, goal.target, "kk", goal.hypotheses, goal.weights)
    logger = RunLogger(tmp_path)
    res, _ = run_episode(env, policy, Executor(env, cfg), ep, cfg, run_id="t", use_oracle_aux=True, logger=logger,
                         language=goal, calls=calls)
    logger.close()
    evs = [e for e in read_jsonl(tmp_path / "events.jsonl", "verification") if e["called"]]
    assert res.success and evs
    for e in evs:
        assert len(e["box_xyxy"]) == 4 and len(e["crop_xyxy"]) == 4
        assert 0.0 <= e["box_goal_fraction"] <= 1.0 and 0.0 <= e["crop_goal_fraction"] <= 1.0
    assert evs[-1]["is_goal_by_content"] is True and evs[-1]["is_goal_privileged"] is True   # the oracle box is the goal


# ---------------------------------------------------------------- the tool
def fake_run(root: Path, name: str, *, role="dev", backend="qolda", sha="abc", n_eps=12, seed=0):
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "role": role, "config": {"dataset": {"name": "hm3d_objectnav_v1", "split": "train"}},
        "backend": {"name": backend, "model_id": f"{backend}-4b", "prompts": {"verify_candidate": {"sha256": sha}}}}))
    p, y, _ = overconfident(n=n_eps * 6, seed=seed)
    with open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for i, (pi, yi) in enumerate(zip(p, y)):
            f.write(json.dumps({"type": "verification", "episode_id": f"{name}:{i // 6}", "called": True,
                                "p_raw": float(pi), "is_goal_by_content": bool(yi), "is_goal_privileged": bool(yi)}) + "\n")
    (d / "episodes.csv").write_text("episode_id\n" + "\n".join(f"{name}:{i}" for i in range(n_eps)) + "\n")
    return d


def run_tool(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["fit_calibration.py", *argv])
    tool.main()


def test_tool_writes_an_artifact_the_runner_accepts(tmp_path, monkeypatch):
    runs = [fake_run(tmp_path, "devA", seed=1), fake_run(tmp_path, "devB", seed=2)]
    out = tmp_path / "calib" / "qolda_dev.json"
    run_tool(monkeypatch, [*map(str, runs), "--out", str(out)])
    art = CalibrationArtifact.load(out)
    assert art.backend == "qolda" and art.prompt_sha256 == "abc" and art.n_samples == 144
    assert len(art.dev_episode_ids) == 24 and art.dev_split == "hm3d_objectnav_v1/train"
    assert art.extra["ece_fit"] < art.extra["ece_raw"] and art.calibrator["kind"] == "platt"
    assert out.with_suffix(".md").exists() and (out.parent / "qolda_dev_reliability.png").exists()
    with pytest.raises(ValueError):
        art.assert_disjoint_from(["devA:3"])


@pytest.mark.parametrize("bad", ["benchmark", "mixed_backend", "mixed_prompt"])
def test_tool_guards(tmp_path, monkeypatch, bad):
    a = fake_run(tmp_path, "a")
    b = fake_run(tmp_path, "b", role="benchmark" if bad == "benchmark" else "dev",
                 backend="qwen3vl_4b" if bad == "mixed_backend" else "qolda", sha="zzz" if bad == "mixed_prompt" else "abc")
    with pytest.raises(SystemExit):
        run_tool(monkeypatch, [str(a), str(b), "--out", str(tmp_path / "c.json")])


# ---------------------------------------------------------------- fixed subsets
def test_balanced_subset_and_scene_choice(tmp_path):
    eps = dataset_episode(tmp_path, starts=[(0, 0, i) for i in range(30)])
    ids = balanced_subset(eps, 10, seed=0)
    assert len(ids) == 10 and ids == balanced_subset(eps, 10, seed=0) and ids != balanced_subset(eps, 10, seed=1)
    assert ids == [e.episode_id for e in eps if e.episode_id in set(ids)]          # dataset order
    assert pick_scenes(eps, 1) == ["mockscene"] and balanced_subset(eps, 5, scenes=["other"]) == []


def test_random_subset_is_seeded_nested_and_in_dataset_order(tmp_path):
    eps = dataset_episode(tmp_path, starts=[(0, 0, i) for i in range(30)])
    ids = random_subset(eps, 12, seed=0)
    assert len(ids) == 12 and ids == random_subset(eps, 12, seed=0) and ids != random_subset(eps, 12, seed=1)
    assert ids == [e.episode_id for e in eps if e.episode_id in set(ids)]          # dataset order
    assert set(random_subset(eps, 5, seed=0)) <= set(ids) and len(random_subset(eps, 99)) == 30
