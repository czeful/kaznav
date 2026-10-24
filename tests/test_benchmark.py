"""Benchmark runner without HM3D (D-041): an ObjectNav v1 split written here in habitat-lab's layout, run on MockEnv."""
import csv
import dataclasses
import gzip
import json
import math
from pathlib import Path

import pytest

from kznav.language.base import ListSink
from kznav.policy.calibration import CalibrationArtifact
from kznav.run import benchmark as bm
from kznav.language.commands import load_commands
from kznav.run.benchmark import (BenchmarkRunner, Exclusion, LanguageRun, ViewPointCache, lookup_object,
                                 prepare_episode, view_point_params, wants_video)
from kznav.run.config import compose_config
from kznav.run.logging import RunLogger
from kznav.sim.episodes import GoalInstance, compute_view_points
from kznav.sim.mock_env import MockBox, MockEnv, room_with_boxes
from kznav.sim.objectnav_dataset import load_objectnav_v1, scene_stem, select_episodes, stable_index

from test_smoke_pipeline import small_cfg

REPO = Path(__file__).resolve().parents[1]
CHAIR = MockBox((1.6, 0.0, -2.3), (2.1, 0.9, -1.8), semantic_id=7, color=(200, 40, 40), category="chair")
TABLE = MockBox((-1.0, 0.0, -0.6), (0.4, 0.75, 0.6), semantic_id=3, category="table")
SCENE = "data/scene_datasets/hm3d/val/00800-mockscene/mockscene.basis.glb"


def goal(object_id="7", position=(1.85, 0.45, -2.05), vps=((1.85, 0.0, -1.2),)):
    return {"object_id": object_id, "object_name": f"chair_{object_id}", "object_category": "chair",
            "position": list(position), "radius": None, "room_id": None, "room_name": None,
            "view_points": [{"agent_state": {"position": list(p), "rotation": [0, 0, 0, 1]}, "iou": 0.5} for p in vps]}


def episode(start, scene=SCENE, category="chair"):
    return {"episode_id": "0", "scene_id": scene, "start_position": list(start), "start_rotation": [0, 0, 0, 1],
            "info": {"geodesic_distance": 5.0}, "goals": [], "object_category": category}


def write_split(root: Path, scenes: dict) -> Path:
    """habitat-lab layout: val.json.gz (category maps, no episodes) + content/<scene>.json.gz."""
    (root / "content").mkdir(parents=True, exist_ok=True)
    maps = {"category_to_task_category_id": {"chair": 0}, "category_to_scene_annotation_category_id": {"chair": 0}}
    with gzip.open(root / "val.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"episodes": [], **maps}, f)
    for name, (eps, goals) in scenes.items():
        with gzip.open(root / "content" / f"{name}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"episodes": eps, "goals_by_category": goals, **maps}, f)
    return root / "val.json.gz"


def mock_split(tmp_path, goals_list=None, starts=((-2.0, 0.0, 2.0), (-1.5, 0.0, 2.2), (-0.3, 0.0, 0.0))):
    key = "mockscene.basis.glb_chair"
    return write_split(tmp_path / "split", {"mockscene": ([episode(s) for s in starts],
                                                          {key: goals_list or [goal()]})})


def env_for(cfg, extra=()):
    return MockEnv(cfg, room_with_boxes(6.0, extra=[CHAIR, TABLE, *extra]))


# ---------------------------------------------------------------- loader
def test_loader_reads_content_files_and_names_episodes_uniquely(tmp_path):
    other = "data/scene_datasets/hm3d/val/00801-other/other.basis.glb"
    split = write_split(tmp_path / "s", {
        "mockscene": ([episode((0, 0, 0)), episode((1, 0, 1))], {"mockscene.basis.glb_chair": [goal()]}),
        "other": ([episode((2, 0, 2), scene=other)], {"other.basis.glb_chair": [goal("9")]})})
    eps = load_objectnav_v1(split, tmp_path / "scenes", dataset="hm3d_objectnav", split="val")
    assert [e.episode_id for e in eps] == ["mockscene:0", "mockscene:1", "other:0"]
    assert eps[0].scene_path == str(tmp_path / "scenes" / "hm3d/val/00800-mockscene/mockscene.basis.glb")
    assert eps[2].goals[0].object_id == "9" and eps[0].goals[0].view_points == [[1.85, 0.0, -1.2]]
    assert eps[0].start_yaw == pytest.approx(math.pi / 2)       # identity rotation faces -Z = +y_map
    assert eps[0].info["geodesic_distance"] == 5.0 and scene_stem(SCENE) == "mockscene"


def test_loader_accepts_goals_inside_episodes(tmp_path):
    e = episode((0, 0, 0))
    e["goals"] = [goal()]
    with gzip.open(tmp_path / "inline.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"episodes": [e]}, f)
    eps = load_objectnav_v1(tmp_path / "inline.json.gz", tmp_path, dataset="d", split="val")
    assert len(eps) == 1 and eps[0].category == "chair" and eps[0].goals[0].object_id == "7"


def test_selection_and_stable_choices(tmp_path):
    eps = load_objectnav_v1(mock_split(tmp_path, starts=[(0, 0, i) for i in range(6)]), tmp_path,
                            dataset="d", split="val")
    ids = [e.episode_id for e in eps]
    assert [e.episode_id for e in select_episodes(eps, subset=[ids[4], ids[1]])] == [ids[1], ids[4]]
    assert [e.episode_id for e in select_episodes(eps, limit=4, shard="1/2")] == [ids[1], ids[3]]
    with pytest.raises(KeyError):
        select_episodes(eps, subset=["nope:0"])
    with pytest.raises(ValueError):
        select_episodes(eps, shard="2/2")
    assert stable_index("mockscene:3|kk", 3) == stable_index("mockscene:3|kk", 3) < 3
    assert len({stable_index(f"x:{i}", 1000) for i in range(50)}) > 40


# ---------------------------------------------------------------- per-embodiment preparation
def dataset_episode(tmp_path, **kw):
    return load_objectnav_v1(mock_split(tmp_path, **kw), tmp_path, dataset="hm3d_objectnav", split="val")


def test_standard_agent_keeps_the_dataset_view_points(tmp_path):
    cfg = small_cfg("habitat_agent")
    env = env_for(cfg)
    ep = prepare_episode(env, dataset_episode(tmp_path)[0], cfg, objects=env.semantic_objects())
    assert ep.goals[0].view_points == [[1.85, 0.0, -1.2]] and ep.goals[0].semantic_id == 7
    assert ep.start_position == [-2.0, 0.0, 2.0] and ep.info["view_points_source"] == "dataset"
    assert math.isfinite(ep.start_geodesic_m) and ep.start_geodesic_m > 2


def test_wheeled_robot_recomputes_view_points_and_caches_them(tmp_path, monkeypatch):
    cfg = small_cfg("wheeled_vacuum")
    env = env_for(cfg)
    calls = []
    real = bm.compute_view_points
    monkeypatch.setattr(bm, "compute_view_points", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = ViewPointCache(tmp_path / "cache" / "mockscene.json", view_point_params(cfg))
    eps = dataset_episode(tmp_path)
    a = prepare_episode(env, eps[0], cfg, objects=env.semantic_objects(), cache=cache)
    b = prepare_episode(env, eps[1], cfg, objects=env.semantic_objects(), cache=cache)
    assert len(calls) == 1 and a.goals[0].view_points == b.goals[0].view_points
    assert a.goals[0].view_points and a.goals[0].view_points != [[1.85, 0.0, -1.2]]
    cache.save()
    again = ViewPointCache(tmp_path / "cache" / "mockscene.json", view_point_params(cfg))
    assert again.get("7") == a.goals[0].view_points
    changed = ViewPointCache(tmp_path / "cache" / "mockscene.json",
                             view_point_params(small_cfg("wheeled_vacuum", steps=10) | {
                                 "dataset": {**cfg["dataset"], "success_distance_m": 0.05}}))
    assert changed.get("7") is None                                     # another parameter set -> recompute


@pytest.mark.parametrize("case,reason", [
    ("unknown_id", "goal_not_in_scene"), ("wrong_position", "goal_not_in_scene"),
    ("start_in_box", "start_off_navmesh"), ("walled_off", "unreachable")])
def test_exclusions(tmp_path, case, reason):
    cfg = small_cfg("wheeled_vacuum")
    extra = ()
    kw = {}
    if case == "unknown_id":
        kw["goals_list"] = [goal("99")]
    elif case == "wrong_position":
        kw["goals_list"] = [goal(position=(-2.0, 0.4, 2.0))]
    elif case == "start_in_box":
        kw["starts"] = [(-0.3, 0.0, 0.0)]                                   # inside TABLE
    else:
        extra = (MockBox((-3.0, 0.0, -0.05), (3.0, 2.5, 0.05)),)            # wall across the room
    env = env_for(cfg, extra)
    ex = prepare_episode(env, dataset_episode(tmp_path, **kw)[0], cfg, objects=env.semantic_objects())
    assert isinstance(ex, Exclusion) and ex.reason == reason and ex.episode_id == "mockscene:0"


def test_lookup_by_annotation_id():
    objs = {12: bm.SemanticObject(12, "object_137", "chair", [0, 0, 0], [1, 1, 1])}
    assert lookup_object(objs, "12").semantic_id == 12 and lookup_object(objs, "137").semantic_id == 12
    assert lookup_object(objs, "5") is None


# ---------------------------------------------------------------- the runner
def runner_for(tmp_path, cfg, language=None, out_name="run"):
    out = tmp_path / out_name
    logger = RunLogger(out)
    return BenchmarkRunner(cfg, out, run_id="t", env_factory=lambda scene: env_for(cfg), logger=logger,
                           language=language, cache_root=tmp_path / "cache"), logger


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def agent_view_points(cfg):
    """What habitat-lab's dataset holds: the standard agent's view points around the object, on a grid."""
    inst = GoalInstance("7", 7, "chair", [1.85, 0.45, -2.05], list(CHAIR.lo), list(CHAIR.hi))
    return compute_view_points(env_for(cfg), inst, success_radius_m=1.0, grid_step_m=0.1, min_pixels=20, max_snap_m=0.05)


def test_runner_runs_excludes_and_resumes(tmp_path):
    cfg = small_cfg("habitat_agent")
    eps = dataset_episode(tmp_path, goals_list=[goal(vps=agent_view_points(cfg))])   # the third start is in TABLE ...
    eps[2].goals = [dataclasses.replace(eps[2].goals[0], object_id="99")]   # ... and its goal id is unknown
    r, logger = runner_for(tmp_path, cfg)
    assert r.run(eps) == {"run": 2, "excluded": 1, "skipped_done": 0, "failed": 0}
    logger.close()
    done = rows(r.episodes_csv)
    assert [d["episode_id"] for d in done] == ["mockscene:0", "mockscene:1"]
    assert all(d["outcome"] == "success" for d in done) and done[0]["cell_id"] == "nolang"
    assert json.loads(done[0]["notes"])["view_points_source"] == "dataset"
    assert rows(r.exclusions_csv)[0]["reason"] == "goal_not_in_scene"
    again, logger = runner_for(tmp_path, cfg)
    assert again.run(eps) == {"run": 0, "excluded": 0, "skipped_done": 3, "failed": 0}   # resume: nothing left
    logger.close()


def lang_cfg(**over):
    cfg = small_cfg("habitat_agent")
    return {**cfg, "backend": compose_config(["backend=mock", "backend.mock.verify_mode=constant",
                                              "backend.mock.verify_p=0.95"])["backend"], **over}


def test_runner_with_language_and_verification(tmp_path):
    cfg = lang_cfg()
    eps = dataset_episode(tmp_path, goals_list=[goal(vps=agent_view_points(cfg))])[:1]
    out = tmp_path / "lang"
    logger = RunLogger(out)
    language = LanguageRun(cfg, "kk", verify=True, call_sink=ListSink(), repo=REPO,
                           eval_episode_ids=[e.episode_id for e in eps])
    r = BenchmarkRunner(cfg, out, run_id="t", env_factory=lambda s: env_for(cfg), logger=logger, language=language)
    assert r.run(eps)["run"] == 1
    logger.close()
    row = rows(r.episodes_csv)[0]
    g = rows(out / "goals.csv")[0]
    assert row["cell_id"] == "mock-kk-verify" and row["outcome"] == "success" and row["command_lang"] == "kk"
    assert g["text"] == language.commands.command("mockscene:0", "kk", "chair")[1] and g["goal_mode"] == "multi"
    assert int(row["vlm_calls_verify"]) >= 1


def test_commands_follow_the_episode_id_and_name_the_object_per_language():
    cs = load_commands(REPO / "commands" / "hm3d_objectnav_v1.yaml", REPO)
    a = cs.command("scene:4", "kk", "chair")
    assert a == cs.command("scene:4", "kk", "chair") and "Орындық" in a[1]
    assert cs.object_name("kk_ru", "chair") == ("стул", "ru") and cs.object_name("kk", "bed") == ("төсек", "kk")


def calibration(tmp_path, dev_ids, backend="mock", sha=None):
    cfg = lang_cfg()
    from kznav.language.registry import create_backend

    sha = sha or create_backend(cfg["backend"]).verify_prompt.sha256
    art = CalibrationArtifact({"kind": "identity"}, {"threshold": 0.6}, backend, "mock-0", sha, "dev", dev_ids, 10, 0.5,
                              "2026-10-04T00:00:00Z")
    art.save(tmp_path / "calib.json")
    return {**cfg, "policy": {**cfg["policy"], "stop": {**cfg["policy"]["stop"],
                                                        "calibration_file": str(tmp_path / "calib.json")}}}


def test_calibration_must_come_from_other_episodes_and_the_same_model(tmp_path):
    ids = ["mockscene:0", "mockscene:1"]
    ok = LanguageRun(calibration(tmp_path, ["train_scene:0"]), "kk", verify=True, call_sink=ListSink(), repo=REPO,
                     eval_episode_ids=ids)
    assert ok.calibrated and ok.threshold == 0.6
    with pytest.raises(ValueError, match="evaluation episodes"):
        LanguageRun(calibration(tmp_path, ["mockscene:1"]), "kk", verify=True, call_sink=ListSink(), repo=REPO,
                    eval_episode_ids=ids)
    with pytest.raises(ValueError, match="was fitted for"):
        LanguageRun(calibration(tmp_path, ["train_scene:0"], backend="qolda"), "kk", verify=True, call_sink=ListSink(),
                    repo=REPO, eval_episode_ids=ids)


def test_video_subsample_is_deterministic():
    picked = [i for i in range(2000) if wants_video(f"s:{i}", 0.05)]
    assert picked == [i for i in range(2000) if wants_video(f"s:{i}", 0.05)] and 60 < len(picked) < 140


def test_loader_reads_only_requested_scene_files_and_scene_categories(tmp_path):
    from kznav.sim.objectnav_dataset import scene_categories, scenes_of_ids

    other = "data/scene_datasets/hm3d/val/00801-other/other.basis.glb"
    split = write_split(tmp_path / "s", {
        "mockscene": ([episode((0, 0, 0)), episode((1, 0, 1))], {"mockscene.basis.glb_chair": [goal()]}),
        "other": ([episode((2, 0, 2), scene=other, category="bed")], {"other.basis.glb_bed": [goal("9")]})})
    (tmp_path / "s" / "content" / "other.json.gz").write_bytes(b"not gzip")        # must not be read
    eps = load_objectnav_v1(split, tmp_path, dataset="d", split="val", scenes=["mockscene"])
    assert [e.episode_id for e in eps] == ["mockscene:0", "mockscene:1"]
    assert scenes_of_ids(["b:1", "a:2", "b:3"]) == ["b", "a"]
    (tmp_path / "s2").mkdir()
    split2 = write_split(tmp_path / "s2", {
        "mockscene": ([episode((0, 0, 0)), episode((1, 0, 1))], {"mockscene.basis.glb_chair": [goal()]}),
        "other": ([episode((2, 0, 2), scene=other, category="bed")], {"other.basis.glb_bed": [goal("9")]})})
    assert scene_categories(split2) == {"mockscene": {"chair": 2}, "other": {"bed": 1}}


def test_view_point_caches_of_different_cameras_coexist_and_legacy_files_are_read(tmp_path):
    a = ViewPointCache(tmp_path / "s.json", {"pitch": 0})
    b = ViewPointCache(tmp_path / "s.json", {"pitch": 20})
    a.put("7", [[1.0, 0.0, 2.0]])
    b.put("7", [[3.0, 0.0, 4.0]])
    a.save()
    b.save()
    assert a.path != b.path and len(list(tmp_path.glob("s_*.json"))) == 2
    assert ViewPointCache(tmp_path / "s.json", {"pitch": 0}).get("7") == [[1.0, 0.0, 2.0]]
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps({"params": {"pitch": 5}, "params_sha256": ViewPointCache(legacy, {"pitch": 5}).key,
                                  "view_points": {"9": [[0.0, 0.0, 0.0]]}}), encoding="utf-8")
    c = ViewPointCache(legacy, {"pitch": 5})
    assert c.get("9") == [[0.0, 0.0, 0.0]] and c.dirty                 # migrated on the next save
    c.save()
    assert c.path.exists() and c.path != legacy
