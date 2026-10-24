"""Benchmark episodes (HM3D ObjectNav, HM3D-OVON) on the evaluated embodiment, and the runner around run_episode.

Dataset episodes are made for habitat-lab's 0.88 m agent. Per run (D-001, D-041):

* goal instances are looked up in the scene's semantic annotation (semantic id, AABB) - the funnel columns
  and the privileged is-goal checks need them. A dataset goal position farther than
  ``max_goal_center_offset_m`` from the instance's AABB centre means the id mapping is wrong: that
  instance is dropped, and an episode left without instances is excluded;
* view points: the dataset's when the embodiment uses the dataset navmesh (the standard agent, table (a),
  comparable with the literature), otherwise recomputed for the embodiment's camera and navmesh
  (``compute_view_points``, cached per scene and parameter set);
* start: the dataset's; the wheeled robot needs it on its own navmesh, within ``max_start_snap_m``;
* the start geodesic distance is measured on the embodiment's navmesh.

An episode that fails a step is not run: it goes to ``exclusions.csv`` with the reason, and every report
counts the exclusions. A run directory can be resumed: episodes already in ``episodes.csv`` or
``exclusions.csv`` are skipped. A resumed episode runs in a new simulator process, so it is reproducible
per episode but not bit-identical to an uninterrupted run (Bullet state, D-036); the manifest lists resumes.

Failures (D-046): an exception during an episode (model server unreachable, simulator error, a bug) does not end
the run. The attempt goes to ``failures.csv`` (traceback in ``failures/``), the scene's simulator is rebuilt and
the run continues; ``run.max_consecutive_failures`` failures in a row abort it (the server is down). A resume
retries failed episodes until ``run.max_episode_attempts``; an episode that keeps failing stays in
``failures.csv`` and is never written to ``episodes.csv`` as if it had been run. Reports count failures; the
analysis plan says what happens to their pairs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..eval.records import column_names
from ..language.base import ListSink, TeeSink, TransportError
from ..language.commands import WORD_LANG, load_commands  # noqa: F401 - WORD_LANG re-exported
from ..language.goal import LanguageGoal, make_goal_cache, with_target_name
from ..language.lexicon import Lexicon
from ..language.registry import create_backend
from ..perception.base import create_detector, create_segmenter, create_value_encoder
from ..perception.camera import Intrinsics
from ..policy.calibration import CalibrationArtifact
from ..policy.objectnav import ObjectNavPolicy
from ..policy.verification import load_stop_calibration, make_verifier
from ..sim.episodes import GoalInstance, ObjectNavEpisode, SemanticObject, compute_view_points
from ..sim.objectnav_dataset import DatasetEpisode, stable_index
from .episode import Executor, run_episode

EXCLUSION_REASONS = ("goal_not_in_scene", "no_view_points", "start_off_navmesh", "unreachable", "start_at_goal")
FAILURE_COLUMNS = ("episode_id", "scene_id", "category", "attempt", "kind", "error_type", "message", "traceback_file",
                   "utc")


@dataclass
class Exclusion:
    episode_id: str
    scene_id: str
    category: str
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------- geometry per embodiment
def uses_dataset_view_points(cfg: Mapping[str, Any]) -> bool:
    """The dataset's view points and navmesh belong together: both are the standard agent's (D-001)."""
    return str(cfg["embodiment"]["navmesh"]["source"]) == "dataset"


def view_point_params(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Everything the recomputed view points depend on; its hash keys the cache."""
    emb, vp = cfg["embodiment"], cfg["eval"]["view_points"]
    return {"embodiment": emb["name"], "navmesh": emb["navmesh"], "rgbd": emb["sensors"]["rgbd"],
            "view_points": vp, "success_distance_m": float(cfg["dataset"]["success_distance_m"])}


class ViewPointCache:
    """Recomputed view points of one scene, keyed by object id; invalid as soon as any parameter changes. One file
    per (scene, parameter set): ``<scene>_<params hash>.json`` next to ``path``, so runs with different cameras
    (e.g. the pitch study) never overwrite each other's cache; a legacy ``<scene>.json`` with matching parameters is
    still read."""

    def __init__(self, path: Path, params: Mapping[str, Any]):
        self.params = dict(params)
        self.key = hashlib.sha256(json.dumps(self.params, sort_keys=True, default=str).encode()).hexdigest()
        legacy = Path(path)
        self.path = legacy.with_name(f"{legacy.stem}_{self.key[:12]}{legacy.suffix}")
        self.view_points: Dict[str, List[List[float]]] = {}
        self.dirty = False
        for f in (self.path, legacy):
            if f.exists():
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("params_sha256") == self.key:
                    self.view_points = data["view_points"]
                    self.dirty = f is legacy          # migrate to the per-parameter file on the next save
                    break

    def get(self, object_id: str) -> Optional[List[List[float]]]:
        return self.view_points.get(str(object_id))

    def put(self, object_id: str, vps: List[List[float]]) -> None:
        self.view_points[str(object_id)] = vps
        self.dirty = True

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"params": self.params, "params_sha256": self.key,
                                       "view_points": self.view_points}, default=str), encoding="utf-8")
            tmp.replace(self.path)                        # atomic: a crash never leaves a half-written cache
            self.dirty = False


def lookup_object(objects: Mapping[int, SemanticObject], object_id: str) -> Optional[SemanticObject]:
    """By semantic id first, then by the annotation's own id ("137" or "..._137")."""
    try:
        sid: Optional[int] = int(object_id)
    except (TypeError, ValueError):
        sid = None
    if sid is not None and sid in objects:
        return objects[sid]
    for o in objects.values():
        if o.object_id == str(object_id) or o.object_id.endswith(f"_{object_id}"):
            return o
    return None


def prepare_episode(env: Any, de: DatasetEpisode, cfg: Mapping[str, Any], *, objects: Mapping[int, SemanticObject],
                    cache: Optional[ViewPointCache] = None) -> Union[ObjectNavEpisode, Exclusion]:
    bm, vp = cfg["eval"]["benchmark"], cfg["eval"]["view_points"]
    dataset_vps = uses_dataset_view_points(cfg)

    def excluded(reason: str, detail: str = "") -> Exclusion:
        return Exclusion(de.episode_id, de.scene_id, de.category, reason, detail)

    goals: List[GoalInstance] = []
    problems: List[str] = []
    for g in de.goals:
        obj = lookup_object(objects, g.object_id)
        if obj is None:
            problems.append(f"{g.object_id}: not in the semantic annotation")
            continue
        off = math.dist(obj.center, g.position)
        if off > float(bm["max_goal_center_offset_m"]):
            problems.append(f"{g.object_id}: dataset position {off:.2f} m from the instance's AABB centre")
            continue
        inst = GoalInstance(object_id=g.object_id, semantic_id=obj.semantic_id, category=de.category,
                            position=obj.center, aabb_min=list(obj.aabb_min), aabb_max=list(obj.aabb_max))
        if dataset_vps:
            inst.view_points = [list(p) for p in g.view_points]
        else:
            cached = cache.get(g.object_id) if cache is not None else None
            if cached is None:
                cached = compute_view_points(env, inst, success_radius_m=float(vp["success_radius_m"]),
                                             grid_step_m=float(vp["grid_step_m"]), min_pixels=int(vp["min_pixels"]),
                                             max_snap_m=float(vp["max_snap_m"]),
                                             max_height_above_floor_m=float(vp.get("max_object_height_above_floor_m", 2.0)),
                                             success_distance_m=float(cfg["dataset"]["success_distance_m"]))
                if cache is not None:
                    cache.put(g.object_id, cached)
            inst.view_points = cached
        goals.append(inst)
    if not goals:
        return excluded("goal_not_in_scene", "; ".join(problems[:3]))
    vps = [p for g in goals for p in g.view_points]
    if not vps:
        return excluded("no_view_points", f"{len(goals)} instance(s)")
    start = np.asarray(de.start_position, dtype=np.float64)
    snap_m = 0.0
    if not dataset_vps:
        snapped = env.snap(start)
        if not np.all(np.isfinite(snapped)):
            return excluded("start_off_navmesh", "no navmesh point near the start")
        snap_m = float(math.hypot(snapped[0] - start[0], snapped[2] - start[2]))
        if snap_m > float(bm["max_start_snap_m"]) or abs(snapped[1] - start[1]) > float(bm["max_start_height_m"]):
            return excluded("start_off_navmesh", f"nearest navmesh point {snap_m:.2f} m away")
        start = snapped
    l = float(env.geodesic_distance(start, vps))
    if not math.isfinite(l):
        return excluded("unreachable", f"{len(vps)} view points")
    if l <= 0.0:
        return excluded("start_at_goal")
    return ObjectNavEpisode(
        episode_id=de.episode_id, scene_id=de.scene_id, dataset=de.dataset, split=de.split,
        start_position=[float(v) for v in start], start_yaw=de.start_yaw, goal_category=de.category, goals=goals,
        start_geodesic_m=l, embodiment=str(cfg["embodiment"]["name"]),
        info={"dataset_geodesic_m": de.info.get("geodesic_distance"), "start_snap_m": snap_m,
              "goal_problems": problems, "view_points_source": "dataset" if dataset_vps else "recomputed"})


# ---------------------------------------------------------------------------- the language side
class LanguageRun:
    """The VLM side of a run: backend, goal parsing (with its ablation mode), commands, verifier factory."""

    def __init__(self, cfg: Mapping[str, Any], lang: str, *, verify: bool, call_sink: Any, repo: Path,
                 eval_episode_ids: Sequence[str]):
        self.cfg, self.lang, self.verify = cfg, lang, verify
        self.calls = ListSink()
        self.backend = create_backend(cfg["backend"], sink=TeeSink(call_sink, self.calls))
        self.commands = load_commands(repo / (cfg["dataset"].get("commands") or f"commands/{cfg['dataset']['name']}.yaml"),
                                      repo, role=str(cfg["dataset"]["role"]))
        self.goals = make_goal_cache(cfg["goal"], self.backend, self.calls, self.commands.lexicon,
                                     translator_sink=call_sink)
        _, self.threshold, self.base_rate, self.calibrated = load_stop_calibration(cfg["policy"])
        vv = cfg["policy"]["verification"]
        # D-075: "vlm" asks the language model; "siglip_head" / "siglip_zero" answer locally (local_verifier)
        self.verifier_backend = str(vv.get("backend", "vlm"))
        self.head_file = vv.get("head_file")
        self._local, self._local_encoder = None, None
        calib = cfg["policy"]["stop"].get("calibration_file")
        if verify and calib:
            art = CalibrationArtifact.load(Path(calib))
            art.assert_disjoint_from(eval_episode_ids)
            if self.verifier_backend == "vlm":
                if art.backend != cfg["backend"]["name"] or art.prompt_sha256 != self.backend.verify_prompt.sha256:
                    raise ValueError(f"calibration {calib} was fitted for {art.backend} / prompt {art.prompt_sha256[:12]}, "
                                     f"this run uses {cfg['backend']['name']} / {self.backend.verify_prompt.sha256[:12]}")
            else:
                sha = hashlib.sha256(Path(self.head_file).read_bytes()).hexdigest()                     if self.verifier_backend == "siglip_head" else None
                if art.backend != self.verifier_backend or (sha is not None and art.prompt_sha256 != sha):
                    raise ValueError(f"calibration {calib} was fitted for {art.backend}; this run verifies with "
                                     f"{self.verifier_backend}" + (f" (head {self.head_file})" if sha else ""))

    def warmup(self) -> None:
        self.backend.warmup(int(self.cfg["backend"].get("warmup_calls", 0)))

    def goal_for(self, ep: ObjectNavEpisode, word_type: str) -> LanguageGoal:
        cid, text = self.commands.command(ep.episode_id, self.lang, ep.goal_category)
        word, word_lang = self.commands.object_name(self.lang, ep.goal_category)
        self.backend.context.episode_id = ep.episode_id
        return self.goals.get(command_id=cid, text=text, lang=self.lang, category=ep.goal_category,
                              word_type=word_type, word=word, word_lang=word_lang)

    def attach(self, policy: ObjectNavPolicy, lg: LanguageGoal) -> None:
        texts, weights = list(lg.hypotheses), list(lg.weights)
        if self.cfg["perception"]["detection"].get("add_target_name", False):      # D-072
            texts, weights = with_target_name(texts, weights, lg.target_en)
        policy.queries = texts
        policy.query_weights = weights if texts else None
        policy.rooms = list(lg.rooms)
        policy.verifier = make_verifier(self.cfg, self._verifier_backend(policy), lg.target, lg.effective_lang, texts,
                                        weights, lg.target_en) if self.verify and lg.parse_ok else None

    def _verifier_backend(self, policy: ObjectNavPolicy) -> Any:
        """The language model, or (D-075) the local SigLIP verifier on the policy's own value encoder."""
        if self.verifier_backend == "vlm":
            return self.backend
        if self._local is None or self._local_encoder is not policy.value_encoder:
            from ..policy.local_verifier import SiglipVerifierBackend

            self._local = SiglipVerifierBackend(policy.value_encoder, self.verifier_backend, self.head_file)
            self._local_encoder = policy.value_encoder
        return self._local

    def manifest_extra(self) -> Dict[str, Any]:
        t = self.goals.translator
        return {"backend": self.backend.describe_config(), "goal": self.cfg["goal"], "lang": self.lang,
                "verify": self.verify, "translator": t.describe_config() if t is not None else None,
                "commands": self.commands.path, "commands_kind": self.commands.kind,
                "commands_review_status": self.commands.review_status,
                "stop_threshold": self.threshold, "base_rate": self.base_rate, "calibrated": self.calibrated}


# ---------------------------------------------------------------------------- the runner
def _csv_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {r["episode_id"] for r in csv.DictReader(f)}


def _append_csv(path: Path, row: Mapping[str, Any], columns: Sequence[str]) -> None:
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in columns})


def read_failures(path: Path) -> Dict[str, int]:
    """episode id -> number of failed attempts in a failures.csv."""
    if not Path(path).exists():
        return {}
    out: Dict[str, int] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["episode_id"]] = out.get(r["episode_id"], 0) + 1
    return out


class RunAborted(RuntimeError):
    """Too many consecutive episode failures: the infrastructure is down, not the method."""


def wants_video(episode_id: str, fraction: float) -> bool:
    """Deterministic subsample: the same episodes get a video in every arm, subset and resume."""
    return stable_index(f"{episode_id}|video", 1_000_000) < fraction * 1_000_000


class BenchmarkRunner:
    """Runs dataset episodes scene by scene; ``env_factory(scene_path)`` builds the simulator (HabitatEnv, or
    MockEnv in tests). Outputs in ``out``: episodes.csv, exclusions.csv, goals.csv, events.jsonl, vlm_calls.jsonl."""

    def __init__(self, cfg: Mapping[str, Any], out: Path, *, run_id: str, env_factory: Callable[[str], Any],
                 logger: Any, language: Optional[LanguageRun] = None,
                 video_factory: Optional[Callable[[Path, ObjectNavEpisode, Optional[LanguageGoal]], Any]] = None,
                 cache_root: Optional[Path] = None, word_types: Optional[Mapping[str, str]] = None,
                 progress: Optional[Callable[[str], None]] = None):
        self.cfg, self.out, self.run_id = cfg, Path(out), run_id
        self.env_factory, self.logger, self.language = env_factory, logger, language
        self.video_factory = video_factory
        self.cache_root = cache_root
        self.word_types = dict(word_types or {})
        self.progress = progress or (lambda s: None)
        self.cell_id = "nolang" if language is None else (
            f"{cfg['backend']['name']}-{language.lang}"
            + ("" if cfg["goal"]["name"] == "multi" else f"-{cfg['goal']['name']}")
            + ("-verify" if language.verify else "-noverify"))
        rgbd = cfg["embodiment"]["sensors"]["rgbd"]
        self.K = Intrinsics.from_hfov(float(rgbd["hfov_deg"]), int(rgbd["resolution_hw"][1]), int(rgbd["resolution_hw"][0]))
        self.max_attempts = int(cfg["run"].get("max_episode_attempts", 2))
        self.max_consecutive = int(cfg["run"].get("max_consecutive_failures", 3))
        self.consecutive_failures = 0
        self.counts = {"run": 0, "excluded": 0, "skipped_done": 0, "failed": 0}

    @property
    def episodes_csv(self) -> Path:
        return self.out / "episodes.csv"

    @property
    def exclusions_csv(self) -> Path:
        return self.out / "exclusions.csv"

    @property
    def failures_csv(self) -> Path:
        return self.out / "failures.csv"

    def done_ids(self) -> set:
        """Finished, excluded, or failed ``max_episode_attempts`` times (not retried again)."""
        given_up = {e for e, n in read_failures(self.failures_csv).items() if n >= self.max_attempts}
        return _csv_ids(self.episodes_csv) | _csv_ids(self.exclusions_csv) | given_up

    def run(self, episodes: Sequence[DatasetEpisode]) -> Dict[str, int]:
        done = self.done_ids()
        by_scene: Dict[str, List[DatasetEpisode]] = {}
        for de in episodes:
            if de.episode_id in done:
                self.counts["skipped_done"] += 1
                continue
            by_scene.setdefault(de.scene_path, []).append(de)
        for scene_path, todo in by_scene.items():
            self._run_scene(scene_path, todo)
        return dict(self.counts)

    def _record_failure(self, de: DatasetEpisode, exc: BaseException) -> None:
        attempt = read_failures(self.failures_csv).get(de.episode_id, 0) + 1
        tb_dir = self.out / "failures"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_file = tb_dir / f"{de.episode_id.replace(':', '_')}_attempt{attempt}.txt"
        tb_file.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
        kind = "model_server" if isinstance(exc, TransportError) else "exception"
        _append_csv(self.failures_csv, {
            "episode_id": de.episode_id, "scene_id": de.scene_id, "category": de.category, "attempt": attempt,
            "kind": kind, "error_type": type(exc).__name__, "message": str(exc)[:300],
            "traceback_file": tb_file.relative_to(self.out).as_posix(),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, FAILURE_COLUMNS)
        self.counts["failed"] += 1
        self.consecutive_failures += 1
        self.progress(f"{de.episode_id}: FAILED attempt {attempt} ({type(exc).__name__}: {str(exc)[:120]})")
        if self.logger is not None:
            self.logger.event("episode_failed", episode_id=de.episode_id, step=-1, kind=kind,
                              error=f"{type(exc).__name__}: {str(exc)[:300]}")
            self.logger.attempt = None
        if self.consecutive_failures >= self.max_consecutive:
            raise RunAborted(f"{self.consecutive_failures} episodes failed in a row (last: {type(exc).__name__}: "
                             f"{str(exc)[:200]}); fix the cause and resume the run") from exc

    def _make_policy(self) -> ObjectNavPolicy:
        cfg = self.cfg
        venc = create_value_encoder(cfg["value_encoder"]) if cfg["value_encoder"].get("enabled", False) else None
        return ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]),
                               queries=[], intrinsics=self.K, seed=int(cfg["run"]["seed"]), value_encoder=venc)

    def _run_scene(self, scene_path: str, todo: Sequence[DatasetEpisode]) -> None:
        cfg = self.cfg
        cache = None
        if not uses_dataset_view_points(cfg) and self.cache_root is not None:
            cache = ViewPointCache(self.cache_root / f"{todo[0].scene_stem}.json", view_point_params(cfg))
        env = None
        try:
            for de in todo:
                try:
                    if env is None:          # (re)built after a failure: the simulator may be in a bad state
                        env = self.env_factory(scene_path)
                        objects = env.semantic_objects()
                        policy = self._make_policy()
                        executor = Executor(env, cfg)
                    ep = prepare_episode(env, de, cfg, objects=objects, cache=cache)
                    if isinstance(ep, Exclusion):
                        _append_csv(self.exclusions_csv, asdict(ep), list(asdict(ep)))
                        self.counts["excluded"] += 1
                        self.progress(f"{de.episode_id}: excluded ({ep.reason}: {ep.detail})")
                        continue
                    self._run_one(env, policy, executor, ep)
                    self.consecutive_failures = 0
                except Exception as exc:          # noqa: BLE001 - recorded with its traceback, the run goes on
                    if env is not None:
                        try:
                            env.close()
                        except Exception:         # noqa: BLE001
                            pass
                    env = None
                    self._record_failure(de, exc)
        finally:
            if cache is not None:
                cache.save()
            if env is not None:
                env.close()

    def _run_one(self, env: Any, policy: ObjectNavPolicy, executor: Executor, ep: ObjectNavEpisode) -> None:
        cfg, lang = self.cfg, self.language
        word_type = self.word_types.get(ep.goal_category, "universal")
        attempt = read_failures(self.failures_csv).get(ep.episode_id, 0) + 1
        if self.logger is not None:
            self.logger.attempt = f"{ep.episode_id}#{attempt}"
        lg = None
        if lang is not None:
            lg = lang.goal_for(ep, word_type)
            lang.attach(policy, lg)
            self.logger.event("goal", episode_id=ep.episode_id, step=-1, **lg.summary())
        else:
            policy.queries, policy.query_weights, policy.verifier = [ep.goal_category], None, None
        rec = None
        if self.video_factory is not None and wants_video(ep.episode_id, float(cfg["run"]["video"]["fraction"])):
            rec = self.video_factory(self.out / f"video_{ep.episode_id.replace(':', '_')}.mp4", ep, lg)
        try:
            res, _ = run_episode(env, policy, executor, ep, cfg, run_id=self.run_id, cell_id=self.cell_id,
                                 backend=cfg["backend"]["name"] if lang else "none",
                                 model_id=cfg["backend"]["model_id"] if lang else "none",
                                 command_lang=lang.lang if lang else "en", command_text=lg.text if lg else "",
                                 command_id=lg.command_id if lg else "", word_type=word_type,
                                 use_oracle_aux=cfg["detector"]["type"] == "oracle_semantic", on_frame=rec,
                                 logger=self.logger, language=lg, calls=lang.calls if lang else None)
        finally:
            if rec is not None:
                rec.close()
        row = res.to_row()
        notes = {k: ep.info[k] for k in ("dataset_geodesic_m", "start_snap_m", "view_points_source")}
        notes["attempt"] = attempt
        row["notes"] = json.dumps(notes)
        if lg is not None:
            _append_csv(self.out / "goals.csv", {"episode_id": ep.episode_id, **lg.summary()},
                        ["episode_id"] + list(lg.summary()))
        _append_csv(self.episodes_csv, row, column_names())
        if self.logger is not None:
            self.logger.event("episode_done", episode_id=ep.episode_id, step=-1, outcome=res.outcome,
                              failure_stage=res.failure_stage)
            self.logger.attempt = None
        self.counts["run"] += 1
        self.progress(f"{ep.episode_id}: {res.outcome:13s} SR={int(res.success)} SPL={res.spl:.3f} "
                      f"l={res.start_geodesic:.2f} steps={res.num_decision_steps} vlm={res.vlm_calls}")


def word_types_from(lexicon: Lexicon) -> Dict[str, str]:
    return {name: c.word_type for name, c in lexicon.categories.items()}


def iter_scene_paths(episodes: Iterable[DatasetEpisode]) -> List[str]:
    seen: List[str] = []
    for e in episodes:
        if e.scene_path not in seen:
            seen.append(e.scene_path)
    return seen
