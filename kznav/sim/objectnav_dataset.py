"""Reader for habitat-lab ObjectNav v1 episode files (HM3D ObjectNav v1, HM3D-OVON), without habitat-lab.

Layout (habitat-lab 0.2.4, ``ObjectNavDatasetV1``): the split file (``val/val.json.gz``) holds the
category maps and usually no episodes; ``content/<scene>.json.gz`` next to it holds each scene's
episodes and ``goals_by_category``, keyed ``"<scene file name>_<category>"``. Each goal has the
object's id in the scene's semantic annotation and its view points (agent states of the standard
0.88 m agent from which the object is visible).

habitat-lab names an episode by its index inside the content file, so the same id repeats across
scenes; here ``episode_id = "<scene stem>:<index>"`` is unique in the split and stable under any
subset, shard or resume. Nothing here needs the simulator; the robot-specific geometry (navmesh,
view points) is computed later (kznav.run.benchmark).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .frames import yaw_from_quat

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"     # habitat-lab strips this from scene ids
DEFAULT_CONTENT_PATH = "{data_path}/content/{scene}.json.gz"


@dataclass
class DatasetGoal:
    object_id: str                   # instance id in the scene's semantic annotation
    category: str
    position: List[float]
    view_points: List[List[float]]   # positions only (the standard agent's), habitat world frame
    name: str = ""
    room: Optional[str] = None


@dataclass
class DatasetEpisode:
    episode_id: str                  # "<scene stem>:<index in the content file>"
    dataset: str
    split: str
    scene_id: str                    # as written in the file
    scene_path: str                  # resolved against the scene root
    index: int
    category: str
    start_position: List[float]
    start_rotation_xyzw: List[float]
    goals: List[DatasetGoal]
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def scene_stem(self) -> str:
        return scene_stem(self.scene_id)

    @property
    def start_yaw(self) -> float:
        x, y, z, w = self.start_rotation_xyzw
        return yaw_from_quat((w, x, y, z))


def scene_stem(scene_id: str) -> str:
    """'hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb' -> 'TEEsavR23oF'."""
    return Path(scene_id).name.split(".")[0]


def _read_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def resolve_scene_path(scene_id: str, scene_root: Path) -> str:
    sid = scene_id[len(DEFAULT_SCENE_PATH_PREFIX):] if scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX) else scene_id
    return str(Path(scene_root) / sid)


def _goal(g: Dict[str, Any], category: str) -> DatasetGoal:
    vps = [[float(v) for v in vp["agent_state"]["position"]] for vp in g.get("view_points") or []]
    return DatasetGoal(object_id=str(g["object_id"]), category=str(g.get("object_category") or category),
                       position=[float(v) for v in g["position"]], view_points=vps,
                       name=str(g.get("object_name") or ""), room=g.get("room_id"))


def _episodes_from(content: Dict[str, Any], *, dataset: str, split: str, scene_root: Path) -> List[DatasetEpisode]:
    by_cat = content.get("goals_by_category") or {}
    parsed: Dict[str, List[DatasetGoal]] = {}      # goals_by_category entries are shared by all episodes of the key
    out: List[DatasetEpisode] = []                 # (as in habitat-lab): parse once - HM3D train has ~50k per scene
    for i, e in enumerate(content.get("episodes") or []):
        cat = e.get("object_category") or (e["goals"][0]["object_category"] if e.get("goals") else None)
        if cat is None:
            raise ValueError(f"episode {i} of {e.get('scene_id')} has no object_category")
        key = f"{os.path.basename(e['scene_id'])}_{cat}"
        if e.get("goals"):
            goals = [_goal(g, cat) for g in e["goals"]]
        else:
            if key not in parsed:
                if not by_cat.get(key):
                    raise ValueError(f"no goals for {key!r} (episode {i})")
                parsed[key] = [_goal(g, cat) for g in by_cat[key]]
            goals = list(parsed[key])               # own list, shared (read-only) DatasetGoal objects
        out.append(DatasetEpisode(
            episode_id=f"{scene_stem(e['scene_id'])}:{i}", dataset=dataset, split=split, scene_id=e["scene_id"],
            scene_path=resolve_scene_path(e["scene_id"], scene_root), index=i, category=str(cat),
            start_position=[float(v) for v in e["start_position"]],
            start_rotation_xyzw=[float(v) for v in e["start_rotation"]],
            goals=goals, info=dict(e.get("info") or {})))
    return out


def load_objectnav_v1(split_file: Path, scene_root: Path, *, dataset: str, split: str,
                      scenes: Optional[Sequence[str]] = None) -> List[DatasetEpisode]:
    """All episodes of a split (episodes inline in the split file, else every content file), scene by scene
    in sorted order. ``scenes``: only these scene stems - when the content files are named by scene stem (HM3D
    ObjectNav v1: ``content/<stem>.json.gz``) only those files are read; HM3D v1 train has ~50k episodes per scene."""
    split_file = Path(split_file)
    top = _read_json_gz(split_file)
    episodes = _episodes_from(top, dataset=dataset, split=split, scene_root=scene_root)
    if not episodes:
        files = content_files(split_file, top)
        if scenes is not None:
            wanted = set(scenes)
            named = [f for f in files if _stem_of_file(f) in wanted]
            if {_stem_of_file(f) for f in named} == wanted:
                files = named           # otherwise names differ from the scene ids: read all, filter below
        for f in files:     # content file names differ between releases; filter by the scene ids inside
            episodes += _episodes_from(_read_json_gz(f), dataset=dataset, split=split, scene_root=scene_root)
    if scenes is not None:
        episodes = [e for e in episodes if e.scene_stem in set(scenes)]
    return episodes


def content_files(split_file: Path, top: Optional[Dict[str, Any]] = None) -> List[Path]:
    split_file = Path(split_file)
    top = top if top is not None else _read_json_gz(split_file)
    pattern = str(top.get("content_scenes_path") or DEFAULT_CONTENT_PATH)
    content_dir = Path(pattern.split("{scene}")[0].format(data_path=str(split_file.parent)))
    ext = pattern.split("{scene}")[1]
    files = sorted(content_dir.glob(f"*{ext}"))
    if not files:
        raise FileNotFoundError(f"no episodes in {split_file} and no {ext} files in {content_dir}")
    return files


def _stem_of_file(f: Path) -> str:
    return f.name.split(".")[0]


def scene_categories(split_file: Path) -> Dict[str, Dict[str, int]]:
    """scene stem -> {category: episodes}, reading one content file at a time (no episode objects kept)."""
    split_file = Path(split_file)
    top = _read_json_gz(split_file)
    out: Dict[str, Dict[str, int]] = {}
    sources = [top] if top.get("episodes") else (_read_json_gz(f) for f in content_files(split_file, top))
    for data in sources:
        for e in data["episodes"]:
            c = out.setdefault(scene_stem(e["scene_id"]), {})
            c[e["object_category"]] = c.get(e["object_category"], 0) + 1
    return out


def scenes_of_ids(ids: Sequence[str]) -> List[str]:
    """Scene stems of episode ids ("<stem>:<index>"), in first-seen order."""
    return list(dict.fromkeys(str(i).split(":")[0] for i in ids))


def read_id_list(path: Path) -> List[str]:
    """Episode ids, one per line; blank lines and '#' comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]


def select_episodes(episodes: Sequence[DatasetEpisode], *, subset: Optional[Sequence[str]] = None,
                    limit: Optional[int] = None, shard: Optional[str] = None) -> List[DatasetEpisode]:
    """Subset (ids) -> limit (the first N) -> shard ("i/n": every n-th episode starting at i). Order is kept."""
    out = list(episodes)
    if subset is not None:
        want = set(subset)
        missing = want - {e.episode_id for e in out}
        if missing:
            raise KeyError(f"{len(missing)} subset ids are not in the split, e.g. {sorted(missing)[:3]}")
        out = [e for e in out if e.episode_id in want]
    if limit is not None:
        out = out[: int(limit)]
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        if not 0 <= i < n:
            raise ValueError(f"shard must be i/n with 0 <= i < n, got {shard!r}")
        out = out[i::n]
    return out


def stable_index(key: str, n: int) -> int:
    """Deterministic choice in [0, n) from a string (e.g. the command template of an episode): unlike a
    round-robin counter it does not change with the subset, the shard or a resume."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % int(n)


# ---------------------------------------------------------------------------- fixed subsets (tools/make_split.py)
def _shuffled(items: Sequence[Any], seed: int, salt: str) -> List[Any]:
    return sorted(items, key=lambda x: hashlib.sha256(f"{seed}|{salt}|{x}".encode("utf-8")).hexdigest())


def balanced_subset(episodes: Sequence[DatasetEpisode], per_category: int, *, seed: int = 0,
                    scenes: Optional[Sequence[str]] = None) -> List[str]:
    """Up to ``per_category`` episodes of every category (optionally from some scenes only), chosen by a seeded
    hash of the id (independent of the file order). Returned in dataset order."""
    pool = [e for e in episodes if scenes is None or e.scene_stem in set(scenes)]
    picked = set()
    for cat in sorted({e.category for e in pool}):
        ids = [e.episode_id for e in pool if e.category == cat]
        picked |= set(_shuffled(ids, seed, cat)[:per_category])
    return [e.episode_id for e in pool if e.episode_id in picked]


def random_subset(episodes: Sequence[DatasetEpisode], n: int, *, seed: int = 0) -> List[str]:
    """``n`` episodes of the whole split, not stratified: the categories keep the split's own mix (D-064). Seeded hash
    of the id as in ``balanced_subset``; a smaller ``n`` with the same seed gives a subset. Returned in dataset order."""
    picked = set(_shuffled([e.episode_id for e in episodes], seed, "random")[:n])
    return [e.episode_id for e in episodes if e.episode_id in picked]


def pick_scenes(episodes: Any, n: int, *, seed: int = 0) -> List[str]:
    """``n`` scenes, seeded; among scenes ordered by how many categories they cover (most first). ``episodes``:
    the episodes, or a ``scene_categories`` mapping (scene -> categories) for splits too large to load."""
    cats: Dict[str, set] = {}
    if isinstance(episodes, dict):
        cats = {k: set(v) for k, v in episodes.items()}
    else:
        for e in episodes:
            cats.setdefault(e.scene_stem, set()).add(e.category)
    order = _shuffled(sorted(cats), seed, "scenes")
    order.sort(key=lambda s: -len(cats[s]))          # stable: ties keep the seeded order
    return sorted(order[:n])
