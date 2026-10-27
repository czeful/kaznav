"""Training crops for a light verifier (D-075 step 1): what the robot's detector proposes from the vacuum camera
(0.15 m) in train scenes used NOWHERE else, labelled by the simulator.

    bash tools/wsl.sh python tools/make_verifier_data.py --scenes 40 --frames 300 --out runs/verifier_data/v1

Scenes: HM3D ObjectNav v1 train scenes not in any dev split (``--exclude-split``, repeatable; dev4 is reserved before
this runs). Per scene, half of the frames from random navigable poses and half from poses 0.5-2.5 m from an instance
of one of the six goal categories, facing it. Per frame a target category is drawn (the faced one, or a random one);
the detector runs on its queries as the method's would - the bare English name and two of the hypotheses the
language models actually produced for it (harvested from the goal events of the finished runs). Every box of score
>= ``--min-score`` becomes a crop (the verifier's crop: box + 25% padding), labelled "goal" when at least 30% of the
box is pixels of an instance of the target category (``eval.funnel.verify_label_box_overlap``, D-042). Output: crops as
PNG under ``<out>/crops/<scene>/`` and one row per crop in ``<out>/crops.csv``; ``<out>/meta.json`` holds the scenes,
queries and parameters.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.policy.verification import crop_bounds  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.sim.objectnav_dataset import (content_files, read_id_list, scene_categories,  # noqa: E402
                                         stable_index)

CATEGORIES = ["bed", "chair", "plant", "sofa", "toilet", "tv_monitor"]
NAMES = {"bed": "bed", "chair": "chair", "plant": "plant", "sofa": "sofa", "toilet": "toilet", "tv_monitor": "television"}


def harvested_hypotheses() -> dict:
    """Hypotheses the language models wrote for each category in the finished runs (goal events), deduplicated."""
    out = defaultdict(set)
    for f in glob.glob(str(REPO / "runs/matrix/*/*/seed0/events.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if '"type": "goal"' not in line:
                    continue
                e = json.loads(line)
                if e.get("category") in NAMES and e.get("parse_ok"):
                    out[e["category"]].update(h for h in e.get("hypotheses") or [] if h.isascii())
    return {c: sorted(v) for c, v in out.items()}


def goals_of_scene(split_file: Path, stem: str):
    """(category -> object ids of its goal instances, the scene id as written in the file), from the content file."""
    for f in content_files(split_file):
        if Path(f).name.split(".")[0] != stem:
            continue
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            content = json.load(fh)
        out = {}
        for key, goals in (content.get("goals_by_category") or {}).items():
            cat = key.rsplit("_", 1)[-1] if not key.endswith("tv_monitor") else "tv_monitor"
            if cat in NAMES:
                out[cat] = [str(g["object_id"]) for g in goals]
        eps = content.get("episodes") or []
        return out, (eps[0]["scene_id"] if eps else None)
    return {}, None


def main() -> None:
    """Command-line entry point: harvest labelled detector crops from robot-camera frames in unused train scenes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hm3d_objectnav_v1_train")
    ap.add_argument("--embodiment", default="wheeled_vacuum_kinematic")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--frames", type=int, default=300, help="frames per scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--exclude-split", action="append", default=[])
    ap.add_argument("--detector-device", default="cuda", help="cpu: a smoke test while the GPU is busy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    cfg = compose_config([f"dataset={args.dataset}", f"embodiment={args.embodiment}",
                          "detector=mm_grounding_dino_base", f"detector.box_threshold={args.min_score}",
                          f"detector.device={args.detector_device}"])
    ds = cfg["dataset"]
    split_file = Path(ds["episodes_path"])
    used = {i.split(":")[0] for f in args.exclude_split for i in read_id_list(Path(f))}
    cats = scene_categories(split_file)
    pool = sorted(s for s, c in cats.items() if s not in used and sum(c.get(k, 0) for k in NAMES) > 0)
    scenes = sorted(pool, key=lambda s: stable_index(f"{args.seed}:{s}", 1 << 30))[: args.scenes]
    hyps = harvested_hypotheses()
    queries = {c: [NAMES[c]] + [h for h in hyps.get(c, []) if h.lower() != NAMES[c]] for c in NAMES}
    meta = {"args": vars(args), "excluded_scenes": sorted(used), "scenes": scenes, "queries": queries,
            "label_overlap": float(cfg["eval"]["funnel"]["verify_label_box_overlap"])}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(scenes)} scenes of {len(pool)} free ({len(used)} excluded); queries per category: "
          f"{ {c: len(q) for c, q in queries.items()} }", flush=True)

    from kznav.perception.base import create_detector
    from kznav.run.benchmark import lookup_object
    from kznav.sim.env import HabitatEnv, SceneSpec
    from kznav.sim.frames import habitat_to_map
    from kznav.sim.objectnav_dataset import resolve_scene_path

    detector = create_detector(cfg["detector"])
    thr = meta["label_overlap"]
    rows_path = out / "crops.csv"
    header = ("scene,frame,kind,target,query,score,x1,y1,x2,y2,goal_fraction,label,file\n")
    done_scenes = set()
    if rows_path.exists():                                  # resume: skip scenes already written
        import pandas as pd
        done_scenes = set(pd.read_csv(rows_path).scene.unique())
    else:
        rows_path.write_text(header, encoding="utf-8")
    for si, stem in enumerate(scenes):
        if stem in done_scenes:
            continue
        t0 = time.perf_counter()
        goal_ids, scene_id = goals_of_scene(split_file, stem)
        if scene_id is None:
            print(f"[{si + 1}/{len(scenes)}] {stem}: no content file, skipped", flush=True)
            continue
        scene_path = resolve_scene_path(scene_id, Path(ds["scene_root"]))
        env = HabitatEnv(cfg, SceneSpec(scene_path, ds.get("scene_dataset_config")), out / "work", seed=0)
        rng = random.Random(f"{args.seed}:{stem}")
        lines = []
        try:
            env.reset(env.random_navigable_point(), 0.0)
            objects = env.semantic_objects()
            sem = {c: {o.semantic_id for o in (lookup_object(objects, i) for i in ids) if o is not None}
                   for c, ids in goal_ids.items()}
            instances = [(c, objects[s]) for c, ids in sem.items() for s in ids]
            (out / "crops" / stem).mkdir(exist_ok=True)
            for fi in range(args.frames):
                if fi % 2 and instances:                    # facing an instance of a goal category
                    cat, obj = rng.choice(instances)
                    centre = np.asarray(obj.center)
                    for _ in range(20):
                        p = env.snap(centre + np.array([rng.uniform(-2.5, 2.5), 0.0, rng.uniform(-2.5, 2.5)]))
                        d = math.dist(habitat_to_map(p), habitat_to_map(centre))
                        if np.all(np.isfinite(p)) and 0.5 <= d <= 2.5:
                            break
                    else:
                        continue
                    (px, py), (ox, oy) = habitat_to_map(p), habitat_to_map(centre)
                    yaw, kind = math.atan2(oy - py, ox - px) + rng.uniform(-0.3, 0.3), "facing"
                else:
                    cat, p, yaw, kind = rng.choice(CATEGORIES), env.random_navigable_point(), rng.uniform(-math.pi, math.pi), "random"
                obs = env.render_at(p, yaw)
                rgb = np.asarray(obs["rgb"])[..., :3]
                semantic = obs.get("semantic")
                if semantic is None:
                    continue
                goal_px = np.isin(np.asarray(semantic), list(sem.get(cat, set())))
                qs = [queries[cat][0]] + rng.sample(queries[cat][1:], min(2, len(queries[cat]) - 1))
                for det in detector.detect(rgb, qs):
                    x1, y1, x2, y2 = (float(v) for v in det.box_xyxy)
                    region = goal_px[max(0, int(y1)):max(0, int(math.ceil(y2))), max(0, int(x1)):max(0, int(math.ceil(x2)))]
                    frac = float(region.mean()) if region.size else 0.0
                    a, c, b, d_ = crop_bounds(rgb.shape, det.box_xyxy, 0.25, 48)
                    name = f"{stem}_{fi:04d}_{len(lines):05d}.png"
                    from PIL import Image
                    Image.fromarray(np.ascontiguousarray(rgb[c:d_, a:b])).save(out / "crops" / stem / name)
                    lines.append(f"{stem},{fi},{kind},{cat},\"{qs[det.query_index]}\",{det.score:.4f},"
                                 f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{frac:.4f},{int(frac >= thr)},"
                                 f"crops/{stem}/{name}\n")
        finally:
            env.close()
        with open(rows_path, "a", encoding="utf-8") as fh:            # whole scene at once: safe to resume
            fh.writelines(lines)
        pos = sum(l.split(",")[-2] == "1" for l in lines)
        print(f"[{si + 1}/{len(scenes)}] {stem}: {len(lines)} crops, {pos} goal, {time.perf_counter() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
