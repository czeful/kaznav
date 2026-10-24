"""First-contact check of an ObjectNav split on its scenes (D-041 open point): does every goal's object_id find its
instance in the scene's semantic annotation, how far is the dataset's goal position from that instance, and which
episodes each embodiment can run.

    bash tools/wsl.sh python tools/check_hm3d.py --dataset hm3d_objectnav_v1_minival --per-scene 20 \\
        --embodiments habitat_agent,wheeled_vacuum --out runs/hm3d_check

Per scene: semantic instances, goal instances found by semantic id / by annotation id / not found, offset of the
dataset goal position from the instance's AABB centre (median, max), categories of the found instances (should
match the goal category), then for each embodiment the episodes prepared or excluded by reason (kznav.run.benchmark
.prepare_episode) and the seconds it took. Output in --out: check.md, check.csv, and episodes.csv (one row per episode
and embodiment: ok or the exclusion reason, start-to-goal floor height difference) for make_split.py --runnable.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.benchmark import Exclusion, ViewPointCache, prepare_episode, view_point_params  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.sim.objectnav_dataset import load_objectnav_v1  # noqa: E402


def mapping(objects, goals):
    """How each dataset goal finds its semantic instance: by semantic id, by annotation id, or not at all."""
    by_sem, by_ann, missing, offsets, cats = 0, 0, 0, [], Counter()
    for g in goals:
        try:
            sid = int(g.object_id)
        except (TypeError, ValueError):
            sid = None
        o = objects.get(sid) if sid is not None else None
        how = "semantic"
        if o is None:
            o = next((x for x in objects.values() if x.object_id == str(g.object_id)
                      or x.object_id.endswith(f"_{g.object_id}")), None)
            how = "annotation"
        if o is None:
            missing += 1
            continue
        by_sem += how == "semantic"
        by_ann += how == "annotation"
        offsets.append(math.dist(o.center, g.position))
        cats[(g.category, o.category)] += 1
    return by_sem, by_ann, missing, offsets, cats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embodiments", default="habitat_agent,wheeled_vacuum")
    ap.add_argument("--per-scene", type=int, default=20, help="episodes prepared per scene and embodiment (0 = all)")
    ap.add_argument("--scenes", type=int, default=0, help="first N scenes only (0 = all)")
    ap.add_argument("--subset", default="", help="episode id file: check only these episodes (dev splits of train)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from kznav.sim.env import HabitatEnv, SceneSpec

    base = compose_config([f"dataset={args.dataset}"])
    ds = base["dataset"]
    from kznav.sim.objectnav_dataset import read_id_list, scenes_of_ids

    subset = read_id_list(Path(args.subset)) if args.subset else None
    eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"],
                            scenes=scenes_of_ids(subset) if subset else None)
    if subset:
        keep = set(subset)
        eps = [e for e in eps if e.episode_id in keep]
    scenes = list(dict.fromkeys(e.scene_path for e in eps))
    if args.scenes:
        scenes = scenes[: args.scenes]
    print(f"{len(eps)} episodes, {len(scenes)} scenes; scene dataset config {ds.get('scene_dataset_config')}")
    rows, cat_pairs, per_episode = [], Counter(), []
    for scene in scenes:
        group = [e for e in eps if e.scene_path == scene]
        goals = {(g.object_id, g.category): g for e in group for g in e.goals}.values()
        row = {"scene": group[0].scene_stem, "episodes": len(group), "exists": Path(scene).exists()}
        for emb in args.embodiments.split(","):
            cfg = compose_config([f"dataset={args.dataset}", f"embodiment={emb}"])
            t0 = time.perf_counter()
            env = HabitatEnv(cfg, SceneSpec(scene, ds.get("scene_dataset_config")), out / "work" / emb, seed=0)
            try:
                t_load = time.perf_counter() - t0
                objects = env.semantic_objects()
                if "semantic objects" not in row:
                    s, a, m, off, cats = mapping(objects, goals)
                    cat_pairs.update(cats)
                    row.update({"semantic objects": len(objects), "goal instances": len(goals), "by semantic id": s,
                                "by annotation id": a, "not found": m,
                                "offset median m": float(np.median(off)) if off else float("nan"),
                                "offset max m": float(np.max(off)) if off else float("nan"),
                                "navigable m2": round(float(env.navigable_area()), 1)})
                # the cache run_eval.py uses: the full-split check precomputes the view points of the real runs
                cache = ViewPointCache(REPO / "runs" / "cache" / "view_points" / ds["name"] / cfg["embodiment"]["name"]
                                       / f"{group[0].scene_stem}.json", view_point_params(cfg))
                reasons, n_ok, vps = Counter(), 0, []
                t1 = time.perf_counter()
                todo = group[: args.per_scene or None]
                for e in todo:
                    r = prepare_episode(env, e, cfg, objects=objects, cache=cache)
                    vps_y = [vp[1] for g in e.goals for vp in g.view_points]
                    dy = min(abs(y - e.start_position[1]) for y in vps_y) if vps_y else float("nan")
                    per_episode.append({"episode_id": e.episode_id, "scene": e.scene_stem, "category": e.category,
                                        "embodiment": emb, "result": r.reason if isinstance(r, Exclusion) else "ok",
                                        "detail": r.detail if isinstance(r, Exclusion) else "",
                                        "floor_dy_m": round(dy, 3)})
                    if isinstance(r, Exclusion):
                        reasons[r.reason] += 1
                    else:
                        n_ok += 1
                        vps.append(sum(len(g.view_points) for g in r.goals))
                cache.save()
                n = len(todo)
                row.update({f"{emb}: load s": round(t_load, 1), f"{emb}: ok": f"{n_ok}/{n}",
                            f"{emb}: excluded": ", ".join(f"{k} {v}" for k, v in reasons.items()) or "—",
                            f"{emb}: view points/ep": round(float(np.mean(vps)), 1) if vps else 0,
                            f"{emb}: prep s/ep": round((time.perf_counter() - t1) / max(1, n), 2)})
            finally:
                env.close()
        rows.append(row)
        print({k: v for k, v in row.items() if not str(k).endswith("load s")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "check.csv", index=False)
    pe = pd.DataFrame(per_episode)
    pe.to_csv(out / "episodes.csv", index=False)          # tools/make_split.py --runnable reads it
    pairs = pd.DataFrame([{"goal category": a, "annotation category": b, "instances": n}
                          for (a, b), n in sorted(cat_pairs.items(), key=lambda kv: -kv[1])])
    by_cat = pe.pivot_table(index="category", columns=["embodiment", "result"], values="episode_id", aggfunc="count",
                            fill_value=0) if len(pe) else pd.DataFrame()
    text = (f"# {ds['name']} / {ds['split']}: goal instances against the scenes' semantic annotation\n\n"
            f"{df.to_markdown(index=False)}\n\n## Goal category vs. the matched instance's category\n\n"
            f"{pairs.to_markdown(index=False) if len(pairs) else '(none matched)'}\n\n"
            f"## Episodes by category and result\n\n{by_cat.to_markdown() if len(by_cat) else ''}\n")
    (out / "check.md").write_text(text, encoding="utf-8")
    print(f"\n-> {out / 'check.md'}")


if __name__ == "__main__":
    main()
