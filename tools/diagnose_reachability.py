"""Why the wheeled robot cannot run a dataset episode: another floor (stairs - a legitimate exclusion for a robot
vacuum) or a navmesh that falls apart on the scan's thresholds and floor noise (an artefact of the climb setting).

    bash tools/wsl.sh python tools/diagnose_reachability.py --dataset hm3d_objectnav_v1_valmini --climbs 0.01,0.02,0.03 \\
        --out runs/hm3d_check/reach

Per episode: the height difference between the start and the goal's floor (from the dataset's view points, which
stand on the floor next to the goal), and for each navmesh climb height the wheeled robot's preparation result
(kznav.run.benchmark.prepare_episode). An episode whose goal is on the start's floor but that is unreachable at the
robot's measured climb height and reachable at a slightly larger one is a candidate artefact.
Output: reach.csv and reach.md in --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.benchmark import Exclusion, ViewPointCache, prepare_episode, view_point_params  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.sim.objectnav_dataset import load_objectnav_v1  # noqa: E402

SAME_FLOOR_M = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--climbs", default="0.01,0.02,0.03")
    ap.add_argument("--embodiment", default="wheeled_vacuum")
    ap.add_argument("--scenes", type=int, default=0)
    ap.add_argument("--per-scene", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from kznav.sim.env import HabitatEnv, SceneSpec

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = compose_config([f"dataset={args.dataset}"])["dataset"]
    eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"])
    scenes = list(dict.fromkeys(e.scene_path for e in eps))[: args.scenes or None]
    rows = []
    for scene in scenes:
        group = [e for e in eps if e.scene_path == scene][: args.per_scene or None]
        for e in group:
            vps_y = [vp[1] for g in e.goals for vp in g.view_points]
            dy = min(abs(y - e.start_position[1]) for y in vps_y) if vps_y else float("nan")
            rows.append({"scene": e.scene_stem, "episode_id": e.episode_id, "category": e.category,
                         "floor dy m": round(dy, 2), "same floor": bool(dy <= SAME_FLOOR_M),
                         "dataset geodesic m": e.info.get("geodesic_distance")})
        for climb in [float(c) for c in args.climbs.split(",")]:
            cfg = compose_config([f"dataset={args.dataset}", f"embodiment={args.embodiment}",
                                  f"embodiment.navmesh.agent_max_climb_m={climb}"])
            env = HabitatEnv(cfg, SceneSpec(scene, ds.get("scene_dataset_config")), out / "work", seed=0)
            try:
                objects = env.semantic_objects()
                cache = ViewPointCache(out / "cache" / f"{group[0].scene_stem}_{climb}.json", view_point_params(cfg))
                area = env.navigable_area()
                for e in group:
                    r = prepare_episode(env, e, cfg, objects=objects, cache=cache)
                    row = next(x for x in rows if x["episode_id"] == e.episode_id)
                    row[f"climb {climb}"] = r.reason if isinstance(r, Exclusion) else "ok"
                    row[f"area {climb} m2"] = round(area, 1)
                cache.save()
            finally:
                env.close()
        print(pd.DataFrame([r for r in rows if r["scene"] == group[0].scene_stem]).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "reach.csv", index=False)
    climbs = [c for c in df.columns if c.startswith("climb ")]
    summary = []
    for c in climbs:
        for floor in (True, False):
            sub = df[df["same floor"] == floor]
            summary.append({"climb": c, "goal floor": "same as start" if floor else "other floor", "episodes": len(sub),
                            "ok": int((sub[c] == "ok").sum()),
                            **{k: int((sub[c] == k).sum()) for k in sorted(set(df[c])) if k != "ok"}})
    sm = pd.DataFrame(summary).fillna(0)
    text = (f"# Reachability of {ds['name']}/{ds['split']} for {args.embodiment}\n\n{sm.to_markdown(index=False)}\n\n"
            f"## Episodes\n\n{df.to_markdown(index=False)}\n")
    (out / "reach.md").write_text(text, encoding="utf-8")
    print(sm.to_string(index=False))
    print(f"-> {out / 'reach.md'}")


if __name__ == "__main__":
    main()
