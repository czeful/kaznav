"""Dev-split studies that fix open parameters before any reported run: camera pitch (D-004), episode budget (D-005).

    bash tools/wsl.sh python tools/dev_studies.py pitch --pitches -15,-5,0,10,20 --out runs/dev/pitch \\
        dataset=hm3d_objectnav_v1_train run.episodes.subset_file=splits/hm3d_v1_dev.txt
    python tools/dev_studies.py budget runs/eval/<dev run with a large budget> [...] --out runs/dev/budget

pitch: for each pitch, the wheeled robot's view points are recomputed for every goal instance of the dev episodes;
reported: instances visible from at least one view point, view points per instance, episodes excluded. A pitch that
keeps more goals visible from the robot's height is the candidate; the choice is confirmed with oracle-detector dev
runs (tools/run_eval.py ... detector=oracle_semantic embodiment.sensors.rgbd.mount.pitch_deg=<p>).
budget: SR as a function of the decision-step limit, from dev runs made with a generous limit
(eval.episode_budget.max_decision_steps); the recommended limit is the smallest whose SR is >= 99% of the SR at the
largest limit. Both only on dev data (role: dev); the val split is never used to choose them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.config import compose_config  # noqa: E402


def pitch_study(pitches, overrides, out: Path) -> pd.DataFrame:
    """Goal visibility from the wheeled robot's camera for each pitch: view points per goal instance,
    exclusions and runnable episodes per category.

    Args:
        pitches: Camera pitches in degrees (positive = up).
        overrides: Hydra overrides of the dev configuration.
        out: Output directory.

    Returns:
        One row per pitch.
    """
    from kznav.run.benchmark import Exclusion, ViewPointCache, prepare_episode, view_point_params
    from kznav.sim.env import HabitatEnv, SceneSpec
    from kznav.sim.objectnav_dataset import load_objectnav_v1, read_id_list, scenes_of_ids, select_episodes

    rows = []
    for p in pitches:
        cfg = compose_config([*overrides, f"embodiment.sensors.rgbd.mount.pitch_deg={p}"])
        ds = cfg["dataset"]
        if ds["role"] != "dev":
            raise SystemExit(f"the pitch is chosen on dev data only; {ds['name']}/{ds['split']} is {ds['role']}")
        if cfg["embodiment"]["navmesh"]["source"] == "dataset":
            raise SystemExit("the standard agent keeps the dataset's camera; the study is for the wheeled robot")
        sel = cfg["run"]["episodes"]
        subset = read_id_list(Path(sel["subset_file"])) if sel.get("subset_file") else None
        eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"],
                                scenes=scenes_of_ids(subset) if subset else None)
        eps = select_episodes(eps, subset=subset, limit=sel.get("limit"))
        inst_vps, excl, ok_by_cat = {}, {}, {}
        for scene in dict.fromkeys(e.scene_path for e in eps):
            env = HabitatEnv(cfg, SceneSpec(scene, ds.get("scene_dataset_config")), out / "work", seed=0)
            try:
                objects = env.semantic_objects()
                group = [e for e in eps if e.scene_path == scene]
                cache = ViewPointCache(REPO / "runs" / "cache" / "view_points" / ds["name"] / cfg["embodiment"]["name"]
                                       / f"{group[0].scene_stem}.json", view_point_params(cfg))
                for e in group:
                    r = prepare_episode(env, e, cfg, objects=objects, cache=cache)
                    if isinstance(r, Exclusion):
                        excl[r.reason] = excl.get(r.reason, 0) + 1
                    else:
                        ok_by_cat[e.category] = ok_by_cat.get(e.category, 0) + 1
                        for g in r.goals:
                            inst_vps[(e.scene_stem, g.object_id)] = len(g.view_points)
                cache.save()
            finally:
                env.close()
        n = np.array(list(inst_vps.values()) or [0])
        rows.append({"pitch_deg": p, "episodes": len(eps), "goal instances": len(inst_vps),
                     "visible from >= 1 view point": f"{(n > 0).mean():.3f}", "view points / instance": f"{n.mean():.1f}",
                     "excluded": sum(excl.values()), **{f"excl {k}": v for k, v in excl.items()},
                     **{f"runnable {k}": v for k, v in sorted(ok_by_cat.items())}})
    return pd.DataFrame(rows).fillna(0)


def budget_curve(runs, out: Path) -> pd.DataFrame:
    """SR as a function of the decision-step limit, with the smallest limit reaching 99% of the full SR.

    Args:
        runs: Run directories (their ``episodes.csv`` is read).
        out: Output directory of the figure.

    Returns:
        The curve, with the recommended limit marked.
    """
    df = pd.concat([pd.read_csv(Path(r) / "episodes.csv") for r in runs], ignore_index=True)
    steps = df.num_decision_steps.astype(int)
    succ = df.success.astype(bool)
    limit = int(steps.max())
    ks = sorted(set(np.linspace(50, limit, 20).astype(int)) | {limit})
    rows = [{"max_decision_steps": k, "SR": float((succ & (steps <= k)).mean())} for k in ks]
    curve = pd.DataFrame(rows)
    full = curve.SR.iloc[-1]
    rec = int(curve[curve.SR >= 0.99 * full].max_decision_steps.min()) if full > 0 else limit
    curve["recommended"] = curve.max_decision_steps == rec
    from kznav.report.figures import ACCENT, _despine, _plt, save

    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(curve.max_decision_steps.to_numpy(), 100 * curve.SR.to_numpy(), "-o", color=ACCENT, ms=3)   # mpl 3.3 + pandas 2
    ax.axvline(rec, color="#8c8c8c", lw=1, ls="--")
    ax.set_xlabel("decision-step limit")
    ax.set_ylabel("SR, %")
    ax.set_title(f"SR saturates by {rec} steps ({len(df)} dev episodes)", fontsize=10, loc="left")
    _despine(ax)
    save(fig, out / "budget_curve")
    return curve


def pitch_select(runs: dict) -> pd.DataFrame:
    """D-004 step 2: oracle-detector runs at several pitches (label -> run dir). SR over the UNION of the episodes run
    at any pitch; an episode not run at a pitch (no view point there) is a failure at that pitch. Highest wins, ties
    go to the smaller pitch (the order given)."""
    eps = {k: pd.read_csv(Path(v) / "episodes.csv") for k, v in runs.items()}
    union = sorted(set().union(*(set(d.episode_id) for d in eps.values())))
    cats = pd.concat(eps.values()).drop_duplicates("episode_id").set_index("episode_id").goal_category
    rows = []
    for k, d in eps.items():
        ok = set(d.loc[d.success.astype(bool), "episode_id"])
        row = {"pitch": k, "episodes run": len(d), "union": len(union),
               "SR over union": sum(e in ok for e in union) / len(union),
               "SR over run": d.success.astype(bool).mean()}
        for c in sorted(cats.unique()):
            members = [e for e in union if cats[e] == c]
            row[f"SR {c}"] = f"{sum(e in ok for e in members)}/{len(members)}"
        rows.append(row)
    out = pd.DataFrame(rows)
    best = out["SR over union"].max()
    out["chosen"] = False
    out.loc[out.index[out["SR over union"] == best][0], "chosen"] = True
    return out


def main() -> None:
    """Command-line entry point: ``pitch``, ``pitch-select`` or ``budget``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pitch")
    p.add_argument("--pitches", required=True, help="comma-separated degrees, positive = up")
    p.add_argument("--out", required=True)
    p.add_argument("overrides", nargs="*")
    ps = sub.add_parser("pitch-select")
    ps.add_argument("runs", nargs="+", help="label=run_dir, in increasing pitch")
    ps.add_argument("--out", required=True)
    b = sub.add_parser("budget")
    b.add_argument("runs", nargs="+")
    b.add_argument("--out", required=True)
    args = ap.parse_args()            # options first, then overrides (intermixed parsing does not support subcommands)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "pitch-select":
        table = pitch_select(dict(r.split("=", 1) for r in args.runs))
        title = "Camera pitch on dev (D-004, step 2): oracle SR over the union of runnable episodes"
    elif args.cmd == "pitch":
        table = pitch_study([float(x) for x in args.pitches.split(",")], args.overrides, out)
        title = "Camera pitch on dev (D-004): goal visibility from the wheeled robot"
    else:
        table = budget_curve(args.runs, out)
        title = "Episode budget on dev (D-005): SR by decision-step limit"
    table.to_csv(out / f"{args.cmd}.csv", index=False)
    text = f"# {title}\n\n{table.to_markdown(index=False)}\n"
    (out / f"{args.cmd}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
