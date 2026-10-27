"""Fixed episode subsets, written once and passed to every arm as run.episodes.subset_file.

    python tools/make_split.py dev  --dataset hm3d_objectnav_v1_train --scenes 8 --per-category 10 --out splits/hm3d_v1_dev.txt
    python tools/make_split.py test --dataset hm3d_objectnav_v1_val --per-category 80 --out splits/hm3d_v1_val_480.txt
    python tools/make_split.py test --dataset hm3d_objectnav_v1_val --per-category 100 \
        --runnable runs/hm3d_check/val/episodes.csv --embodiment wheeled_vacuum --out splits/hm3d_v1_test_primary.txt
    python tools/make_split.py test --dataset hm3d_objectnav_v1_val --random 300 --out splits/hm3d_v1_table_a.txt

dev: a few train scenes (most categories first, seeded), up to N episodes per category from them - for calibration,
thresholds, prompts, camera pitch (D-004) and the episode budget (D-005). test: up to N episodes per category from the
whole val split. The choice hashes episode ids with the seed, so it does not depend on the file order. The file
starts with comment lines (dataset, parameters, counts); tools read the ids only.
--runnable (D-052): draw only from the episodes the given embodiment can run (tools/check_hm3d.py episodes.csv,
result "ok"), so the wheeled robot's table keeps its size and both tables can use the same episodes. A smaller
--per-category with the same seed and filter gives a subset of the larger one (nested subsets).
--random N (D-064): N episodes of the whole split instead, not stratified, so the category mix is the split's own
(table (a), compared with numbers published on all of val).
--exclude-split FILE (dev, D-066): leave out the scenes of an existing split (fresh scenes for a held-out dev set);
repeat it to leave out several (D-070: dev3 excludes every scene used before).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.config import compose_config  # noqa: E402
from kznav.sim.objectnav_dataset import (balanced_subset, load_objectnav_v1, pick_scenes, random_subset,  # noqa: E402
                                         read_id_list, scene_categories)


def wilson_halfwidth(p: float, n: int) -> float:
    """Half-width of the 95% Wilson interval at proportion ``p`` and size ``n``."""
    z = 1.96
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def main() -> None:
    """Command-line entry point: write a dev or test subset of episode ids."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["dev", "test"])
    ap.add_argument("--dataset", required=True, help="dataset config name, e.g. hm3d_objectnav_v1_val")
    ap.add_argument("--per-category", type=int, default=0)
    ap.add_argument("--random", type=int, default=0, help="test: N episodes of the whole split, not stratified")
    ap.add_argument("--scenes", type=int, default=0, help="dev: number of scenes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runnable", default="", help="check_hm3d.py episodes.csv: keep episodes with result ok")
    ap.add_argument("--embodiment", default="wheeled_vacuum", help="with --runnable: whose results to use")
    ap.add_argument("--exclude-split", action="append", default=[],
                    help="dev: leave out the scenes of this split file (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.per_category > 0) == (args.random > 0) or (args.random and args.kind != "test"):
        raise SystemExit("give --per-category N, or --random N (test only)")
    ds = compose_config([f"dataset={args.dataset}"])["dataset"]
    if args.kind == "dev" and ds["role"] != "dev":
        raise SystemExit(f"a dev split comes from a dev dataset (train scenes), not {args.dataset} ({ds['role']})")
    if args.kind == "test" and ds["role"] != "benchmark":
        raise SystemExit(f"a test subset comes from a benchmark dataset, not {args.dataset} ({ds['role']})")
    scenes = None
    if args.kind == "dev":          # train is large (~4M episodes in HM3D v1): choose scenes first, load only them
        cats = scene_categories(Path(ds["episodes_path"]))
        used = {i.split(":")[0] for f in args.exclude_split for i in read_id_list(Path(f))}
        cats = {s: c for s, c in cats.items() if s not in used}
        scenes = pick_scenes(cats, args.scenes, seed=args.seed)
    eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"],
                            scenes=scenes)
    n_all = len(eps)
    filt = ""
    if args.runnable:
        import pandas as pd

        chk = pd.read_csv(args.runnable)
        chk = chk[chk.embodiment == args.embodiment]
        checked = set(chk.episode_id)
        if args.kind == "test" and {e.episode_id for e in eps} - checked:
            raise SystemExit(f"{len({e.episode_id for e in eps} - checked)} episodes of {args.dataset} were not checked "
                             f"for {args.embodiment}; run tools/check_hm3d.py --per-scene 0 first")
        ok = set(chk.loc[chk.result == "ok", "episode_id"])
        n_checked = sum(e.episode_id in checked for e in eps)
        eps = [e for e in eps if e.episode_id in ok]          # dev: a checked pool of the (huge) train split
        filt = f"runnable for {args.embodiment}: {len(eps)} of {n_checked} checked ({args.runnable})"
    if args.random:
        ids = random_subset(eps, args.random, seed=args.seed)
    else:
        ids = balanced_subset(eps, args.per_category, seed=args.seed, scenes=scenes)
    by_id = {e.episode_id: e for e in eps}
    cats = Counter(by_id[i].category for i in ids)
    n = len(ids)
    header = [f"# {args.kind} subset of {ds['name']}/{ds['split']} ({args.dataset}): {n} of {n_all} episodes"
              + (f"; {filt}" if filt else ""),
              (f"# simple random sample of {args.random}, not stratified, seed {args.seed}" if args.random else
               f"# per category <= {args.per_category}, seed {args.seed}" + (f", scenes {scenes}" if scenes else "")),
              f"# by category: {dict(sorted(cats.items()))}",
              f"# SR 95% CI half-width at n={n}: +-{100 * wilson_halfwidth(0.3, n):.1f} pp (SR 0.3), "
              f"+-{100 * wilson_halfwidth(0.5, n):.1f} pp (SR 0.5)"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + ids) + "\n", encoding="utf-8")
    print("\n".join(header))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
