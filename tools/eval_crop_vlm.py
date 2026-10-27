"""The same verifier crops, the same English question, asked of each VLM live (D-083 step 1b).

    bash tools/start_gemma.sh
    bash tools/wsl.sh python tools/eval_crop_vlm.py --out runs/verifier_bench/vlm_same_crops -- backend=gemma3_4b

Crops: the labelled verifier crops of the finished runs (``eval_crop_verifiers.collect``: label by crop content,
D-042). A fixed sample is drawn once (seed 0, half goal crops and half others) and saved as ``<out>/sample.csv``, so
every model is asked about exactly the same crops. Question: the English block of the verification prompt with the
English name of the episode's goal category (as the English short-name question asks, D-071). Score: the model's
P(yes) from first-token logprobs.

Writes ``<out>/<backend>.csv`` (resumable) and rewrites ``<out>/summary.md`` with AUROC, TPR at FPR 0.1 and FPR at
TPR 0.7 for every model asked so far, overall and per robot, plus the server's peak memory.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from eval_crop_verifiers import NAMES, _table, collect, rates  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402
from kznav.language.schemas import VerifyQuery  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402


def sample_crops(out: Path, n: int, seed: int) -> pd.DataFrame:
    path = out / "sample.csv"
    if path.exists():
        return pd.read_csv(path)
    runs = [Path(r) for r in sorted(glob.glob(str(REPO / "runs/matrix/*/*/seed0")))
            if (Path(r) / "images").is_dir() and (Path(r) / "vlm_calls.jsonl").exists()]
    df = pd.concat([collect(r) for r in runs], ignore_index=True).drop_duplicates(subset=["file", "category"])
    df = df.groupby("goal", group_keys=False).apply(lambda g: g.sample(n=min(n // 2, len(g)), random_state=seed))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["file"] = [str(Path(f).relative_to(REPO)) for f in df.file]
    out.mkdir(parents=True, exist_ok=True)
    df[["file", "category", "goal", "embodiment", "scenes", "run"]].to_csv(path, index=False)
    return pd.read_csv(path)


def server_stats(base_url: str) -> dict:
    import httpx

    try:
        return httpx.get(base_url.rsplit("/v1", 1)[0] + "/kznav/stats", timeout=10).json()
    except Exception as e:  # noqa: BLE001 - the stats are a report, not a requirement
        return {"error": str(e)}


def summarize(out: Path) -> str:
    rows, mem = [], {}
    for f in sorted(out.glob("*.csv")):
        if f.name == "sample.csv":
            continue
        d = pd.read_csv(f).dropna(subset=["p"])
        name = f.stem
        st = out / f"{name}.stats.json"
        if st.exists():
            mem[name] = json.loads(st.read_text(encoding="utf-8")).get("max_allocated_gb")
        for group, g in [("all", d)] + [(e, g) for e, g in d.groupby("embodiment")]:
            y = g.goal.astype(bool).to_numpy()
            rows.append({"model": name, "crops": group, "n": len(g), "goal": int(y.sum()),
                         **{k: round(v, 3) for k, v in rates(g.p.to_numpy(), y).items()}})
    res = pd.DataFrame(rows)
    lines = ["# The same crops, the same English question, each VLM live (D-083 step 1b)", "",
             _table(res) if len(res) else "(no model asked yet)", "",
             "Server peak memory (GB): " + ", ".join(f"{k} {v:.2f}" for k, v in mem.items() if v is not None)]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "runs/verifier_bench/vlm_same_crops"))
    ap.add_argument("--n", type=int, default=2000, help="crops in the shared sample (half goal)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    crops = sample_crops(out, args.n, args.seed)
    cfg = compose_config(args.overrides)
    backend = create_backend(cfg["backend"], sink=lambda rec: None)
    name = cfg["backend"]["name"]
    res_path = out / f"{name}.csv"
    done = set(pd.read_csv(res_path).file) if res_path.exists() else set()
    from PIL import Image

    todo = crops[~crops.file.isin(done)]
    print(f"[crops] {name}: {len(todo)} to ask, {len(done)} done", flush=True)
    new = not res_path.exists()
    with open(res_path, "a", encoding="utf-8") as f:
        if new:
            f.write("file,category,goal,embodiment,p\n")
        for i, r in enumerate(todo.itertuples(), 1):
            img = np.asarray(Image.open(REPO / r.file).convert("RGB"))
            res = backend.verify_candidate(img, VerifyQuery(NAMES[r.category], "en"))
            p = "" if res.p_target is None else f"{res.p_target:.6f}"
            f.write(f"{r.file},{r.category},{bool(r.goal)},{r.embodiment},{p}\n")
            if i % 200 == 0:
                f.flush()
                print(f"[crops] {name}: {i}/{len(todo)}", flush=True)
    (out / f"{name}.stats.json").write_text(json.dumps(server_stats(cfg["backend"]["base_url"])), encoding="utf-8")
    print(summarize(out))
    print("CROPS_DONE", flush=True)


if __name__ == "__main__":
    main()
