"""How well does a light image-text model tell the goal from other objects on the verifier's own crops? (step 0 of a
custom verifier, after D-073). Every verifier call of the finished matrix runs whose crop was saved is scored again by
SigLIP 2 (no robot, no VLM) and compared, crop by crop, with the VLM's own P(yes) logged for it.

    bash tools/wsl.sh python tools/eval_crop_verifiers.py [--runs runs/matrix/*/*/seed0] [--out runs/verifier_bench/siglip0]

Labels: crop content (goal-instance pixels in the detection box, D-042), as the calibration uses. SigLIP is given the
English name of the episode's goal category (what the English short-name question asks about, D-071). Two SigLIP
scores: ``sig`` = SigLIP's own sigmoid for "a photo of a <name>."; ``soft`` = softmax over the six HM3D categories and
a few common distractors (wall, floor, door, table, cabinet, window, shelf, box), probability of the goal's name.
Per group (model x question language x robot x scene set): AUROC, TPR at FPR 0.1, FPR at TPR 0.7.
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

from calib_transfer import auroc  # noqa: E402
from kznav.run.logging import completed_events  # noqa: E402

NAMES = {"bed": "bed", "chair": "chair", "plant": "plant", "sofa": "sofa", "toilet": "toilet", "tv_monitor": "television"}
DISTRACTORS = ["wall", "floor", "door", "table", "cabinet", "window", "shelf", "box"]
SCENE_SET = {"hm3d_v1_dev.txt": "dev10", "hm3d_v1_dev_agent.txt": "dev10", "hm3d_v1_dev_tune85.txt": "dev10",
             "hm3d_v1_dev2_demo.txt": "dev2", "hm3d_v1_dev3.txt": "dev3"}


def collect(run: Path) -> pd.DataFrame:
    """One row per labelled verifier call whose crop image was saved."""
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    cfg = man.get("config") or {}
    subset = Path(str(((cfg.get("run") or {}).get("episodes") or {}).get("subset_file") or "")).name
    emb = (cfg.get("embodiment") or {}).get("name", "?")
    eps = pd.read_csv(run / "episodes.csv").set_index("episode_id")
    files = {}                                         # (episode, p_raw) -> image file
    for line in open(run / "vlm_calls.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("method") != "verify_candidate" or not r.get("image_files") or not r.get("output"):
            continue
        p = r["output"].get("p_target")
        if p is not None:
            files[(r["context"]["episode_id"], round(float(p), 12))] = (run / r["image_files"][0], r["inputs"])
    rows = []
    for e in completed_events(run / "events.jsonl", "verification"):
        if not e.get("called") or e.get("p_raw") is None or e.get("is_goal_by_content") is None:
            continue
        key = (e["episode_id"], round(float(e["p_raw"]), 12))
        if key not in files or e["episode_id"] not in eps.index:
            continue
        f, inp = files[key]
        if not f.exists():
            continue
        rows.append({"run": str(run.relative_to(REPO)), "backend": man.get("backend", {}).get("name", "?"),
                     "question_lang": inp.get("verify_lang", "?"), "embodiment": emb,
                     "scenes": SCENE_SET.get(subset, subset or "?"), "episode_id": e["episode_id"],
                     "category": eps.loc[e["episode_id"], "goal_category"], "asked": inp.get("target"),
                     "p_vlm": float(e["p_raw"]), "goal": bool(e["is_goal_by_content"]), "file": str(f)})
    return pd.DataFrame(rows)


def rates(score: np.ndarray, y: np.ndarray) -> dict:
    pos, neg = np.sort(score[y]), np.sort(score[~y])
    if not len(pos) or not len(neg):
        return {"auroc": np.nan, "tpr@fpr0.1": np.nan, "fpr@tpr0.7": np.nan}
    thr10 = np.quantile(neg, 0.9)                              # 10% of the negatives above it
    thr70 = np.quantile(pos, 0.3)                              # 70% of the positives above it
    return {"auroc": auroc(score, y), "tpr@fpr0.1": float((pos > thr10).mean()), "fpr@tpr0.7": float((neg >= thr70).mean())}


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:                                # tabulate missing in the environment
        return df.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=sorted(glob.glob(str(REPO / "runs/matrix/*/*/seed0"))))
    ap.add_argument("--out", default=str(REPO / "runs/verifier_bench/siglip0"))
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs if (Path(r) / "images").is_dir() and (Path(r) / "vlm_calls.jsonl").exists()]
    df = pd.concat([collect(r) for r in runs], ignore_index=True)
    df = df.drop_duplicates(subset=["file", "category"]).reset_index(drop=True)
    print(f"{len(df)} labelled crops from {df.run.nunique()} runs; goal crops {int(df.goal.sum())}")

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    texts = [f"a photo of a {t}." for t in list(NAMES.values()) + DISTRACTORS]
    with torch.inference_mode():
        ti = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        temb = model.get_text_features(**ti)
        temb = temb / temb.norm(dim=-1, keepdim=True)
        scale, bias = model.logit_scale.exp().float(), model.logit_bias.float()
        logits = []
        for i in range(0, len(df), args.batch):
            imgs = [Image.open(f).convert("RGB") for f in df.file[i : i + args.batch]]
            ii = proc(images=imgs, return_tensors="pt").to("cuda")
            emb = model.get_image_features(pixel_values=ii["pixel_values"].half())
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits.append((scale * (emb @ temb.T).float() + bias).cpu().numpy())
    L = np.concatenate(logits)
    col = {name: j for j, name in enumerate(NAMES)}
    idx = df.category.map(col).to_numpy()
    df["sig"] = 1 / (1 + np.exp(-L[np.arange(len(df)), idx]))
    soft = np.exp(L - L.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    df["soft"] = soft[np.arange(len(df)), idx]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["file"]).to_csv(out / "crops.csv", index=False)
    groups = [("all", df)] + [(f"{b} {q} {e} {s}", g) for (b, q, e, s), g in
                               df.groupby(["backend", "question_lang", "embodiment", "scenes"]) if len(g) >= 40]
    rows = []
    for name, g in groups:
        y = g.goal.to_numpy()
        for score in ("p_vlm", "sig", "soft"):
            rows.append({"group": name, "crops": len(g), "goal": int(y.sum()), "score": score,
                         **{k: round(v, 3) for k, v in rates(g[score].to_numpy(), y).items()}})
    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    md = ["# Verifier crops: VLM P(yes) vs SigLIP 2 zero-shot", "", f"{len(df)} labelled crops; model {args.model}", "",
          _table(res)]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
