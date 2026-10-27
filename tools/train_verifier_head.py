"""Train the light verifier (D-075 step 2): a small MLP on SigLIP 2 features of the crop and of "a photo of a <name>.",
trained on the robot-camera crops of tools/make_verifier_data.py, judged on the verifier crops of the finished runs.

    bash tools/wsl.sh python tools/train_verifier_head.py --data runs/verifier_data/v1 --out models/verifier/siglip_head_v1

20% of the data scenes are held out (stable hash): early stopping, Platt calibration and the max-F1 stop threshold
come from them. The evaluation crops (dev scenes, never in the data) only judge. Writes ``<out>/head.pt``, ``<out>/
summary.md`` and two calibration artifacts the robot loads: ``calib/siglip_head_v1.json`` (the head) and
``calib/siglip_zero_v1.json`` (SigLIP without training, the D-075 fallback). Prints the D-075 rule's verdict.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from calib_transfer import auroc  # noqa: E402
from eval_crop_verifiers import DISTRACTORS, NAMES, collect, rates  # noqa: E402
from kznav.policy.calibration import CalibrationArtifact, PlattCalibrator, choose_threshold  # noqa: E402

TEMPLATE = "a photo of a {}."
# names a parse may give the target (target_en): training draws one per crop, so the head does not hang on one word
SYNONYMS = {"bed": ["bed"], "chair": ["chair", "armchair"], "plant": ["plant", "potted plant", "houseplant", "flower"],
            "sofa": ["sofa", "couch"], "toilet": ["toilet"], "tv_monitor": ["television", "tv", "tv monitor"]}


def embed_images(model, proc, files, batch=64):
    """Normalised SigLIP 2 image embeddings of files, in batches."""
    import torch
    from PIL import Image

    out = []
    with torch.inference_mode():
        for i in range(0, len(files), batch):
            ims = [Image.open(f).convert("RGB") for f in files[i : i + batch]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to("cuda").half()
            e = model.get_image_features(pixel_values=px)
            out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out)


def embed_texts(model, proc, texts):
    """Normalised SigLIP 2 embeddings of names in the prompt template."""
    import torch

    with torch.inference_mode():
        t = proc(text=[TEMPLATE.format(x) for x in texts], padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        e = model.get_text_features(**t)
        return (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()


def features(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """Head input: image embedding, text embedding and their element-wise product."""
    return np.concatenate([img, txt, img * txt], axis=1).astype(np.float32)


def make_head(dim: int, hidden: int = 512):
    """The MLP head: linear, GELU, dropout, linear to one logit."""
    import torch.nn as nn

    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))


def predict(head, X: np.ndarray) -> np.ndarray:
    """Sigmoid outputs of the head."""
    import torch

    head.eval()
    with torch.inference_mode():
        z = head(torch.from_numpy(X).to("cuda")).squeeze(1)
    return torch.sigmoid(z).cpu().numpy()


def artifact(p, y, name: str, model_id: str, sha: str, note: str) -> CalibrationArtifact:
    """Calibration artifact of the head's scores (Platt fit, max-F1 threshold)."""
    cal = PlattCalibrator.fit(p, y)
    thr = choose_threshold(cal(p), y)
    return CalibrationArtifact(calibrator=cal.to_dict(), threshold=asdict(thr), backend=name, model_id=model_id,
                               prompt_sha256=sha, dev_split="verifier_data held-out scenes", dev_episode_ids=[],
                               n_samples=int(len(p)), base_rate=float(np.mean(y)),
                               created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), notes=note)


def main() -> None:
    """Command-line entry point: train the head, evaluate it on the verifier crops and save it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "runs/verifier_data/v1"))
    ap.add_argument("--out", default=str(REPO / "models/verifier/siglip_head_v1"))
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.manual_seed(args.seed)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(data / "crops.csv")
    tr["path"] = [str(data / f) for f in tr.file]
    held = {s for s in tr.scene.unique() if int(hashlib.sha256(f"{args.seed}:{s}".encode()).hexdigest(), 16) % 5 == 0}
    tr["val"] = tr.scene.isin(held)
    runs = [Path(r) for r in sorted(glob.glob(str(REPO / "runs/matrix/*/*/seed0")))
            if (Path(r) / "images").is_dir() and (Path(r) / "vlm_calls.jsonl").exists()]
    dv = pd.concat([collect(r) for r in runs], ignore_index=True).drop_duplicates(subset=["file", "category"])
    dv = dv.reset_index(drop=True)
    print(f"data: {len(tr)} crops ({int(tr.label.sum())} goal) from {tr.scene.nunique()} scenes, {len(held)} held out; "
          f"evaluation: {len(dv)} verifier crops", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    cats = list(NAMES)
    T = embed_texts(model, proc, [NAMES[c] for c in cats])
    Tall = embed_texts(model, proc, [NAMES[c] for c in cats] + DISTRACTORS)
    scale, bias = float(model.logit_scale.exp()), float(model.logit_bias)
    Itr = embed_images(model, proc, list(tr.path))
    Idv = embed_images(model, proc, list(dv.file))
    ci_tr = tr.target.map({c: i for i, c in enumerate(cats)}).to_numpy()
    ci_dv = dv.category.map({c: i for i, c in enumerate(cats)}).to_numpy()

    def zero_shot(I, ci):
        L = scale * (I @ Tall.T) + bias
        sig = 1 / (1 + np.exp(-L[np.arange(len(I)), ci]))
        sm = np.exp(L - L.max(axis=1, keepdims=True))
        sm /= sm.sum(axis=1, keepdims=True)
        return sig, sm[np.arange(len(I)), ci]

    syn = [(c, n) for c in cats for n in SYNONYMS[c]]
    Tsyn = embed_texts(model, proc, [n for _, n in syn])
    pick = np.array([[j for j, (c, _) in enumerate(syn) if c == cat][int(hashlib.sha256(f.encode()).hexdigest(), 16)
                     % len(SYNONYMS[cat])] for cat, f in zip(tr.target, tr.file)])
    Xtr, ytr = features(Itr, Tsyn[pick]), tr.label.to_numpy().astype(np.float32)
    Xdv, ydv = features(Idv, T[ci_dv]), dv.goal.to_numpy()
    fit, val = ~tr.val.to_numpy(), tr.val.to_numpy()
    head = make_head(Xtr.shape[1]).to("cuda")
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor((1 - ytr[fit].mean()) / max(ytr[fit].mean(), 1e-6), device="cuda")
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xf, yf = torch.from_numpy(Xtr[fit]).to("cuda"), torch.from_numpy(ytr[fit]).to("cuda")
    best, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Xf), device="cuda")
        for i in range(0, len(Xf), 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            lossf(head(Xf[b]).squeeze(1), yf[b]).backward()
            opt.step()
        a = auroc(predict(head, Xtr[val]), ytr[val].astype(bool))
        print(f"epoch {ep + 1}: held-out scenes AUROC {a:.3f}", flush=True)
        if a > best + 1e-4:
            best, best_state, patience = a, {k: v.detach().clone() for k, v in head.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    head.load_state_dict(best_state)
    p_val, p_dv = predict(head, Xtr[val]), predict(head, Xdv)
    sig_val, soft_val = zero_shot(Itr[val], ci_tr[val])
    sig_dv, soft_dv = zero_shot(Idv, ci_dv)

    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "in_dim": Xtr.shape[1],
                "hidden": 512, "model_id": args.model, "template": TEMPLATE}, out / "head.pt")
    sha = hashlib.sha256((out / "head.pt").read_bytes()).hexdigest()
    yv = ytr[val].astype(bool)
    # calibration and stop threshold on the robot's own verifier crops (realistic base rate ~0.2; the data crops are
    # ~0.4 goal), as every calibration since D-067 - without the dev3 scenes, where step 3a runs (D-075 addendum)
    cal = (dv.scenes != "dev3").to_numpy()
    artifact(p_dv[cal], ydv[cal], "siglip_head", f"{out.name}/head.pt", sha,
             f"MLP on SigLIP features; data {data}; calibrated on {int(cal.sum())} verifier crops of dev scenes "
             f"except dev3").save(REPO / "calib/siglip_head_v1.json")
    # only "sig" can be used on the robot: "soft" needs the closed list of categories, and a free-form name
    # (target_en "couch") would compete with its own synonym in it (D-075 addendum)
    zero_variant = "sig"
    artifact(sig_dv[cal], ydv[cal], "siglip_zero", f"{args.model}#{zero_variant}",
             hashlib.sha256(zero_variant.encode()).hexdigest(),
             f"SigLIP zero-shot ({zero_variant}); calibrated on {int(cal.sum())} verifier crops of dev scenes except dev3"
             ).save(REPO / "calib/siglip_zero_v1.json")

    rows = []
    groups = [("held-out data scenes", None)] + [("all verifier crops", np.ones(len(dv), bool))] + [
        (f"{b} {q} {e} {s}", (dv.backend == b) & (dv.question_lang == q) & (dv.embodiment == e) & (dv.scenes == s))
        for (b, q, e, s), g in dv.groupby(["backend", "question_lang", "embodiment", "scenes"]) if len(g) >= 40]
    for name, m in groups:
        if m is None:
            scores, y = {"head": p_val, "sig": sig_val, "soft": soft_val}, yv
        else:
            m = np.asarray(m)
            scores, y = {"VLM": dv.p_vlm.to_numpy()[m], "head": p_dv[m], "sig": sig_dv[m], "soft": soft_dv[m]}, ydv[m]
        for k, s in scores.items():
            rows.append({"crops": name, "n": int(len(y)), "goal": int(y.sum()), "score": k,
                         **{a: round(v, 3) for a, v in rates(s, y).items()}})
    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    a_head, a_soft = auroc(p_dv, ydv), auroc(soft_dv, ydv)
    choice = "head" if a_head >= 0.90 and a_head >= a_soft + 0.01 else f"zero-shot ({zero_variant})"
    lines = [f"# Light verifier: {out.name}", "", f"data {data}: {len(tr)} crops, {tr.scene.nunique()} scenes "
             f"({len(held)} held out); evaluation: {len(dv)} verifier crops of the finished runs", "",
             res.to_markdown(index=False) if hasattr(res, "to_markdown") else res.to_string(index=False), "",
             f"D-075 rule: head AUROC on the verifier crops {a_head:.3f}, zero-shot soft {a_soft:.3f} -> **{choice}**"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    (out / "choice.json").write_text(json.dumps({"choice": choice, "auroc_head": a_head, "auroc_soft": a_soft,
                                                  "zero_variant": zero_variant}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
