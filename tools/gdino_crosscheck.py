"""D-009: the bench's GroundingDINO (HF transformers) against the original implementation on the same images.

    bash tools/wsl.sh python tools/gdino_crosscheck.py frames --out runs/gdino_check        # habitat_env: render
    bash tools/wsl.sh python tools/gdino_crosscheck.py detect --impl hf --dir runs/gdino_check
    KZNAV_CONDA_ENV=nav_env bash tools/wsl.sh python tools/gdino_crosscheck.py detect --impl original --dir runs/gdino_check
    python tools/gdino_crosscheck.py compare --dir runs/gdino_check

Images: the robot's camera (0.15 m) at view points of the debug scene's objects and at random poses, plus the
pilot's 8 photos. Both implementations get the same caption ("q1 . q2 . q3 .") and thresholds (box 0.35, text 0.25,
configs/detector/grounding_dino_tiny.yaml) and the same SwinT-OGC weights. The original runs in the pilot's nav_env
(groundingdino 0.1.0, transformers 4.35.2), so the 'original' mode imports nothing from kznav.

Agreement criterion, fixed before looking at the numbers (D-009): boxes matched one-to-one at IoU >= 0.5 cover at
least 90% of each implementation's boxes, the median IoU of matched boxes is >= 0.9, and the median score
difference is <= 0.05. Output: <dir>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "qolda check"
PHOTO_QUERIES = {"divan": "sofa", "kazan": "cauldron", "samovar": "samovar", "sarmak": "rug", "sebet": "basket",
                 "shapan": "robe", "shkav": "cabinet", "vaza": "vase"}     # the objects in the pilot's photos
BOX_THRESHOLD, TEXT_THRESHOLD = 0.35, 0.25


# ---------------------------------------------------------------------------- frames (habitat_env)
def make_frames(out: Path, per_object: int, n_random: int, seed: int) -> None:
    sys.path.insert(0, str(REPO))
    from PIL import Image

    from kznav.language.lexicon import Lexicon
    from kznav.run.config import compose_config
    from kznav.sim.env import HabitatEnv, SceneSpec
    from kznav.sim.episodes import ObjectSpec, compute_view_points, placement_populate
    from kznav.sim.frames import habitat_to_map

    cfg = compose_config(["dataset=debug_ycb"])
    ds, vp = cfg["dataset"], cfg["eval"]["view_points"]
    queries = [Lexicon.load(REPO / ds["lexicon"]).categories[o["category"]].en()[0] for o in ds["objects"]]
    rng = np.random.default_rng(seed)
    placed = []
    specs = [ObjectSpec(o["category"], str(o["config_path"]), int(o["count"])) for o in ds["objects"]]
    scene = str(Path(cfg["paths"]["data_root"]) / ds["scenes"][0])
    env = HabitatEnv(cfg, SceneSpec(scene, populate=placement_populate(specs, rng, placed, float(ds["min_separation_m"]))),
                     out / "work", seed=seed)
    frames = []
    (out / "images").mkdir(parents=True, exist_ok=True)
    try:
        for g in placed:
            vps = compute_view_points(env, g, success_radius_m=float(vp["success_radius_m"]) * 2, grid_step_m=0.1,
                                      min_pixels=int(vp["min_pixels"]), max_snap_m=float(vp["max_snap_m"]))
            for j in rng.choice(len(vps), size=min(per_object, len(vps)), replace=False) if vps else []:
                p = vps[int(j)]
                here, centre = np.asarray(habitat_to_map(p)), np.asarray(habitat_to_map(g.position))
                yaw = math.atan2(*(centre - here)[::-1]) + float(rng.uniform(-0.3, 0.3))
                frames.append(("object", g.category, env.render_at(p, yaw)["rgb"]))
        for _ in range(n_random):
            frames.append(("random", "", env.render_at(env.random_navigable_point(), float(rng.uniform(-math.pi, math.pi)))["rgb"]))
    finally:
        env.close()
    manifest = []
    for i, (kind, cat, rgb) in enumerate(frames):
        name = f"images/sim_{i:03d}.png"
        Image.fromarray(np.asarray(rgb)[..., :3]).save(out / name)
        manifest.append({"image": name, "kind": f"sim_{kind}", "object": cat, "queries": queries})
    for photo in sorted((PILOT / "data" / "scenes").glob("*.jpg")):
        q = PHOTO_QUERIES.get(photo.stem)
        if q:
            manifest.append({"image": str(photo), "kind": "photo", "object": q, "queries": [q]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} images -> {out / 'manifest.json'}")


# ---------------------------------------------------------------------------- detection
def image_path(d: Path, item: dict) -> Path:
    p = Path(item["image"])
    return p if p.is_absolute() else d / p


def detect_hf(d: Path, items: list) -> dict:
    sys.path.insert(0, str(REPO))
    from PIL import Image

    from kznav.perception.base import create_detector
    from kznav.run.config import compose_config

    spec = dict(compose_config(["detector=grounding_dino_tiny"])["detector"])
    # the D-009 settings, not today's config (D-054 threshold and box filter, D-058 query mode, D-059 resize)
    spec.update(box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, max_box_area_frac=1.0, query_mode="joint",
                input_resize="default")
    det = create_detector(spec)
    out = {}
    for it in items:
        rgb = np.asarray(Image.open(image_path(d, it)).convert("RGB"))
        out[it["image"]] = [{"box": list(x.box_xyxy), "score": x.score, "phrase": x.phrase} for x in det.detect(rgb, it["queries"])]
    return out


def detect_original(d: Path, items: list) -> dict:
    import torch
    from groundingdino.util.inference import load_image, load_model, predict

    cfg = PILOT / "third_party" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    model = load_model(str(cfg), str(PILOT / "third_party" / "weights" / "groundingdino_swint_ogc.pth"))
    out = {}
    for it in items:
        src, img = load_image(str(image_path(d, it)))
        h, w = src.shape[:2]
        caption = " . ".join(q.strip().lower() for q in it["queries"]) + " ."
        with torch.inference_mode():
            boxes, logits, phrases = predict(model=model, image=img, caption=caption, box_threshold=BOX_THRESHOLD,
                                             text_threshold=TEXT_THRESHOLD, device="cuda")
        dets = []
        for (cx, cy, bw, bh), s, ph in zip(boxes.tolist(), logits.tolist(), phrases):
            dets.append({"box": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                         "score": float(s), "phrase": ph})
        out[it["image"]] = dets
    return out


# ---------------------------------------------------------------------------- comparison
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(a: list, b: list, min_iou: float = 0.5) -> list:
    """Greedy one-to-one matching by IoU, best pairs first: [(i, j, iou)]."""
    pairs = sorted(((iou(x["box"], y["box"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if v < min_iou:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j, v))
    return out


def compare(d: Path) -> str:
    items = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    hf = json.loads((d / "detections_hf.json").read_text(encoding="utf-8"))
    org = json.loads((d / "detections_original.json").read_text(encoding="utf-8"))
    rows = {}
    for it in items:
        a, b = hf.get(it["image"], []), org.get(it["image"], [])
        m = match(a, b)
        r = rows.setdefault(it["kind"], {"images": 0, "hf": 0, "orig": 0, "matched": 0, "ious": [], "dscore": [], "same_phrase": 0})
        r["images"] += 1
        r["hf"] += len(a)
        r["orig"] += len(b)
        r["matched"] += len(m)
        r["ious"] += [v for _, _, v in m]
        r["dscore"] += [abs(a[i]["score"] - b[j]["score"]) for i, j, _ in m]
        r["same_phrase"] += sum(a[i]["phrase"].strip().lower() == b[j]["phrase"].strip().lower() for i, j, _ in m)
    tot = {k: sum(r[k] for r in rows.values()) if k not in ("ious", "dscore") else sum((r[k] for r in rows.values()), [])
           for k in ("images", "hf", "orig", "matched", "ious", "dscore", "same_phrase")}
    rows["all"] = tot
    lines = ["# GroundingDINO: HF transformers vs the original implementation (D-009)", "",
             f"Same images, captions, thresholds (box {BOX_THRESHOLD}, text {TEXT_THRESHOLD}) and SwinT-OGC weights.", "",
             "| images | n | boxes HF | boxes original | matched (IoU >= 0.5) | share of HF | share of original | "
             "median IoU | median abs score diff | same phrase |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        n = r["matched"]
        share_hf = f"{n / r['hf']:.2f}" if r["hf"] else "—"
        share_org = f"{n / r['orig']:.2f}" if r["orig"] else "—"
        med_iou = f"{np.median(r['ious']):.3f}" if n else "—"
        med_ds = f"{np.median(r['dscore']):.3f}" if n else "—"
        lines.append(f"| {k} | {r['images']} | {r['hf']} | {r['orig']} | {n} | {share_hf} | {share_org} | {med_iou} | "
                     f"{med_ds} | {r['same_phrase']}/{n} |")
    t = rows["all"]
    ok = (t["hf"] and t["orig"] and t["matched"] / t["hf"] >= 0.9 and t["matched"] / t["orig"] >= 0.9
          and np.median(t["ious"]) >= 0.9 and np.median(t["dscore"]) <= 0.05)
    lines += ["", f"Criterion (fixed in advance): >= 90% of each side's boxes matched, median IoU >= 0.9, median score "
              f"difference <= 0.05 -> **{'met' if ok else 'NOT met'}**."]
    text = "\n".join(lines) + "\n"
    (d / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--out", required=True)
    f.add_argument("--per-object", type=int, default=12)
    f.add_argument("--random", type=int, default=20)
    f.add_argument("--seed", type=int, default=0)
    dt = sub.add_parser("detect")
    dt.add_argument("--impl", required=True, choices=["hf", "original"])
    dt.add_argument("--dir", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True)
    args = ap.parse_args()
    if args.cmd == "frames":
        make_frames(Path(args.out), args.per_object, args.random, args.seed)
    elif args.cmd == "detect":
        d = Path(args.dir)
        items = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        out = detect_hf(d, items) if args.impl == "hf" else detect_original(d, items)
        (d / f"detections_{args.impl}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{sum(len(v) for v in out.values())} boxes on {len(out)} images -> detections_{args.impl}.json")
    else:
        print(compare(Path(args.dir)))


if __name__ == "__main__":
    main()
