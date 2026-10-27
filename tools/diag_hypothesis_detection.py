"""D-058: GroundingDINO scores for the bare label vs the VLM hypotheses, on the robot camera's frames (dev only).

    bash tools/wsl.sh python tools/diag_hypothesis_detection.py frames                 # habitat: render the frames
    bash tools/start_qolda.sh && bash tools/wsl.sh python tools/diag_hypothesis_detection.py parse --backend qolda
    bash tools/start_qwen.sh  && bash tools/wsl.sh python tools/diag_hypothesis_detection.py parse --backend qwen3vl_4b
    bash tools/wsl.sh python tools/diag_hypothesis_detection.py detect                 # GPU, no VLM server
    bash tools/wsl.sh python tools/diag_hypothesis_detection.py report                 # -> runs/diag_hyp_detect/report.md

  frames: re-render frames at the oracle run's decision poses (dev split, D-054 oracle arm): 'pos' = the oracle saw
          the goal (n_detections > 0), 'neg' = it did not.
  parse:  parse the 12 dev commands x {kk, kk_ru, en} with the running VLM server.
  detect: GroundingDINO (query_mode joint, so a multi-query caption is the old joint caption) at a low threshold;
          per frame and caption the best matched-box score (area filter as configured). Caption families:
          label:category / label:vlfm (bare names), hyp:<backend>:<lang>:<cmd> (all hypotheses in one caption),
          one:<backend>:<lang>:<cmd>:<j> (each hypothesis alone - what query_mode separate computes).
  report: detection rate on 'pos' frames and false-positive rate on 'neg' frames at 0.35 / 0.5 / 0.65.
  detectors [--names a,b]: D-060 candidates (CANDIDATES), every query alone; boxes scored on/off the goal mask.
  detectors-report: the D-060 rule (TPR at FPR 10% for the method's use) -> runs/diag_hyp_detect/detectors_report.md
  masks [--segmenter sam2_tiny]: SAM masks vs the box on goal-visible frames -> runs/diag_hyp_detect/masks_<seg>.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs" / "diag_hyp_detect"
OUT.mkdir(parents=True, exist_ok=True)
ORACLE = REPO / "runs" / "matrix" / "dev_detector" / "oracle" / "seed0" / "events.jsonl"
VLFM_NAMES = {"chair": "chair", "bed": "bed", "plant": "potted plant", "toilet": "toilet", "tv_monitor": "tv",
              "sofa": "couch"}     # VLFM's HM3D names


def frames(per_ep_pos: int, per_ep_neg: int) -> None:
    """Render robot-camera frames of dev episodes: poses where the oracle saw the goal and poses where it did not."""
    from PIL import Image

    from kznav.run.benchmark import lookup_object
    from kznav.run.config import compose_config
    from kznav.sim.env import HabitatEnv, SceneSpec
    from kznav.sim.frames import map_to_habitat
    from kznav.sim.objectnav_dataset import load_objectnav_v1, read_id_list, scenes_of_ids, select_episodes

    cfg = compose_config(["dataset=hm3d_objectnav_v1_train", "embodiment=wheeled_vacuum_kinematic"])
    ds = cfg["dataset"]
    subset = read_id_list(REPO / "splits" / "hm3d_v1_dev.txt")
    eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"],
                            scenes=scenes_of_ids(subset))
    eps = {e.episode_id: e for e in select_episodes(eps, subset=subset)}
    poses = defaultdict(lambda: {"pos": [], "neg": []})
    for line in open(ORACLE, encoding="utf-8"):
        d = json.loads(line)
        if d["type"] == "decision" and d["episode_id"] in eps:
            poses[d["episode_id"]]["pos" if d["n_detections"] > 0 else "neg"].append(d["pose"])
    rng = np.random.default_rng(0)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    manifest = []
    by_scene = defaultdict(list)
    for eid, e in eps.items():
        by_scene[e.scene_path].append(e)
    for scene, group in sorted(by_scene.items()):
        env = HabitatEnv(cfg, SceneSpec(scene, ds.get("scene_dataset_config")), OUT / "work", seed=0)
        try:
            objects = env.semantic_objects()
            for e in group:
                goal_ids = [o.semantic_id for o in (lookup_object(objects, g.object_id) for g in e.goals) if o is not None]
                h = float(e.start_position[1])
                for kind, n in (("pos", per_ep_pos), ("neg", per_ep_neg)):
                    ps = poses[e.episode_id][kind]
                    if not ps:
                        continue
                    idx = np.unique(np.linspace(0, len(ps) - 1, n).round().astype(int)) if kind == "pos" else \
                        rng.choice(len(ps), size=min(n, len(ps)), replace=False)
                    for i in idx:
                        x, y, yaw = ps[int(i)]
                        p = env.snap(map_to_habitat(x, y, h))
                        obs = env.render_at(p, yaw)
                        rgb = np.asarray(obs["rgb"])[..., :3]
                        name = f"images/{e.episode_id.replace(':', '_')}_{kind}_{int(i)}.png"
                        Image.fromarray(rgb).save(OUT / name)
                        goal = np.isin(np.asarray(obs["semantic"]), goal_ids)      # pixels of any goal instance
                        Image.fromarray((goal * 255).astype(np.uint8)).save(OUT / name.replace(".png", "_goal.png"))
                        manifest.append({"image": name, "episode_id": e.episode_id, "category": e.category, "kind": kind,
                                         "goal_mask": name.replace(".png", "_goal.png"), "goal_pixels": int(goal.sum())})
        finally:
            env.close()
        print(scene.split("/")[-1], len(manifest), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} frames")


def parse(backend_name: str) -> None:
    """Parse the dev commands with one backend and save the hypotheses."""
    from kznav.language.base import ListSink
    from kznav.language.commands import load_commands
    from kznav.language.goal import GoalCache
    from kznav.language.registry import create_backend
    from kznav.run.config import compose_config

    cfg = compose_config(["dataset=hm3d_objectnav_v1_train", f"backend={backend_name}"])
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=calls)
    backend.warmup(int(cfg["backend"].get("warmup_calls", 0)))
    cmds = load_commands(REPO / cfg["dataset"]["commands"], REPO, role="dev")
    cache = GoalCache(backend, calls, cmds.lexicon)
    out = []
    for c in cmds.commands:
        if c.split != "dev":
            continue
        for lang in ("kk", "kk_ru", "en"):
            g = cache.get(command_id=c.id, text=c.texts[lang], lang=lang, category=c.category, word_type="universal")
            out.append({"backend": backend_name, "command_id": c.id, "category": c.category, "lang": lang,
                        "text": c.texts[lang], "parse_ok": g.parse_ok, "target": g.target, "hypotheses": g.hypotheses,
                        "weights": g.weights})
            print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    (OUT / f"parses_{backend_name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def detect() -> None:
    """Score each frame with GroundingDINO for the bare label, the VLFM name and the parsed hypotheses."""
    from PIL import Image

    from kznav.perception.base import create_detector
    from kznav.run.config import compose_config

    spec = dict(compose_config(["detector=grounding_dino_tiny"])["detector"])
    spec.update(box_threshold=0.15, query_mode="joint", input_resize="default")    # as measured for D-058
    det = create_detector(spec)
    items = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    parses = []
    for p in sorted(OUT.glob("parses_*.json")):
        parses += json.loads(p.read_text(encoding="utf-8"))
    caps = defaultdict(dict)      # category -> caption key -> queries
    for cat, name in VLFM_NAMES.items():
        caps[cat]["label:category"] = [cat]
        caps[cat]["label:vlfm"] = [name]
    for p in parses:
        if p["parse_ok"] and p["hypotheses"]:
            caps[p["category"]][f"hyp:{p['backend']}:{p['lang']}:{p['command_id']}"] = p["hypotheses"]
            for j, h in enumerate(p["hypotheses"]):
                caps[p["category"]][f"one:{p['backend']}:{p['lang']}:{p['command_id']}:{j}"] = [h]
    res = []
    for k, it in enumerate(items):
        rgb = np.asarray(Image.open(OUT / it["image"]).convert("RGB"))
        for key, qs in caps[it["category"]].items():
            ds = det.detect(rgb, qs)
            res.append({"image": it["image"], "key": key, "best": max((d.score for d in ds), default=0.0)})
        if k % 50 == 0:
            print(k, len(items), flush=True)
    (OUT / "scores.json").write_text(json.dumps({"captions": {c: v for c, v in caps.items()}, "scores": res},
                                                ensure_ascii=False, indent=0), encoding="utf-8")


def report() -> None:
    """Detection rates and median scores per query family, written to ``report.md``."""
    import pandas as pd

    items = {it["image"]: it for it in json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))}
    data = json.loads((OUT / "scores.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(data["scores"])
    df["kind"] = df.image.map(lambda i: items[i]["kind"])
    df["category"] = df.image.map(lambda i: items[i]["category"])
    df["family"] = df.key.map(lambda k: ":".join(k.split(":")[:2]) if k.startswith("label") else
                              ":".join(k.split(":")[:3]))
    lines = ["# GroundingDINO on robot-camera frames: bare label vs VLM hypotheses", "",
             f"frames: {df[df.kind == 'pos'].image.nunique()} pos (oracle sees the goal), "
             f"{df[df.kind == 'neg'].image.nunique()} neg", ""]
    for t in (0.35, 0.5, 0.65):
        g = df.assign(hit=df.best >= t).groupby(["family", "kind"]).hit.mean().unstack()
        lines += [f"## threshold {t}", "", g.round(3).to_markdown(), ""]
    g = df[df.kind == "pos"].assign(hit=lambda d: d.best >= 0.65).groupby(["category", "family"]).hit.mean().unstack()
    lines += ["## pos detection rate at 0.65 by category", "", g.round(2).to_markdown(), ""]
    med = df.groupby(["family", "kind"]).best.median().unstack()
    lines += ["## median best score", "", med.round(3).to_markdown(), ""]
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------------------- D-060: which open-vocabulary detector
# Candidates load through the same class (AutoModelForZeroShotObjectDetection, GroundingDINO-style processor).
CANDIDATES = {
    "gdino_tiny": "IDEA-Research/grounding-dino-tiny",
    "gdino_base": "IDEA-Research/grounding-dino-base",
    "mmgdino_tiny": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
    "mmgdino_base": "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
    "llmdet_tiny": "iSEE-Laboratory/llmdet_tiny",
    "llmdet_base": "iSEE-Laboratory/llmdet_base",
}
ON_GOAL = 0.25          # a box is on the goal when >= 25% of its pixels belong to a goal instance (fixed in advance)
BENCH_LANGS = ("kk", "en")


def bench_queries() -> dict:
    """category -> {"label": the category name, "parses": [(backend, lang, cmd, [hypotheses])]} for kk and en."""
    parses = []
    for p in sorted(OUT.glob("parses_*.json")):
        parses += json.loads(p.read_text(encoding="utf-8"))
    out = {cat: {"label": cat, "parses": []} for cat in VLFM_NAMES}
    for p in parses:
        if p["parse_ok"] and p["hypotheses"] and p["lang"] in BENCH_LANGS:
            out[p["category"]]["parses"].append((p["backend"], p["lang"], p["command_id"], list(p["hypotheses"])))
    return out


def detectors(names: list) -> None:
    """Every frame x every query alone (separate mode, native input, box filter 0.8), low threshold: per box its score
    and the share of its pixels on a goal instance. -> det_<name>.json"""
    from PIL import Image

    from kznav.perception.base import create_detector
    from kznav.run.config import compose_config

    items = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert all("goal_mask" in it for it in items), "re-render the frames (frames) to get goal masks"
    qs = bench_queries()
    for name in names:
        path = OUT / f"det_{name}.json"
        if path.exists():
            print(f"{name}: done already", flush=True)
            continue
        spec = dict(compose_config(["detector=grounding_dino_tiny"])["detector"])
        spec.update(model_id=CANDIDATES[name], box_threshold=0.15, query_mode="separate", input_resize="native",
                    max_box_area_frac=0.8)
        det = create_detector(spec)
        res, lat = {}, []
        for k, it in enumerate(items):
            rgb = np.asarray(Image.open(OUT / it["image"]).convert("RGB"))
            goal = np.asarray(Image.open(OUT / it["goal_mask"])) > 0
            cat = qs[it["category"]]
            queries = sorted({cat["label"]} | {h for *_, hyps in cat["parses"] for h in hyps})
            per_q = {}
            for q in queries:
                boxes = []
                for d in det.detect(rgb, [q]):
                    x1, y1, x2, y2 = (int(round(v)) for v in d.box_xyxy)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    region = goal[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
                    boxes.append([round(d.score, 4), round(float(region.mean()) if region.size else 0.0, 4)])
                per_q[q] = boxes
                lat.append(det.last_latency_s)
            res[it["image"]] = per_q
            if k % 100 == 0:
                print(f"{name} {k}/{len(items)}", flush=True)
        path.write_text(json.dumps({"model_id": CANDIDATES[name], "latency_median_s": float(np.median(lat)),
                                    "results": res}, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: median {np.median(lat) * 1000:.0f} ms per query", flush=True)
        del det
        import torch
        torch.cuda.empty_cache()


def _rates(scores_tp: np.ndarray, scores_fp: np.ndarray, pos: np.ndarray, grid: np.ndarray):
    tpr = np.array([(scores_tp[pos] >= t).mean() for t in grid])
    fpr = np.array([(scores_fp >= t).mean() for t in grid])
    return tpr, fpr


def tpr_at_fpr(tpr: np.ndarray, fpr: np.ndarray, target: float) -> float:
    """Best detection rate among thresholds whose false-positive frame rate is <= target."""
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def detectors_report() -> None:
    """Rule (D-060, fixed before the numbers): primary = TPR at FPR 10% for the method's use (a frame x parse is a hit
    when any of the parse's hypotheses has a box ON the goal at the threshold; a false positive when any has a box OFF
    the goal), averaged over Qolda and Qwen parses, kk and en. TPR over goal-visible frames, FPR over all frames.
    Winner: the highest primary; within 2 points the faster one wins."""
    import pandas as pd

    items = {it["image"]: it for it in json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))}
    qs = bench_queries()
    grid = np.round(np.arange(0.15, 0.951, 0.01), 2)
    rows = []
    for name in CANDIDATES:
        path = OUT / f"det_{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        res = data["results"]
        imgs = list(res)
        pos = np.array([items[i]["kind"] == "pos" and items[i]["goal_pixels"] > 0 for i in imgs])

        def best(i, queries):
            tp = max((s for q in queries for s, g in res[i][q] if g >= ON_GOAL), default=0.0)
            fp = max((s for q in queries for s, g in res[i][q] if g < ON_GOAL), default=0.0)
            return tp, fp

        r = {"detector": name, "ms/query": round(data["latency_median_s"] * 1000)}
        lab = np.array([best(i, [qs[items[i]["category"]]["label"]]) for i in imgs])
        t, f = _rates(lab[:, 0], lab[:, 1], pos, grid)
        r["label TPR@FPR10"] = tpr_at_fpr(t, f, 0.10)
        r["label TPR@FPR5"] = tpr_at_fpr(t, f, 0.05)
        for backend in ("qolda", "qwen3vl_4b"):
            tp_all, fp_all, pos_all = [], [], []
            for i, p in zip(imgs, pos):
                for b, _, _, hyps in qs[items[i]["category"]]["parses"]:
                    if b == backend:
                        tp, fp = best(i, hyps)
                        tp_all.append(tp), fp_all.append(fp), pos_all.append(p)
            t, f = _rates(np.array(tp_all), np.array(fp_all), np.array(pos_all), grid)
            r[f"{backend} TPR@FPR10"] = tpr_at_fpr(t, f, 0.10)
            r[f"{backend} TPR@FPR5"] = tpr_at_fpr(t, f, 0.05)
        r["primary"] = (r["qolda TPR@FPR10"] + r["qwen3vl_4b TPR@FPR10"]) / 2
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("primary", ascending=False)
    top = df.iloc[0]
    near = df[df.primary >= top.primary - 0.02]
    winner = near.sort_values("ms/query").iloc[0].detector
    text = "\n".join([
        "# D-060: open-vocabulary detectors on the robot camera (dev frames)", "",
        f"frames: {int(sum(items[i]['kind'] == 'pos' for i in items))} goal-visible, {len(items)} in all; a box is on "
        f"the goal when >= {ON_GOAL:.0%} of its pixels are goal pixels; queries one at a time, native input, boxes <= 80%.",
        "", df.round(3).to_markdown(index=False), "",
        f"Rule: highest primary (method use, TPR at FPR 10%, Qolda+Qwen, kk+en); within 2 points the faster -> "
        f"**{winner}**"])
    (OUT / "detectors_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def masks(segmenter: str = "sam2_tiny", pad: float = 0.10) -> None:
    """D-060: is a SAM mask closer to the object than the box itself? Per goal-visible frame the largest goal
    component's box, padded by ``pad`` of its size on each side (a detector box is rarely tight), prompts the
    segmenter. Precision = share of the mask on the goal; recall = share of the goal inside the box that the mask
    covers. The box as a mask (the pipeline without a segmenter) has recall 1 and precision = goal share of the box."""
    import pandas as pd
    from PIL import Image
    from scipy import ndimage

    from kznav.perception.base import create_segmenter
    from kznav.run.config import compose_config

    seg = create_segmenter(compose_config([f"segmenter={segmenter}"])["segmenter"])
    items = [it for it in json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
             if it["kind"] == "pos" and it.get("goal_pixels", 0) >= 200]
    rows = []
    for it in items:
        rgb = np.asarray(Image.open(OUT / it["image"]).convert("RGB"))
        goal = np.asarray(Image.open(OUT / it["goal_mask"])) > 0
        lab, n = ndimage.label(goal)
        if n == 0:
            continue
        comp = lab == (1 + int(np.argmax(ndimage.sum(goal, lab, range(1, n + 1)))))
        ys, xs = np.nonzero(comp)
        h, w = goal.shape
        dx, dy = pad * (xs.max() - xs.min() + 1), pad * (ys.max() - ys.min() + 1)
        box = (max(xs.min() - dx, 0), max(ys.min() - dy, 0), min(xs.max() + 1 + dx, w), min(ys.max() + 1 + dy, h))
        inbox = np.zeros_like(goal)
        inbox[int(box[1]):int(np.ceil(box[3])), int(box[0]):int(np.ceil(box[2]))] = True
        target = goal & inbox
        m = seg.segment(rgb, box)
        r = {"category": it["category"], "box_precision": target.sum() / inbox.sum(), "ms": seg.last_latency_s * 1000}
        if m is None:
            r.update(sam_precision=np.nan, sam_recall=0.0, sam_iou=0.0, no_mask=True)
        else:
            inter = (m & target).sum()
            r.update(sam_precision=inter / m.sum(), sam_recall=inter / target.sum(),
                     sam_iou=inter / (m | target).sum(), no_mask=False)
        rows.append(r)
    df = pd.DataFrame(rows)
    by = df.groupby("category")[["box_precision", "sam_precision", "sam_recall", "sam_iou"]].mean()
    by.loc["all"] = df[["box_precision", "sam_precision", "sam_recall", "sam_iou"]].mean()
    text = "\n".join([f"# D-060: {segmenter} masks vs the box, {len(df)} goal-visible frames (box padded {pad:.0%})", "",
                      by.round(3).to_markdown(), "",
                      f"empty masks (fallback to the box): {int(df.no_mask.sum())}; "
                      f"latency median {df.ms.median():.0f} ms (first box of a frame: image encoding + decoding)"])
    (OUT / f"masks_{segmenter}.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def closed_set(detector: str = "rtdetr_coco") -> None:
    """D-062: the closed-set COCO detector on the same frames, bare category label only (it reads no language):
    TPR at FPR 5% / 10% (as in detectors-report) and the rates at its configured threshold."""
    from PIL import Image

    from kznav.perception.base import create_detector
    from kznav.run.config import compose_config

    spec = dict(compose_config([f"detector={detector}"])["detector"])
    at = float(spec["box_threshold"])
    spec["box_threshold"] = 0.15
    det = create_detector(spec)
    items = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    tp, fp, pos, lat = [], [], [], []
    for it in items:
        rgb = np.asarray(Image.open(OUT / it["image"]).convert("RGB"))
        goal = np.asarray(Image.open(OUT / it["goal_mask"])) > 0
        s_tp = s_fp = 0.0
        for d in det.detect(rgb, [it["category"]]):
            x1, y1, x2, y2 = (int(round(v)) for v in d.box_xyxy)
            region = goal[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
            if region.size and region.mean() >= ON_GOAL:
                s_tp = max(s_tp, d.score)
            else:
                s_fp = max(s_fp, d.score)
        tp.append(s_tp), fp.append(s_fp), pos.append(it["kind"] == "pos" and it["goal_pixels"] > 0)
        lat.append(det.last_latency_s)
    tp, fp, pos = np.array(tp), np.array(fp), np.array(pos)
    grid = np.round(np.arange(0.15, 0.951, 0.01), 2)
    t, f = _rates(tp, fp, pos, grid)
    text = (f"# D-062: {detector} (closed set, bare label) on the robot-camera dev frames\n\n"
            f"TPR@FPR10 {tpr_at_fpr(t, f, 0.10):.3f}, TPR@FPR5 {tpr_at_fpr(t, f, 0.05):.3f}; at its threshold {at}: "
            f"TPR {(tp[pos] >= at).mean():.3f}, FPR {(fp >= at).mean():.3f}; median {np.median(lat) * 1000:.0f} ms per frame\n")
    (OUT / f"closed_set_{detector}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "parse", "detect", "report", "detectors", "detectors-report", "masks",
                                    "closed-set"])
    ap.add_argument("--detector", default="rtdetr_coco")
    ap.add_argument("--segmenter", default="sam2_tiny")
    ap.add_argument("--backend", default="qolda")
    ap.add_argument("--pos", type=int, default=3)
    ap.add_argument("--neg", type=int, default=1)
    ap.add_argument("--names", default=",".join(CANDIDATES), help="detectors: comma-separated candidate keys")
    a = ap.parse_args()
    {"frames": lambda: frames(a.pos, a.neg), "parse": lambda: parse(a.backend), "detect": detect,
     "report": report, "detectors": lambda: detectors(a.names.split(",")),
     "detectors-report": detectors_report, "masks": lambda: masks(a.segmenter),
     "closed-set": lambda: closed_set(a.detector)}[a.cmd]()
