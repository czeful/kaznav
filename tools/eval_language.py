"""Language-part measurements through the bench's backend interface (re-run of pilot checks 2 and 3).

    bash tools/wsl.sh python tools/eval_language.py parse  [--commands FILE] [hydra overrides ...]
    bash tools/wsl.sh python tools/eval_language.py verify [--manifest FILE] [--tiles 1,4,12] [hydra overrides ...]

parse:  command -> GoalHypotheses for every command; JSON validity (first try / eventual), target match
        against the lexicon (dictionary form), whether any English hypothesis names the object, latency.
verify: every image crop -> P(yes) from first-token logprobs; accuracy, precision/recall, ECE (binary and
        pilot-compatible top-label), Brier, latency — per number of image tiles (D-007).
Everything the model returns is logged raw to vlm_calls.jsonl. Output: runs/language/<UTC timestamp>/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.calibration import brier_score, ece_binary, ece_top_label  # noqa: E402
from kznav.eval.classification import threshold_metrics  # noqa: E402
from kznav.eval.stats import wilson_ci  # noqa: E402
from kznav.language.base import BackendError  # noqa: E402
from kznav.language.lexicon import Lexicon, normalize  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402
from kznav.language.schemas import VerifyQuery  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.run.logging import RunLogger  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402

PILOT = REPO / "qolda check" / "data"


def ci(k: int, n: int) -> str:
    c = wilson_ci(k, n)
    return f"{c.estimate:.3f} [{c.low:.2f}, {c.high:.2f}]"


def lat_stats(xs):
    a = np.asarray(xs, dtype=float)
    return f"median {np.median(a):.2f} s, p95 {np.percentile(a, 95):.2f} s (n={a.size})"


def run_parse(args, cfg, backend, out: Path) -> list:
    cmds = json.loads(Path(args.commands).read_text(encoding="utf-8"))
    lex = Lexicon.load(REPO / "lexicon" / "pilot_objects.yaml")
    rows = []
    for c in cmds:
        backend.context.episode_id = c["id"]
        t0 = time.perf_counter()
        try:
            g = backend.parse_goal(c["command_kk"], "kk")
            ok, err = True, ""
        except BackendError as e:
            g, ok, err = None, False, str(e)[:200]
        cat_name = c["target_kk"]
        en = list(c.get("gold_en", []))
        if cat_name in lex.categories:
            en += lex.categories[cat_name].names.get("en", [])
        hyps = g.texts if g else []
        hyp_gold, hyp_head = hypothesis_match(hyps, en)
        rows.append({
            "id": c["id"], "category": c["category"], "command": c["command_kk"], "gold_target": cat_name,
            "ok": ok, "error": err, "attempts": last_attempts, "latency_s": time.perf_counter() - t0,
            "target": g.target if g else "", "target_match": bool(g) and normalize(g.target) == normalize(cat_name),
            "target_in_lexicon": bool(g) and cat_name in lex.categories and bool(lex.match_target(cat_name, g.target, "kk")),
            "n_hypotheses": len(hyps), "hypotheses": " | ".join(hyps), "hypothesis_gold_match": hyp_gold,
            "hypothesis_head_match": hyp_head,
            "rooms": " | ".join(g.rooms) if g else "",
        })
    return rows


last_attempts = 0


# A hypothesis's object phrase ends before the first of these ("curtain hanging on a wall" -> "curtain").
_PHRASE_END = {"on", "in", "at", "near", "with", "of", "by", "under", "over", "inside", "against", "beside",
               "next", "behind", "stacked", "hanging", "leaning", "lying", "standing", "draped", "made", "filled", "placed"}


def head_noun(phrase: str) -> str:
    words = phrase.split()
    for i, w in enumerate(words):
        if w in _PHRASE_END and i > 0:
            words = words[:i]
            break
    return words[-1] if words else ""


def hypothesis_match(hyps, names_en):
    """(strict, head): strict = some English name occurs as a whole phrase in a hypothesis;
    head = the head noun of some hypothesis equals the head noun of some name ("green blanket" names a
    "quilted blanket", "curtain hanging on a wall" does not name a "wall hanging"). A heuristic: it
    accepts a wrong object with the same head noun ("chest of drawers" for "chest")."""
    names = [normalize(n, english=True) for n in names_en if n.strip()]
    hs = [normalize(h, english=True) for h in hyps]
    strict = any(f" {n} " in f" {h} " for n in names for h in hs)
    heads = {head_noun(n) for n in names}
    head = any(head_noun(h) in heads for h in hs)
    return strict, head


class CountingSink:
    """Wraps the JSONL sink and remembers the number of JSON attempts of the last call."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, rec):
        global last_attempts
        last_attempts = sum(1 for a in rec.attempts if a.kind == "json")
        self.inner(rec)


def run_verify(args, cfg, spec, logger, out: Path) -> list:
    cat = {o["word_kk"]: o["category"] for o in json.loads((PILOT / "cultural_objects.json").read_text(encoding="utf-8"))}
    with open(args.manifest, encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    rows = []
    for tiles in [int(t) for t in args.tiles.split(",")]:
        s = dict(spec)
        s["extra_body"] = {**(spec.get("extra_body") or {}), "max_num_tiles": tiles}
        backend = create_backend(s, sink=logger.call_sink)
        backend.context.run_id = f"verify_tiles{tiles}"
        backend.warmup(int(spec.get("warmup_calls", 0)))
        for it in items:
            img = np.asarray(Image.open(Path(args.manifest).parent.parent / it["image_path"]).convert("RGB"))
            t0 = time.perf_counter()
            try:
                r = backend.verify_candidate(img, VerifyQuery(it["object_kk"], "kk"))
                p, src, cov, text = r.p_target, r.source, r.coverage, r.answer_text
            except BackendError as e:
                p, src, cov, text = None, "error", None, str(e)[:100]
            rows.append({"tiles": tiles, "image": it["image_path"], "object_kk": it["object_kk"],
                         "category": cat.get(it["object_kk"], "?"),
                         "gold": it["gold_label"].strip().lower() in ("yes", "1", "true", "иә"),
                         "p_yes": p, "source": src, "coverage": cov, "answer": text,
                         "latency_s": time.perf_counter() - t0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["parse", "verify"])
    ap.add_argument("--commands", default=str(PILOT / "commands_kk.json"))
    ap.add_argument("--manifest", default=str(PILOT / "crops_manifest.csv"))
    ap.add_argument("--tiles", default="1,4,12")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = compose_config(["backend=qolda", *args.overrides])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / "language" / f"{args.task}_{stamp}"
    logger = RunLogger(out)
    lines = [f"# Language check: {args.task} ({cfg['backend']['name']}, {cfg['backend']['model_id']}, "
             f"{cfg['backend'].get('weights_precision')})", ""]
    if args.task == "parse":
        backend = create_backend(cfg["backend"], sink=CountingSink(logger.call_sink))
        backend.context.run_id = f"parse_{stamp}"
        backend.warmup(int(cfg["backend"].get("warmup_calls", 0)))
        df = pd.DataFrame(run_parse(args, cfg, backend, out))
        df.to_csv(out / "parse.csv", index=False)
        lines += ["| group | n | valid JSON (1st try) | valid JSON (eventual) | target = dictionary form | "
                  "hypothesis names the object (phrase) | (head noun) | latency |", "|---|---|---|---|---|---|---|---|"]
        for name, g in [("all", df)] + list(df.groupby("category")):
            n = len(g)
            lines.append(f"| {name} | {n} | {ci(int(((g.attempts == 1) & g.ok).sum()), n)} | {ci(int(g.ok.sum()), n)} | "
                         f"{ci(int(g.target_match.sum()), n)} | {ci(int(g.hypothesis_gold_match.sum()), n)} | "
                         f"{ci(int(g.hypothesis_head_match.sum()), n)} | {lat_stats(g.latency_s)} |")
        lines += ["", "Mismatches (target != gold dictionary form):", ""]
        for _, r in df[~df.target_match].iterrows():
            lines.append(f"- {r['id']} ({r['category']}): gold `{r['gold_target']}`, got `{r['target']}`; "
                         f"hypotheses: {r['hypotheses']}")
        lines += ["", "No hypothesis names the object (head noun):", ""]
        for _, r in df[~df.hypothesis_head_match].iterrows():
            lines.append(f"- {r['id']} ({r['category']}): `{r['gold_target']}` -> {r['hypotheses']}")
    else:
        df = pd.DataFrame(run_verify(args, cfg, cfg["backend"], logger, out))
        df.to_csv(out / "verify.csv", index=False)
        lines += ["| tiles | group | n | accuracy | precision | recall | ECE binary | ECE top-label | Brier | "
                  "logprobs available | coverage median | latency |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for tiles, gt in df.groupby("tiles"):
            for name, g in [("all", gt)] + list(gt.groupby("category")):
                v = g[g.p_yes.notna()]
                m = threshold_metrics(v.p_yes, v.gold, 0.5) if len(v) else None
                lines.append(
                    f"| {tiles} | {name} | {len(g)} | {ci(int(m.tp + m.tn), m.n) if m else '-'} | "
                    f"{m.precision:.3f} | {m.recall:.3f} | {ece_binary(v.p_yes, v.gold):.3f} | "
                    f"{ece_top_label(v.p_yes, v.gold):.3f} | {brier_score(v.p_yes, v.gold):.3f} | "
                    f"{len(v)}/{len(g)} | {v.coverage.median():.3f} | {lat_stats(g.latency_s)} |")
    logger.close()
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(out / "manifest.json", build_manifest(run_id=out.name, config=cfg, seed=int(cfg["run"]["seed"]),
                                                        repo=REPO, extra={"task": args.task}))
    print("\n".join(lines))
    print(f"\nout: {out}")


if __name__ == "__main__":
    main()
