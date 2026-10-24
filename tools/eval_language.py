"""Language-part measurements through the bench's backend interface (re-run of pilot checks 2 and 3).

    bash tools/wsl.sh python tools/eval_language.py parse  [--commands FILE] [hydra overrides ...]
    bash tools/wsl.sh python tools/eval_language.py verify [--manifest FILE] [--tiles 1,4,12] [hydra overrides ...]

parse:  command -> GoalHypotheses for every command; JSON validity (first try / eventual), target match
        against the lexicon (dictionary form), whether any English hypothesis names the object, latency.
        goal=top1 | mt_word | mt_command: the same measurement through a goal ablation (D-038 / D-039);
        the hypotheses scored are the ones the pipeline would use (mt_word: NLLB of the gold Kazakh word).
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
from kznav.language.base import BackendError, ListSink, TeeSink  # noqa: E402
from kznav.language.goal import make_goal_cache, parse_command  # noqa: E402
from kznav.language.lexicon import Lexicon, normalize  # noqa: E402
from kznav.language.prompts import fewshot_targets  # noqa: E402
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


def run_parse(args, cfg, backend, goals, calls) -> list:
    cmds = json.loads(Path(args.commands).read_text(encoding="utf-8"))
    lex = Lexicon.load(REPO / "lexicon" / "pilot_objects.yaml")
    # English objects of the prompt's own examples: a hypothesis naming one of them for another object is a leak (D-043)
    fewshot_en = [normalize(t, english=True) for t in fewshot_targets(backend.parse_prompt) if t.isascii()]
    rows = []
    for c in cmds:
        backend.context.episode_id = c["id"]
        t0 = time.perf_counter()
        g = parse_command(backend, calls, command_id=c["id"], text=c["command_kk"], lang="kk", category=c["target_kk"],
                          word_type=c["category"], lexicon=None, settings=goals.settings, translator=goals.translator,
                          word=c["target_kk"], word_lang="kk")
        cat_name = c["target_kk"]
        en = list(c.get("gold_en", []))
        if cat_name in lex.categories:
            en += lex.categories[cat_name].names.get("en", [])
        hyps = g.hypotheses
        hyp_gold, hyp_head = hypothesis_match(hyps, en)
        if g.effective_lang == "en":       # mt_command: the parser saw English, its target is English
            t_match = t_lex = g.parse_ok and hypothesis_match([g.target], en)[0]
        else:
            t_match = g.parse_ok and normalize(g.target) == normalize(cat_name)
            t_lex = g.parse_ok and cat_name in lex.categories and bool(lex.match_target(cat_name, g.target, "kk"))
        rows.append({
            "id": c["id"], "category": c["category"], "command": c["command_kk"], "gold_target": cat_name,
            "goal_mode": g.goal_mode, "translation": g.translation,
            "ok": g.parse_ok, "error": g.error, "attempts": g.parse_attempts, "latency_s": time.perf_counter() - t0,
            "translation_latency_s": g.translation_record.latency_total_s if g.translation_record else 0.0,
            "target": g.target, "target_match": t_match, "target_in_lexicon": t_lex,
            "n_hypotheses": len(hyps), "hypotheses": " | ".join(hyps), "hypothesis_gold_match": hyp_gold,
            "hypothesis_head_match": hyp_head, "parsed_hypotheses": " | ".join(g.parsed_hypotheses),
            "fewshot_leak": any(f in normalize(h, english=True).split() for h in hyps for f in fewshot_en)
                            and not any(normalize(e, english=True) in fewshot_en for e in en),
            "rooms": " | ".join(g.rooms),
        })
    return rows


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


def run_verify(args, cfg, spec, logger, out: Path) -> list:
    cat = {o["word_kk"]: o["category"] for o in json.loads((PILOT / "cultural_objects.json").read_text(encoding="utf-8"))}
    with open(args.manifest, encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    rows = []
    # Only the InternVL server (Qolda) has tiles; another model runs once with its own image budget (D-040).
    has_tiles = "max_num_tiles" in (spec.get("extra_body") or {})
    for tiles in ([int(t) for t in args.tiles.split(",")] if has_tiles else ["-"]):
        s = dict(spec)
        if has_tiles:
            s["extra_body"] = {**(spec.get("extra_body") or {}), "max_num_tiles": tiles}
        backend = create_backend(s, sink=logger.call_sink)
        backend.context.run_id = f"verify_tiles{tiles}" if has_tiles else "verify"
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
    args = ap.parse_intermixed_args()      # hydra overrides may follow the options
    cfg = compose_config(["backend=qolda", *args.overrides])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    goal_name = str(cfg["goal"]["name"])
    if goal_name != "multi" and args.task != "parse":
        raise SystemExit(f"goal={goal_name} applies to the parse task only")
    goal_tag = "" if goal_name == "multi" else f"_{goal_name}"
    out = REPO / "runs" / "language" / f"{args.task}{goal_tag}_{stamp}"
    logger = RunLogger(out)
    lines = [f"# Language check: {args.task} ({cfg['backend']['name']}, {cfg['backend']['model_id']}, "
             f"{cfg['backend'].get('weights_precision')}), goal={goal_name}", ""]
    if args.task == "parse":
        calls = ListSink()
        backend = create_backend(cfg["backend"], sink=TeeSink(logger.call_sink, calls))
        backend.context.run_id = f"parse{goal_tag}_{stamp}"
        backend.warmup(int(cfg["backend"].get("warmup_calls", 0)))
        goals = make_goal_cache(cfg["goal"], backend, calls, None, translator_sink=logger.call_sink)
        df = pd.DataFrame(run_parse(args, cfg, backend, goals, calls))
        df.to_csv(out / "parse.csv", index=False)
        lines += [f"parse prompt {backend.parse_prompt.ref} ({backend.parse_prompt.sha256[:12]}), commands {args.commands}", "",
                  "| group | n | valid JSON (1st try) | valid JSON (eventual) | target = dictionary form | "
                  "hypothesis names the object (phrase) | (head noun) | few-shot leak | latency |",
                  "|---|---|---|---|---|---|---|---|---|"]
        no_parse = cfg["goal"]["translation"] == "word"     # mt_word: no VLM parse, the gold word is given
        for name, g in [("all", df)] + list(df.groupby("category")):
            n = len(g)
            json_cols = "— | —" if no_parse else f"{ci(int(((g.attempts == 1) & g.ok).sum()), n)} | {ci(int(g.ok.sum()), n)}"
            target_col = "—" if no_parse else ci(int(g.target_match.sum()), n)
            lines.append(f"| {name} | {n} | {json_cols} | {target_col} | {ci(int(g.hypothesis_gold_match.sum()), n)} | "
                         f"{ci(int(g.hypothesis_head_match.sum()), n)} | {int(g.fewshot_leak.sum())}/{n} | "
                         f"{lat_stats(g.latency_s)} |")
        lines += ["", "Mismatches (target != gold dictionary form):", ""]
        for _, r in df[~df.target_match].iterrows():
            lines.append(f"- {r['id']} ({r['category']}): gold `{r['gold_target']}`, got `{r['target']}`; "
                         f"hypotheses: {r['hypotheses']}")
        lines += ["", "No hypothesis names the object (head noun):", ""]
        for _, r in df[~df.hypothesis_head_match].iterrows():
            tr = f" (translation: «{r['translation']}»)" if r["translation"] else ""
            lines.append(f"- {r['id']} ({r['category']}): `{r['gold_target']}`{tr} -> {r['hypotheses']}")
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
                                                        repo=REPO, extra={"task": args.task, "goal": cfg["goal"]}))
    print("\n".join(lines))
    print(f"\nout: {out}")


if __name__ == "__main__":
    main()
