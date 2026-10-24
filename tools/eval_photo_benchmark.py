"""Cultural-object photo benchmark (option C, H2; D-055): query match and verification per language on real photos
of cultural objects and their matched universal controls (docs/cultural_data_collection.md).

    bash tools/wsl.sh python tools/eval_photo_benchmark.py --photos data/cultural_photos/photos.csv \\
        --commands commands/natural/cultural_pairs.yaml --langs kk,kk_ru,en --split test backend=qolda [goal=...]

parse:  every usable command of every category, per language -> hypotheses; query match = a hypothesis names the
        category (lexicon/cultural_pairs.yaml). goal=top1 | mt_command | mt_word: the same through a goal ablation.
verify: every photo asked about its own category (positive) and about its pair's (hard negative), per language,
        with the lexicon's name of the object in that language (kk_ru: the Russian name). Accuracy and AUROC.
H2:     per pair, (en - L) on the cultural category minus (en - L) on its universal control, for L in kk, kk_ru;
        bootstrap CI and exact sign-flip p over pairs (kznav/eval/photo_bench.py).
Output: runs/photo_bench/<UTC>/ with parse.csv, verify.csv, tables.md, vlm_calls.jsonl, manifest.json.
--split dev is for prompt / threshold development; the reported numbers come from --split test only.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.photo_bench import load_photos, pair_interaction, verify_metrics  # noqa: E402
from kznav.language.base import ListSink, TeeSink  # noqa: E402
from kznav.language.commands import WORD_LANG, NaturalCommands  # noqa: E402
from kznav.language.goal import make_goal_cache  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402
from kznav.language.schemas import VerifyQuery  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.run.logging import RunLogger  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402


def run_parse(bank: NaturalCommands, langs, split: str, goals, backend) -> pd.DataFrame:
    rows = []
    for c in bank.commands:
        if c.split != split or c.status == "rejected" or (split == "test" and c.status != "approved"):
            continue
        for lang in langs:
            text = c.texts.get(lang, "").strip()
            if not text:
                continue
            word, word_lang = bank.object_name(lang, c.category)
            backend.context.episode_id = f"{c.id}|{lang}"
            lg = goals.get(command_id=c.id, text=text, lang=lang, category=c.category,
                           word_type=bank.lexicon.categories[c.category].word_type, word=word, word_lang=word_lang)
            rows.append({"command_id": c.id, "category": c.category, "lang": lang, "text": text,
                         "parse_ok": lg.parse_ok, "target": lg.target, "hypotheses": "; ".join(lg.hypotheses),
                         "query_match": bool(lg.hypothesis_match) if lg.parse_ok else False})
    return pd.DataFrame(rows)


def run_verify(photos: pd.DataFrame, root: Path, lexicon, langs, backend) -> pd.DataFrame:
    rows = []
    for _, ph in photos.iterrows():
        img = np.asarray(Image.open(root / ph.file).convert("RGB"))
        for target, gold in ((ph.category, True), (ph.pair, False)):
            for lang in langs:
                names = lexicon.categories[target].names
                word = (names.get(WORD_LANG[lang]) or names.get("en") or [target])[0]
                backend.context.episode_id = f"{ph.photo_id}|{target}|{lang}"
                t0 = time.perf_counter()
                r = backend.verify_candidate(img, VerifyQuery(word, lang))
                rows.append({"photo_id": ph.photo_id, "photo_category": ph.category, "target": target, "lang": lang,
                             "word": word, "gold": gold, "p_yes": r.p_target, "answer": r.answer_text,
                             "viewpoint": ph.viewpoint, "latency_s": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def h2_table(metric: pd.DataFrame, value: str, pairs, word_type, langs) -> pd.DataFrame:
    rows = []
    for lang in [l for l in langs if l != "en"]:
        r = pair_interaction(metric, value, pairs, word_type, lang)
        rows.append({"metric": value, "lang": lang, **r})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--commands", required=True, help="natural command bank for the pair categories")
    ap.add_argument("--langs", default="kk,kk_ru,en")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    langs = [l for l in args.langs.split(",") if l]
    if "en" not in langs:
        raise SystemExit("H2 needs English as the reference language")
    cfg = compose_config(args.overrides)
    bank = NaturalCommands(REPO / args.commands, REPO, role="debug")
    lex = bank.lexicon
    photos = load_photos(REPO / args.photos, lex.categories)
    photos = photos[photos.split == args.split]
    out = REPO / "runs" / "photo_bench" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = RunLogger(out)
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=TeeSink(logger.call_sink, calls))
    backend.context.run_id = f"photo_bench_{out.name}"
    backend.warmup(int(cfg["backend"].get("warmup_calls", 0)))
    goals = make_goal_cache(cfg["goal"], backend, calls, lex, translator_sink=logger.call_sink)
    write_manifest(out / "manifest.json", build_manifest(run_id=backend.context.run_id, config=cfg, seed=0, repo=REPO,
                                                        extra={"photos": args.photos, "commands": args.commands,
                                                               "split": args.split, "langs": langs,
                                                               "n_photos": int(len(photos))}))
    pairs = dict(zip(photos.category, photos.pair))
    word_type = {c: lex.categories[c].word_type for c in lex.categories}
    md = [f"# Photo benchmark ({args.split}), backend {cfg['backend']['name']}, goal {cfg['goal']['name']}", ""]
    parse = run_parse(bank, langs, args.split, goals, backend)
    parse.to_csv(out / "parse.csv", index=False)
    if len(parse):
        pm = parse.groupby(["lang", "category"]).query_match.mean().rename("query_match").reset_index()
        pm = pm.rename(columns={"category": "target"})
        pm["word_type"] = pm.target.map(word_type)
        md += ["## Query match by language and word type", "",
               pm.groupby(["lang", "word_type"]).query_match.mean().unstack().to_markdown(floatfmt=".3f"), "",
               "## H2: (en - L) on cultural minus (en - L) on the control, per pair", "",
               h2_table(pm, "query_match", pairs, word_type, langs).to_markdown(index=False, floatfmt=".3f"), ""]
    if not args.skip_verify and len(photos):
        ver = run_verify(photos, (REPO / args.photos).parent, lex, langs, backend)
        ver.to_csv(out / "verify.csv", index=False)
        vm = verify_metrics(ver)
        vm["word_type"] = vm.target.map(word_type)
        vm.to_csv(out / "verify_metrics.csv", index=False)
        md += ["## Verification (positives: own photos; negatives: the pair's photos)", "",
               vm.groupby(["lang", "word_type"])[["accuracy", "auroc"]].mean().to_markdown(floatfmt=".3f"), "",
               "## H2 on verification", "",
               pd.concat([h2_table(vm, "accuracy", pairs, word_type, langs),
                          h2_table(vm, "auroc", pairs, word_type, langs)]).to_markdown(index=False, floatfmt=".3f"),
               ""]
    logger.close()
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
