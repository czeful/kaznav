"""Re-parse the commands of finished runs with another parse prompt and compare with what the runs logged (D-071).

    bash tools/wsl.sh python tools/check_parse_prompt.py --runs runs/matrix/dev_demo/qolda_kk_fix/seed0 ... \
        --out runs/language/parse_v3_qolda backend=qolda backend.prompts.parse_goal=default/parse_goal/v3

Commands: every distinct (text, language, category) of the runs' goal events (goal mode "multi" only: translated
commands are parsed from the translator's output and are left out). For each: parse ok, target in the lexicon's
dictionary form for that language, an English hypothesis naming the object - logged (old prompt) against new - and
the new prompt's ``target_en`` (present, Latin script, naming the object). The server of the backend must be up.
Prints the D-071 acceptance rule's verdict and exits 0 if it holds, 1 if not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.language.base import ListSink  # noqa: E402
from kznav.language.goal import GoalSettings, parse_command  # noqa: E402
from kznav.language.lexicon import Lexicon  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402


def logged_goals(runs) -> pd.DataFrame:
    rows = []
    for run in runs:
        for line in open(Path(run) / "events.jsonl", encoding="utf-8"):
            if '"type": "goal"' not in line:
                continue
            e = json.loads(line)
            if e.get("goal_mode", "multi") != "multi" or e.get("translation"):
                continue
            rows.append({k: e.get(k) for k in ("text", "lang", "category", "parse_ok", "target", "target_match",
                                                "hypothesis_match", "hypotheses")})
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["text", "lang", "category"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon", default="lexicon/hm3d_objectnav.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = compose_config(args.overrides)
    lex = Lexicon.load(REPO / args.lexicon)
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=calls)
    settings = GoalSettings.from_config(cfg["goal"])
    old = logged_goals(args.runs)
    rows = []
    for i, r in old.iterrows():
        g = parse_command(backend, calls, command_id=f"chk{i}", text=r.text, lang=r.lang, category=r.category,
                          word_type="universal", lexicon=lex, settings=settings)
        en = g.target_en
        rows.append({"lang": r.lang, "category": r.category, "text": r.text,
                     "old_parse_ok": bool(r.parse_ok), "new_parse_ok": g.parse_ok,
                     "old_target": r.target, "new_target": g.target,
                     "old_target_match": bool(r.target_match), "new_target_match": bool(g.target_match),
                     "old_hyp_match": bool(r.hypothesis_match), "new_hyp_match": bool(g.hypothesis_match),
                     "target_en": en, "target_en_latin": bool(en) and en.isascii(),
                     "target_en_names_object": bool(en) and bool(lex.match_any_en(r.category, [en])),
                     "new_hypotheses": " | ".join(g.hypotheses)})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "commands.csv", index=False, encoding="utf-8-sig")
    n = len(df)
    s = {k: int(df[k].sum()) for k in ("old_parse_ok", "new_parse_ok", "old_target_match", "new_target_match",
                                       "old_hyp_match", "new_hyp_match", "target_en_latin", "target_en_names_object")}
    ok = (s["new_parse_ok"] >= s["old_parse_ok"] - 1 and s["new_target_match"] >= s["old_target_match"] - 1
          and s["new_hyp_match"] >= s["old_hyp_match"] - 1 and s["target_en_latin"] >= 0.95 * n)
    by_lang = df.groupby("lang")[["new_parse_ok", "old_target_match", "new_target_match", "old_hyp_match",
                                  "new_hyp_match", "target_en_latin", "target_en_names_object"]].sum()
    lines = [f"# Parse prompt check: {cfg['backend']['prompts']['parse_goal']} vs the logged parses", "",
             f"backend {cfg['backend']['name']}; {n} distinct commands from {len(args.runs)} runs", "",
             "| | old (logged) | new |", "|---|---|---|",
             f"| parsed | {s['old_parse_ok']} | {s['new_parse_ok']} |",
             f"| target in dictionary form | {s['old_target_match']} | {s['new_target_match']} |",
             f"| a hypothesis names the object | {s['old_hyp_match']} | {s['new_hyp_match']} |",
             f"| target_en present, Latin | — | {s['target_en_latin']} |",
             f"| target_en names the object | — | {s['target_en_names_object']} |", "",
             by_lang.to_markdown(), "",
             f"D-071 rule: {'PASS' if ok else 'FAIL'}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
