"""Natural command bank (D-048): authoring sheet, import, validation, review export.

    python tools/commands.py sheet --lexicon lexicon/hm3d_objectnav.yaml --authors A01,A02,A03 \\
        --per-category 12 --dev-per-category 2 --out commands/natural/hm3d_objectnav_v1_sheet.csv
    python tools/commands.py import commands/natural/hm3d_objectnav_v1_sheet.csv --dataset hm3d_objectnav_v1 \\
        --lexicon lexicon/hm3d_objectnav.yaml --out commands/natural/hm3d_objectnav_v1.yaml
    python tools/commands.py validate commands/natural/hm3d_objectnav_v1.yaml --min-test 10
    python tools/commands.py export commands/natural/hm3d_objectnav_v1.yaml --out review.csv

sheet: one empty row per command to write (category, the object's names as a hint, author, split, style); the
authors fill kk / kk_ru / en, a second native speaker fills reviewer / verdict / note (docs/command_authoring.md).
import: the filled sheet (UTF-8 CSV, e.g. exported from Google Sheets) -> the YAML bank the runner reads; a row
with an ok verdict from a reviewer other than the author becomes ``approved``, ``reject`` -> ``rejected``, otherwise
``draft``. Re-importing replaces the bank (the sheet is the source of truth; keep it under version control).
validate: schema and content checks plus a coverage table (commands per category and split, authors, how often
each language names the object with a dictionary form); exit code 1 if there are errors.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.language.commands import FORMAT, STYLES, NaturalCommands  # noqa: E402
from kznav.language.lexicon import Lexicon  # noqa: E402

SHEET_COLUMNS = ["id", "category", "word_type", "hint_kk", "hint_ru", "hint_en", "author", "split", "style",
                 "kk", "kk_ru", "en", "reviewer", "verdict", "note"]


def sheet(lexicon: Path, authors, per_category: int, dev_per_category: int, out: Path) -> int:
    lex = Lexicon.load(REPO / lexicon if not lexicon.is_absolute() else lexicon)
    rows = []
    for cat in lex.categories.values():
        for i in range(per_category):
            author = authors[i % len(authors)]
            rows.append({
                "id": f"{cat.name}-{author.lower()}-{i // len(authors) + 1:02d}", "category": cat.name,
                "word_type": cat.word_type, "hint_kk": ", ".join(cat.names.get("kk", [])),
                "hint_ru": ", ".join(cat.names.get("ru", [])), "hint_en": ", ".join(cat.names.get("en", [])),
                "author": author, "split": "dev" if i < dev_per_category else "test",
                "style": STYLES[i % len(STYLES)], "kk": "", "kk_ru": "", "en": "", "reviewer": "", "verdict": "",
                "note": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:      # BOM: Excel opens Cyrillic correctly
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def import_sheet(src: Path, dataset: str, lexicon: str, out: Path, min_reviews: int,
                 languages=("kk", "kk_ru", "en")) -> int:
    df = pd.read_csv(src, dtype=str, encoding="utf-8-sig").fillna("")
    missing = {"id", "category", "author", "split", *languages} - set(df.columns)
    if missing:
        raise SystemExit(f"{src}: missing columns {sorted(missing)}")
    cmds = []
    for r in df.to_dict("records"):
        if not any(r.get(lang, "").strip() for lang in languages):
            continue                                                    # not written yet
        reviews = []
        if r.get("reviewer", "").strip():
            reviews.append({"reviewer": r["reviewer"].strip(), "verdict": r.get("verdict", "").strip().lower(),
                            "note": r.get("note", "").strip()})
        ok = [x for x in reviews if x["verdict"] == "ok" and x["reviewer"] != r["author"].strip()]
        status = "rejected" if any(x["verdict"] == "reject" for x in reviews) else (
            "approved" if len(ok) >= min_reviews and not any(x["verdict"] == "fix" for x in reviews) else "draft")
        cmds.append({"id": r["id"].strip(), "category": r["category"].strip(), "author": r["author"].strip(),
                     "split": r["split"].strip(), "style": r.get("style", "direct").strip() or "direct",
                     "texts": {lang: " ".join(r.get(lang, "").split()) for lang in languages},
                     "reviews": reviews, "status": status})
    bank = {"format": FORMAT, "dataset": dataset, "lexicon": lexicon, "languages": list(languages),
            "min_reviews": min_reviews, "source_sheet": str(src), "commands": cmds}
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Natural command bank imported from {src.name} by tools/commands.py - edit the sheet, not this file.\n"
              f"# Authoring and review rules: docs/command_authoring.md (D-048).\n")
    out.write_text(header + yaml.safe_dump(bank, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return len(cmds)


def validate(bank: Path, min_test: int) -> int:
    nc = NaturalCommands(bank, REPO, role="debug")
    issues = nc.validate(categories=list(nc.lexicon.categories), min_test_per_category=min_test)
    cov = pd.DataFrame(nc.coverage())
    print(f"# {bank}: {len(nc.commands)} commands, {len({c.author for c in nc.commands})} authors\n")
    if len(cov):
        print(cov.to_markdown(index=False), "\n")
    for i in issues:
        print(f"{i.level.upper():8s} {i.command_id}: {i.message}")
    n_err = sum(1 for i in issues if i.level == "error")
    print(f"\n{n_err} error(s), {len(issues) - n_err} warning(s)")
    return 1 if n_err else 0


def export(bank: Path, out: Path) -> int:
    nc = NaturalCommands(bank, REPO, role="debug")
    rows = []
    for c in nc.commands:
        last = c.reviews[-1] if c.reviews else {}
        names = nc.lexicon.categories[c.category].names if c.category in nc.lexicon.categories else {}
        rows.append({"id": c.id, "category": c.category,
                     "word_type": nc.lexicon.categories[c.category].word_type if c.category in nc.lexicon.categories else "",
                     "hint_kk": ", ".join(names.get("kk", [])), "hint_ru": ", ".join(names.get("ru", [])),
                     "hint_en": ", ".join(names.get("en", [])), "author": c.author, "split": c.split, "style": c.style,
                     **{lang: c.texts.get(lang, "") for lang in ("kk", "kk_ru", "en")},
                     "reviewer": last.get("reviewer", ""), "verdict": last.get("verdict", ""), "note": last.get("note", "")})
    pd.DataFrame(rows, columns=SHEET_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--lexicon", required=True)
    s.add_argument("--authors", required=True, help="comma-separated anonymised author ids, e.g. A01,A02,A03")
    s.add_argument("--per-category", type=int, default=12)
    s.add_argument("--dev-per-category", type=int, default=2)
    s.add_argument("--out", required=True)
    i = sub.add_parser("import")
    i.add_argument("sheet")
    i.add_argument("--dataset", required=True)
    i.add_argument("--lexicon", required=True)
    i.add_argument("--min-reviews", type=int, default=1)
    i.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("bank")
    v.add_argument("--min-test", type=int, default=10, help="approved test commands wanted per category")
    e = sub.add_parser("export")
    e.add_argument("bank")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "sheet":
        n = sheet(Path(args.lexicon), [a.strip() for a in args.authors.split(",") if a.strip()], args.per_category,
                  args.dev_per_category, Path(args.out))
        print(f"{n} rows to write -> {args.out}")
    elif args.cmd == "import":
        n = import_sheet(Path(args.sheet), args.dataset, args.lexicon, Path(args.out), args.min_reviews)
        print(f"{n} commands -> {args.out}")
    elif args.cmd == "validate":
        raise SystemExit(validate(Path(args.bank), args.min_test))
    else:
        print(f"{export(Path(args.bank), Path(args.out))} commands -> {args.out}")


if __name__ == "__main__":
    main()
