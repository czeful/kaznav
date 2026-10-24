"""Cultural-object photo benchmark (option C, H2; docs/cultural_data_collection.md, D-055): the language and
verification parts of the pipeline on real photos of cultural objects and their matched universal controls.

Per language and category (test split):
* query match - the parsed command's hypotheses name the category (lexicon ``lexicon/cultural_pairs.yaml``);
* verification - P(yes) for "is this <object>?" asked in the command language with the lexicon's name of the object;
  positives are the category's photos, hard negatives the photos of its pair (the same function and shape), so the
  verifier must tell қазан from a cooking pot, not from a sofa. Accuracy at 0.5 (raw) and AUROC.

H2 is the interaction (D-055 follows analysis plan §5): per pair, [M_en - M_L](cultural) - [M_en - M_L](universal
control) for a metric M and language L; the pair is the unit (bootstrap over pairs, exact sign-flip test over pairs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import cluster_bootstrap_mean, cluster_sign_flip_test

PHOTO_COLUMNS = ("photo_id", "file", "category", "pair", "word_type", "split", "viewpoint", "instance", "source",
                 "license", "author")


def load_photos(path: Path, lexicon_categories: Mapping[str, object]) -> pd.DataFrame:
    """The photos table with checks: known categories, symmetric pairs (a cultural category paired with a universal
    one and back), split / viewpoint / word_type values, files present, a license for every photo."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    missing = set(PHOTO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    errors: List[str] = []
    for col, allowed in (("word_type", {"cultural", "universal"}), ("split", {"dev", "test"}),
                         ("viewpoint", {"low", "normal"})):
        bad = sorted(set(df[col]) - allowed)
        if bad:
            errors.append(f"{col}: unexpected {bad}")
    unknown = sorted(set(df.category) - set(lexicon_categories))
    if unknown:
        errors.append(f"categories not in the lexicon: {unknown}")
    pairs = df.groupby("category").pair.agg(lambda s: sorted(set(s)))
    for cat, ps in pairs.items():
        if len(ps) != 1:
            errors.append(f"{cat}: several pairs {ps}")
            continue
        back = pairs.get(ps[0])
        if back is None or back != [cat]:
            errors.append(f"{cat}: pair {ps[0]!r} is missing or not paired back")
    kinds = df.groupby("category").word_type.agg(lambda s: sorted(set(s)))
    for cat, ks in kinds.items():
        if len(ks) == 1 and pairs.get(cat) and pairs[cat][0] in kinds and kinds[pairs[cat][0]] == ks:
            errors.append(f"{cat} and its pair have the same word_type {ks[0]}")
    if (df.license == "").any():
        errors.append(f"{int((df.license == '').sum())} photos without a license")
    if df.photo_id.duplicated().any():
        errors.append("duplicate photo_id")
    absent = [f for f in df.file if not (Path(path).parent / f).exists()]
    if absent:
        errors.append(f"{len(absent)} files missing, e.g. {absent[:3]}")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return df


def auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """P(score of a positive > score of a negative), ties counting half (Mann-Whitney); NaN without both."""
    p = np.asarray([s for s in scores_pos if s is not None and np.isfinite(s)], dtype=float)
    n = np.asarray([s for s in scores_neg if s is not None and np.isfinite(s)], dtype=float)
    if not p.size or not n.size:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (p.size * n.size))


def verify_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (lang, target category): accuracy at 0.5 and AUROC from verification rows with columns lang, target,
    gold (bool), p_yes."""
    out = []
    for (lang, target), g in rows.groupby(["lang", "target"]):
        p = g.p_yes.astype(float)
        ok = p.notna()
        acc = float(((p[ok] >= 0.5) == g.gold[ok].astype(bool)).mean()) if ok.any() else float("nan")
        out.append({"lang": lang, "target": target, "n": int(len(g)), "missing_p": int((~ok).sum()), "accuracy": acc,
                    "auroc": auroc(p[g.gold.astype(bool) & ok], p[~g.gold.astype(bool) & ok])})
    return pd.DataFrame(out)


def pair_interaction(metric: pd.DataFrame, value: str, pairs: Mapping[str, str], word_type: Mapping[str, str],
                     lang: str, ref_lang: str = "en", n_boot: int = 10_000) -> Dict[str, float]:
    """H2 on one metric: per (cultural, universal) pair, [M_ref - M_lang](cultural) - [M_ref - M_lang](universal).
    ``metric`` has columns lang, target and ``value``. Returns the mean over pairs, its bootstrap CI over pairs and
    the exact sign-flip p over pairs."""
    m = metric.set_index(["lang", "target"])[value]
    diffs, names = [], []
    for c, u in sorted(pairs.items()):
        if word_type.get(c) != "cultural":
            continue
        try:
            gap_c = m[(ref_lang, c)] - m[(lang, c)]
            gap_u = m[(ref_lang, u)] - m[(lang, u)]
        except KeyError:
            continue
        if np.isfinite(gap_c) and np.isfinite(gap_u):
            diffs.append(float(gap_c - gap_u))
            names.append(c)
    if not diffs:
        return {"pairs": 0, "interaction": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p": float("nan")}
    d = np.asarray(diffs)
    units = np.arange(d.size)
    ci = cluster_bootstrap_mean(d, units, n_boot=n_boot) if d.size >= 2 else None
    test = cluster_sign_flip_test(d, units)
    return {"pairs": int(d.size), "interaction": float(d.mean()),
            "ci_low": ci.low if ci else float("nan"), "ci_high": ci.high if ci else float("nan"), "p": test.p_value}
