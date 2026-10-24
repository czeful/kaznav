"""Compare runs on the same episodes: one table per arm + paired contrasts against a reference arm.

    python tools/compare_runs.py --ref oracle oracle=runs/debug/<stamp> gd=runs/debug/<stamp> ... [--out DIR]

Each ``label=dir`` is a run directory with episodes.csv. Arms must cover the same episodes (paired on
dataset/split/episode_id; the command may differ between arms). Output: <out>/comparison.md + .csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.aggregate import paired_contrast  # noqa: E402
from kznav.eval.stats import mean_ci, wilson_ci  # noqa: E402

PAIR = ("dataset", "split", "episode_id")


def fmt_ci(est, lo, hi, pct=False):
    if pct:
        return f"{100 * est:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"
    return f"{est:.3f} [{lo:.2f}, {hi:.2f}]"


def verifier_stats(run_dir: Path) -> dict:
    """In-loop verifier accuracy from events.jsonl: raw P(yes) >= 0.5 vs. the privileged is-goal label."""
    f = run_dir / "events.jsonl"
    tp = fn = fp = tn = 0
    if f.exists():
        for line in open(f, encoding="utf-8"):
            if '"verification"' not in line:
                continue
            e = json.loads(line)
            if e.get("type") != "verification" or not e.get("called") or e.get("p_raw") is None:
                continue
            # label by content (goal pixels in the box, D-042) when the run logged it, else by position (D-037)
            lab = e.get("is_goal_by_content")
            yes, goal = e["p_raw"] >= 0.5, bool(e["is_goal_privileged"] if lab is None else lab)
            tp += yes and goal
            fn += (not yes) and goal
            fp += yes and not goal
            tn += (not yes) and not goal
    n = tp + fn + fp + tn
    if n == 0:
        return {"verifier calls": "-", "sens (goal)": "-", "spec (non-goal)": "-"}
    return {"verifier calls": n,
            "sens (goal)": f"{tp}/{tp + fn}" if tp + fn else "-",
            "spec (non-goal)": f"{tn}/{tn + fp}" if tn + fp else "-"}


def arm_row(label: str, g: pd.DataFrame, manifest: dict, run_dir: Path) -> dict:
    n = len(g)
    sr = wilson_ci(int(g.success.sum()), n)
    spl = mean_ci(g.spl.astype(float).to_numpy(), n_boot=5000)
    lat = g.vlm_latency_total_s.astype(float)
    # schema v3 columns (D-038/D-039); older runs have no translation and score their hypotheses the same way
    mt = g["translation_latency_s"].astype(float) if "translation_latency_s" in g else pd.Series(0.0, index=g.index)
    hyp = g.hypothesis_match.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return {
        "arm": label, "n": n,
        "SR": fmt_ci(sr.estimate, sr.low, sr.high, pct=True),
        "SPL": fmt_ci(spl.estimate, spl.low, spl.high),
        "SoftSPL": f"{g.soft_spl.mean():.3f}",
        "in view": f"{g.goal_ever_in_view.mean():.0%}", "detected": f"{g.goal_ever_detected.mean():.0%}",
        "verified": f"{g.goal_ever_verified_positive.mean():.0%}" if manifest.get("verify") else "-",
        "false stops": int((g.outcome == "false_stop").sum()),
        "stuck": int((g.outcome == "stuck").sum()), "not found": int((g.outcome == "not_found").sum()),
        "step limit": int((g.outcome == "step_limit").sum()), "fell": int((g.outcome == "fell").sum()),
        "model fail": int((g.outcome == "model_failure").sum()),
        "VLM calls/ep": f"{g.vlm_calls.mean():.1f}" if g.vlm_calls.sum() else "-",
        "VLM s/ep": f"{lat.mean():.1f}" if g.vlm_calls.sum() else "-",
        "MT s/ep": f"{mt.mean():.1f}" if mt.sum() else "-",
        "query names goal": f"{hyp.mean():.0%}" if hyp.notna().any() else "-",
        "wall s/ep": f"{g.wall_time_s.mean():.0f}",
        **verifier_stats(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="label=run_dir")
    ap.add_argument("--ref", required=True, help="reference arm label for the paired contrasts")
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="Comparison")
    args = ap.parse_args()
    frames, manifests, dirs = [], {}, {}
    for a in args.arms:
        label, d = a.split("=", 1)
        d = Path(d)
        df = pd.read_csv(d / "episodes.csv")
        df["arm"] = label
        frames.append(df)
        # build_manifest merges the runner's extra fields (lang, verify, note, ...) into the top level
        manifests[label] = json.loads((d / "manifest.json").read_text(encoding="utf-8")) \
            if (d / "manifest.json").exists() else {}
        dirs[label] = str(d)
    df = pd.concat(frames, ignore_index=True)
    labels = [a.split("=", 1)[0] for a in args.arms]
    if args.ref not in labels:
        raise SystemExit(f"--ref {args.ref!r} is not one of {labels}")
    out = Path(args.out) if args.out else REPO / "runs" / "compare" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([arm_row(l, df[df.arm == l], manifests[l], Path(dirs[l])) for l in labels])
    contrasts = []
    for l in labels:
        if l == args.ref:
            continue
        for m in ("success", "spl"):
            c = paired_contrast(df.assign(success=df.success.astype(float)), arm_col="arm", arm_a=l, arm_b=args.ref,
                                metric=m, pair_on=PAIR, n_boot=5000)
            contrasts.append(c)
    ct = pd.DataFrame(contrasts)
    lines = [f"# {args.title}", "",
             "role: debug — pipeline comparison on the debug set, NOT a paper result.", "",
             "Arms (run directories):", ""] + [f"- **{l}**: `{dirs[l]}`" for l in labels] + [
             "", "SR: Wilson 95% CI; SPL: bootstrap 95% CI over episodes. Funnel columns: share of episodes "
             "where the goal was ever in view / detected (box on the goal) / accepted by the verifier. "
             "Verifier sens/spec: in-loop calls, raw P(yes) >= 0.5 vs. whether the crop shows the goal "
             "(goal pixels in the box, D-042; runs before it: the candidate's position) - privileged, analysis only. Query names goal: share of the episodes with a parsed goal "
             "whose detector queries name the goal category (lexicon synonyms); MT s/ep: translator latency "
             "(D-039), not part of VLM s/ep.", "",
             table.to_markdown(index=False), "",
             f"Paired contrasts vs **{args.ref}** (same episodes; diff = arm − ref, paired bootstrap 95% CI; "
             "exact McNemar on success):", ""]
    if len(ct):
        view = ct.assign(
            diff=lambda x: x.apply(lambda r: fmt_ci(r["diff"], r.diff_ci_low, r.diff_ci_high), axis=1),
            mcnemar=lambda x: x.apply(lambda r: f"{int(r.mcnemar_only_a)}/{int(r.mcnemar_only_b)}, p={r.mcnemar_p:.3f}"
                                      if r.metric == "success" and not pd.isna(r.get("mcnemar_p")) else "-", axis=1),
            dz=lambda x: x.cohens_dz.map(lambda v: f"{v:.2f}"))[["arm_a", "metric", "n_pairs", "mean_a", "mean_b", "diff",
                                                                  "dz", "mcnemar"]]
        lines.append(view.rename(columns={"arm_a": "arm", "mcnemar": "McNemar (only arm / only ref)"}).to_markdown(
            index=False, floatfmt=".3f"))
    by_cat = df.assign(success=df.success.astype(float)).pivot_table(
        index="arm", columns="goal_category", values="success", aggfunc=lambda s: f"{int(s.sum())}/{len(s)}")
    lines += ["", "Successes by goal category:", "", by_cat.loc[labels].to_markdown()]
    notes = [m.get("note") for m in manifests.values() if m.get("note")]
    if notes:
        lines += ["", "Notes: " + "; ".join(sorted(set(notes)))]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(out / "arms.csv", index=False)
    ct.to_csv(out / "contrasts.csv", index=False)
    df.to_csv(out / "episodes_all.csv", index=False)
    print("\n".join(lines))
    print(f"\nout: {out}")


if __name__ == "__main__":
    main()
