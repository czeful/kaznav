"""Paper tables and figures from run directories, described by a YAML spec.

    python tools/make_paper_tables.py paper/specs/main_hm3d.yaml --out paper/main_hm3d

Spec (tools/run_matrix.py writes one for a matrix):
    title: "HM3D ObjectNav v1 val, 480 episodes, wheeled robot"
    arms:                                     # one or more run directories each (one per seed)
      - {key: gd, label: "VLFM-style, English category", runs: [runs/eval/<a>]}
      - {key: qolda_kk, label: "Qolda, Kazakh", runs: [runs/eval/<b>, runs/eval/<c>], accent: true}
    contrasts: [[qolda_kk, gd], [a, b, c, d]] # a - b on the same episodes; (a - b) - (c - d) as an interaction
    decompositions: [[qolda_en, qolda_kk]]    # SR gap a - b split into failure stages (H1, D-045)
    subset_file: splits/hm3d_v1_test_secondary.txt  # optional: every arm restricted to these episodes
    sensitivity: true                         # the analysis plan's sensitivity tables (command clusters, ...)
    breakdowns: [word_type, command_lang, goal_category]

Statistics (docs/analysis_plan.md, D-047): seeds averaged per episode; scene-cluster bootstrap CIs (>= 5 scenes);
scene sign-flip permutation p-values, Holm-adjusted over the success contrasts; McNemar as a secondary column.
Output in --out: tables.md, tables.tex (booktabs), one CSV per table, sr.png/.pdf (SR with 95% CI),
stages.png/.pdf (failure stages), funnel.png/.pdf, reliability_<key>.png/.pdf for arms with a verifier. Numbers
come only from the runs' files (episodes.csv, exclusions.csv, failures.csv, events.jsonl, vlm_calls.jsonl).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval import paper  # noqa: E402
from kznav.eval.failures import STAGE_GROUP, STAGE_ORDER  # noqa: E402
from kznav.report.figures import funnel_figure, interval_bars, reliability_figure, stage_figure  # noqa: E402


def build(spec: dict, out: Path) -> list:
    """All tables (name, caption, DataFrame) and figures of a spec; returns the tables."""
    out.mkdir(parents=True, exist_ok=True)
    subset = None
    if spec.get("subset_file"):            # restrict every arm to these episodes (a nested subset, D-050)
        from kznav.sim.objectnav_dataset import read_id_list

        subset = read_id_list(REPO / spec["subset_file"])
    arms = [paper.Arm.load(a["key"], a["label"], a["runs"], a.get("group", ""), subset=subset) for a in spec["arms"]]
    accent = [a["label"] for a in spec["arms"] if a.get("accent")]
    title = spec.get("title", "Results")
    main = paper.main_table(arms)
    tables = [("main", f"Main results ({main.attrs['note']})", main),
              ("exclusions", "Episodes run, failed and excluded", paper.exclusion_table(arms))]
    if spec.get("contrasts"):
        ct = paper.contrast_table(arms, [tuple(p) for p in spec["contrasts"]])
        tables.append(("contrasts", "Paired contrasts (a − b, same episodes, seeds averaged; scene sign-flip p, "
                                    "Holm over success contrasts; McNemar secondary)", ct))
    if spec.get("sensitivity") and spec.get("contrasts"):     # pre-specified sensitivity analyses (analysis plan §6)
        pairs = [tuple(p) for p in spec["contrasts"]]
        tables.append(("contrasts_by_command", "Sensitivity: paired contrasts with the command as the unit",
                       paper.contrast_table(arms, pairs, metrics=("success",), cluster="command_id")))
        tables.append(("contrasts_no_model_errors", "Sensitivity: paired contrasts without episodes that had a failed "
                                                    "model call", paper.contrast_table(arms, pairs, metrics=("success",),
                                                                                       drop_model_errors=True)))
    st = paper.stage_table(arms)
    tables.append(("stages", "Failure stages: share of episodes (one stage per failed episode)", st))
    for a, b in spec.get("decompositions") or []:
        la, lb = (next(x.label for x in arms if x.key == k) for k in (a, b))
        variants = [("pipeline", "content", "", ""),
                    ("navigation_first", "content", "_navfirst", " (sensitivity: navigation rules first)")]
        if spec.get("sensitivity"):
            variants.append(("pipeline", "position", "_poslabel", " (sensitivity: position-based stop labels)"))
        for order, label, suffix, tag in variants:
            gd = paper.gap_decomposition(arms, a, b, order=order, stop_label=label)
            tables.append((f"gap_{a}_vs_{b}{suffix}", f"Success gap {la} − {lb} by failure stage{tag}", gd))
    for by in spec.get("breakdowns") or []:
        tables.append((f"by_{by}", f"Successes by {by}", paper.breakdown(arms, by)))
    fn = paper.funnel(arms)
    tables.append(("funnel", "Funnel: share of episodes reaching each stage", fn))
    vt = paper.verifier_table(arms)
    if len(vt):
        tables.append(("verifier", "Verifier in the loop (raw P(yes) >= 0.5)", vt))
    lt = paper.latency_table(arms)
    if len(lt):
        tables.append(("latency", "Model calls: latency per call", lt))
    md = [f"# {title}", ""]
    tex = []
    for name, caption, df in tables:
        df.to_csv(out / f"{name}.csv", index=False)
        md += [f"## {caption}", "", paper.to_markdown(df), ""]
        tex += [paper.to_latex(df, f"{title}: {caption}", f"tab:{name}"), ""]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    rows = []
    for a in arms:
        c, method = paper.sr_interval(a)
        rows.append({"arm": a.label, "estimate": 100 * c.estimate, "low": 100 * c.low, "high": 100 * c.high})
    interval_bars(out / "sr", rows, accent=accent, xlabel=f"success rate, % (95% CI, {main.attrs['ci_methods']})",
                  title=title)
    stages = [s.value for s in STAGE_ORDER]
    focus = [s.value for s in STAGE_ORDER if STAGE_GROUP[s] in ("language", "grounding")]
    stage_figure(out / "stages", {r["Arm"]: {"success": r["SR"], **{s: r.get(s, 0.0) for s in stages}}
                                  for r in st.to_dict("records")}, stages, focus,
                 title="Where episodes fail (blue: language and grounding stages)")
    fstages = [s for s, _ in paper.FUNNEL]
    funnel_figure(out / "funnel", {r["Arm"]: [r[s] for s in fstages] for r in fn.to_dict("records")}, fstages,
                  accent=accent, title="Where the chain breaks")
    for a in arms:
        p, y, label = paper.verifier_samples(a)
        if p.size and y.any() and (~y).any():
            reliability_figure(out / f"reliability_{a.key}", {f"{a.label}, raw ({label} labels)": (p, y)},
                               title=f"{a.label}: verifier reliability")
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    out = Path(args.out)
    tables = build(spec, out)
    print(f"{len(tables)} tables -> {out / 'tables.md'}, {out / 'tables.tex'}")


if __name__ == "__main__":
    main()
