"""Fit the verifier's calibration and stop threshold on DEV runs (D-028, D-042) -> the artifact reported runs load.

    python tools/fit_calibration.py runs/eval/<dev run> [more dev runs ...] --out calib/qolda_dev.json
        [--label content|position] [--method platt|isotonic|identity] [--objective max_f1|min_precision --min-precision 0.9]

Samples: every verifier call with a probability, from the runs' events.jsonl. Label: whether the crop shows the goal
(goal pixels in the detection box, logged since D-042; ``position`` = the older candidate-position label). All runs
must use the same backend, model and verification prompt, and none may be a benchmark (evaluation) run. The artifact
stores the dev episode ids; tools/run_eval.py refuses to evaluate on any of them. Next to it: <out>.md (numbers,
including out-of-fold ones) and <out>_reliability.png/.pdf.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.policy.calibration import CalibrationArtifact, fit_stop_rule  # noqa: E402
from kznav.report.figures import reliability_figure  # noqa: E402
from kznav.run.logging import completed_events  # noqa: E402


def run_identity(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    b = m.get("backend") or {}
    return {"backend": b.get("name"), "model_id": b.get("model_id"),
            "prompt_sha256": ((b.get("prompts") or {}).get("verify_candidate") or {}).get("sha256"),
            "role": m.get("role"), "split": (m.get("config") or {}).get("dataset", {}).get("split"),
            "dataset": (m.get("config") or {}).get("dataset", {}).get("name")}


def collect(runs, label: str):
    p, y, groups, episodes, missing = [], [], [], set(), 0
    for run in runs:
        for e in completed_events(run / "events.jsonl", "verification"):      # not failed attempts (D-046)
            if not e.get("called") or e.get("p_raw") is None:
                continue
            lab = e.get("is_goal_by_content") if label == "content" else e.get("is_goal_privileged")
            if lab is None:
                missing += 1
                continue
            p.append(float(e["p_raw"]))
            y.append(bool(lab))
            groups.append(f"{run.name}/{e['episode_id']}")
        csv = run / "episodes.csv"
        if csv.exists():
            episodes |= set(pd.read_csv(csv)["episode_id"].astype(str))
    return p, y, groups, sorted(episodes), missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="content", choices=["content", "position"])
    ap.add_argument("--method", default="platt", choices=["platt", "isotonic", "identity"])
    ap.add_argument("--objective", default="max_f1", choices=["max_f1", "min_precision"])
    ap.add_argument("--min-precision", type=float, default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs]
    ids = [run_identity(r) for r in runs]
    keys = {(i["backend"], i["model_id"], i["prompt_sha256"]) for i in ids}
    if len(keys) != 1 or None in next(iter(keys)):
        raise SystemExit(f"runs must share one backend / model / verification prompt; got {sorted(map(str, keys))}")
    if any(i["role"] in ("benchmark", "smoke") for i in ids):      # smoke = val scenes (D-052)
        raise SystemExit("benchmark (evaluation) and smoke (val scenes) runs cannot be used for calibration: fit on dev "
                         "runs only")
    p, y, groups, episodes, missing = collect(runs, args.label)
    if missing:
        print(f"note: {missing} calls without a {args.label} label were skipped")
    if not p:
        raise SystemExit("no labelled verifier calls in these runs")
    fit = fit_stop_rule(p, y, groups, method=args.method, objective=args.objective, min_precision=args.min_precision,
                        folds=args.folds)
    backend, model_id, sha = next(iter(keys))
    roles = sorted({str(i["role"]) for i in ids})
    out = Path(args.out)
    art = CalibrationArtifact(
        calibrator=fit.calibrator.to_dict(), threshold=asdict(fit.threshold), backend=backend, model_id=model_id,
        prompt_sha256=sha, dev_split=",".join(sorted({f"{i['dataset']}/{i['split']}" for i in ids})),
        dev_episode_ids=episodes, n_samples=len(p), base_rate=fit.base_rate,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=f"label={args.label}; method={args.method}; objective={args.objective}; roles={roles}; "
              f"runs={[str(r) for r in runs]}" + ("; DEBUG-ONLY (debug runs)" if "debug" in roles else ""),
        extra={k: float(v) for k, v in fit.metrics.items()})
    art.save(out)
    reliability_figure(out.with_name(out.stem + "_reliability"), {"raw P(yes)": (p, y),
                       f"{args.method}": (list(fit.calibrator(p)), y)},
                       title=f"{backend}: verifier calibration on dev ({len(p)} calls)")
    m = fit.metrics
    lines = [f"# Calibration {out.name}", "", f"backend {backend} ({model_id}), verify prompt {sha[:12]}, "
             f"runs: {', '.join(str(r) for r in runs)} ({', '.join(roles)})", "",
             f"{m['n']} calls, {m['n_pos']} show the goal (base rate {fit.base_rate:.3f}); "
             f"{len(episodes)} dev episodes; label: {args.label}", "",
             "| | ECE | Brier |", "|---|---|---|",
             f"| raw P(yes) | {m['ece_raw']:.3f} | {m['brier_raw']:.3f} |",
             f"| {args.method}, fitted and judged on the same calls | {m['ece_fit']:.3f} | {m['brier_fit']:.3f} |",
             f"| {args.method}, out of fold ({args.folds} folds by episode) | {m['oof_ece']:.3f} | — |", "",
             f"Stop threshold ({args.objective}): {fit.threshold.threshold:.3f} — precision {fit.threshold.precision:.3f}, "
             f"recall {fit.threshold.recall:.3f} on the fitting calls; out of fold: precision {m['oof_precision']:.3f}, "
             f"recall {m['oof_recall']:.3f} ({m['oof_n']} calls, {m['oof_folds_skipped']} folds skipped).", ""]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"artifact: {out}")


if __name__ == "__main__":
    main()
