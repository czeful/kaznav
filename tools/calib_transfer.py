"""Does the verifier calibration transfer to other scenes? Fit on one group of dev runs, judge on another (D-070).

    python tools/calib_transfer.py --a runs/matrix/dev_calibration/qolda_{kk,en,kkru}/seed0 \
        --b runs/matrix/dev_demo/qolda_{kk,en,kkru}{,_fix}/seed0 --name Qolda [--out calib/transfer_qolda.md]

Group A and group B should come from different scenes. Both directions are reported: the rule (Platt + max-F1
threshold, as tools/fit_calibration.py) fitted on one group is applied to the other, next to the other group's own
in-sample fit (the best that group could have). Labels: crop content (D-042).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from fit_calibration import collect  # noqa: E402
from kznav.eval.calibration import ece_binary  # noqa: E402
from kznav.eval.classification import threshold_metrics  # noqa: E402
from kznav.policy.calibration import PlattCalibrator, choose_threshold  # noqa: E402


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y], p[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):                    # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(runs):
    p, y, _, eps, _ = collect([Path(r) for r in runs], "content")
    return np.asarray(p, float), np.asarray(y, bool), eps


def judge(cal, thr: float, p: np.ndarray, y: np.ndarray) -> dict:
    pc = cal(p)
    m = threshold_metrics(pc, y, thr)
    return {"ece": ece_binary(pc, y, 10), "precision": m.precision, "recall": m.recall, "f1": m.f1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    groups = {"A": load(args.a), "B": load(args.b)}
    fits = {}
    for g, (p, y, _) in groups.items():
        cal = PlattCalibrator.fit(p, y)
        fits[g] = (cal, choose_threshold(cal(p), y).threshold)
    lines = [f"# Calibration transfer between scene groups{': ' + args.name if args.name else ''}", "",
             f"A: {', '.join(args.a)}", f"B: {', '.join(args.b)}", "",
             "| group | calls | goal crops | base rate | AUROC raw | ECE raw |", "|---|---|---|---|---|---|"]
    for g, (p, y, eps) in groups.items():
        lines.append(f"| {g} ({len(eps)} episodes) | {len(p)} | {int(y.sum())} | {y.mean():.3f} | {auroc(p, y):.3f} | "
                     f"{ece_binary(p, y, 10):.3f} |")
    lines += ["", "| rule fitted on | judged on | a | b | threshold | ECE | precision | recall | F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for src in ("A", "B"):
        for dst in ("A", "B"):
            cal, thr = fits[src]
            r = judge(cal, thr, *groups[dst][:2])
            tag = "in-sample" if src == dst else "**transfer**"
            lines.append(f"| {src} | {dst} ({tag}) | {cal.a:.3f} | {cal.b:.3f} | {thr:.3f} | {r['ece']:.3f} | "
                         f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
