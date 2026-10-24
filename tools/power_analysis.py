"""Sample size for the paper's primary contrasts (D-049): run BEFORE the test runs, put the result into
docs/analysis_plan.md, and do not change n after seeing test data.

    python tools/power_analysis.py table --p-a 0.30 --psi 0.30 --deltas 0.10,0.15,0.20 --m 5 \\
        --sec-per-episode 200 --arms 10 --out paper/power
    python tools/power_analysis.py estimate runs/eval/<arm a> runs/eval/<arm b>        # pilot / dev estimates

table: for each effect size (absolute SR difference) the smallest n reaching 80% power with the primary test
(scene sign-flip permutation, seeds averaged, alpha / m), its power curve over n, the analytic McNemar n for
comparison (no clustering), and the compute it costs (n x arms x seeds x seconds per episode).
estimate: p_a, delta, discordance psi and - with >= 5 scenes - the scene SDs, from two paired runs (dev split
runs, never the test runs whose n is being planned).
Model and assumptions: kznav/eval/power.py.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.power import PowerParams, estimate_from_pairs, mcnemar_n, simulate_power  # noqa: E402

N_GRID = [100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000]


def table(args) -> pd.DataFrame:
    rows, curves = [], {}
    for delta in [float(x) for x in args.deltas.split(",")]:
        base = PowerParams(p_a=args.p_a, delta=delta, psi=max(args.psi, abs(delta)), n=max(N_GRID[0], args.k_scenes),
                           k_scenes=args.k_scenes, seeds=args.seeds, seed_flip=args.seed_flip,
                           sigma_scene=args.sigma_scene, tau=args.tau, alpha=args.alpha, m=args.m)
        curve, n_req = [], None
        for n in [x for x in N_GRID if x <= args.max_n]:
            r = simulate_power(PowerParams(**{**asdict(base), "n": n}), n_sims=args.sims, n_perm=args.perms, seed=args.seed)
            curve.append((n, r["power"], r["power_mcnemar"]))
            if n_req is None and r["power"] >= args.target:
                n_req = n
        curves[delta] = curve
        n_mc = mcnemar_n(args.p_a, delta, base.psi, alpha=args.alpha, power=args.target, m=args.m)
        hours = (n_req or float("nan")) * args.arms * args.seeds * args.sec_per_episode / 3600
        rows.append({"delta (SR points)": f"{100 * delta:.0f}", "psi": f"{base.psi:.2f}",
                     f"n for {100 * args.target:.0f}% power (sign-flip, {args.k_scenes} scenes)":
                         n_req if n_req is not None else f"> {args.max_n}",
                     "n (McNemar, no clustering)": n_mc,
                     "power at max n": f"{curve[-1][1]:.2f}",
                     f"GPU-hours ({args.arms} arms x {args.seeds} seed(s) x {args.sec_per_episode:.0f} s)":
                         f"{hours:.0f}" if n_req else "—"})
    out = pd.DataFrame(rows)
    if args.out:
        o = Path(args.out)
        o.mkdir(parents=True, exist_ok=True)
        out.to_csv(o / "power.csv", index=False)
        pd.DataFrame([{"delta": d, "n": n, "power": p, "power_mcnemar": q} for d, c in curves.items() for n, p, q in c]
                     ).to_csv(o / "power_curves.csv", index=False)
        from kznav.report.figures import ACCENT, GREY, _despine, _plt, save

        plt = _plt()
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for i, (d, c) in enumerate(curves.items()):
            col = ACCENT if i == 0 else GREY
            ax.plot([x[0] for x in c], [x[1] for x in c], "o-", ms=3, color=col, lw=1.2 if i == 0 else 0.9,
                    alpha=1.0 if i == 0 else 0.6 + 0.2 * i / len(curves), label=f"Δ = {100 * d:.0f} points")
        ax.axhline(args.target, color="#bdbdbd", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("episodes per arm")
        ax.set_ylabel("power")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Scene sign-flip test, α = {args.alpha}/{args.m}, SR(a) = {args.p_a:.2f}", fontsize=9, loc="left")
        _despine(ax)
        save(fig, o / "power")
        params = {k: v for k, v in vars(args).items() if k not in ("cmd", "out")}
        (o / "power.md").write_text(f"# Power analysis\n\nParameters: {params}\n\n{out.to_markdown(index=False)}\n",
                                    encoding="utf-8")
    return out


def estimate(run_a: Path, run_b: Path) -> dict:
    a = pd.read_csv(run_a / "episodes.csv")
    b = pd.read_csv(run_b / "episodes.csv")
    key = ["dataset", "split", "episode_id"]
    a, b = a[a.seed == a.seed.min()], b[b.seed == b.seed.min()]
    j = a.merge(b, on=key, suffixes=("_a", "_b"))
    if not len(j):
        raise SystemExit("the two runs share no episodes")
    return estimate_from_pairs(j.success_a.astype(bool), j.success_b.astype(bool), j.scene_id_a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--p-a", type=float, default=0.30, help="SR of the reference arm")
    t.add_argument("--psi", type=float, default=0.30, help="discordance P(A != B) between the two arms")
    t.add_argument("--deltas", default="0.10,0.15,0.20")
    t.add_argument("--m", type=int, default=5, help="number of primary contrasts (Holm family)")
    t.add_argument("--alpha", type=float, default=0.05)
    t.add_argument("--target", type=float, default=0.8)
    t.add_argument("--k-scenes", type=int, default=20)
    t.add_argument("--seeds", type=int, default=1)
    t.add_argument("--seed-flip", type=float, default=0.0, help="P(a seed changes an episode's outcome)")
    t.add_argument("--sigma-scene", type=float, default=0.10)
    t.add_argument("--tau", type=float, default=0.05)
    t.add_argument("--max-n", type=int, default=2000)
    t.add_argument("--sims", type=int, default=800)
    t.add_argument("--perms", type=int, default=4000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--arms", type=int, default=10)
    t.add_argument("--sec-per-episode", type=float, default=200.0)
    t.add_argument("--out", default="")
    e = sub.add_parser("estimate")
    e.add_argument("run_a")
    e.add_argument("run_b")
    args = ap.parse_args()
    if args.cmd == "table":
        print(table(args).to_markdown(index=False))
    else:
        for k, v in estimate(Path(args.run_a), Path(args.run_b)).items():
            print(f"{k:12s} {v:.3f}" if isinstance(v, float) else f"{k:12s} {v}")


if __name__ == "__main__":
    main()
