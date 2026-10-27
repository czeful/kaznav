"""Run the robot physics plausibility suite and write a report.

    bash tools/wsl.sh python tools/validate_robot.py [--only straight wall ...] [hydra overrides ...]

Output: runs/validation/<UTC timestamp>/{report.md, results.json, manifest.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.config import compose_config  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402
from kznav.sim.validation import Suite, results_json, to_markdown  # noqa: E402


def main() -> None:
    """Command-line entry point: run the suite, write the report, results and manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="scenario names to run")
    ap.add_argument("--validation-config", default=str(REPO / "configs/validation/robot.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. embodiment.drive.wheel_friction=0.6")
    args = ap.parse_args()

    cfg = compose_config(["embodiment=wheeled_vacuum", *args.overrides])
    vcfg = yaml.safe_load(Path(args.validation_config).read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else REPO / "runs" / "validation" / stamp
    out.mkdir(parents=True, exist_ok=True)
    results = Suite(cfg, vcfg, out / "work").run(args.only)
    (out / "report.md").write_text(to_markdown(results), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results_json(results), indent=2, default=str), encoding="utf-8")
    write_manifest(out / "manifest.json", build_manifest(run_id=f"validation_{stamp}", config=cfg, seed=int(cfg["run"]["seed"]),
                                                        repo=REPO, extra={"validation_config": vcfg}))
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name:14s} ({r.wall_s:.1f} s)")
        for c in r.checks:
            print(f"    {'ok ' if c.passed else 'BAD'} {c.name}: {c.detail}")
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
