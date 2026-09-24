"""D-003: drive the wheeled robot along its own navmesh paths on scanned meshes and count failures.

    bash tools/wsl.sh python tools/mesh_drive_test.py [--episodes N] [hydra overrides ...]

Output: runs/mesh_drive/<UTC timestamp>/{summary.md, episodes.json, <scene>.png, manifest.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.config import compose_config  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402
from kznav.sim.mesh_drive import episode_dicts, run_scene, summarize  # noqa: E402


def plot(path: Path, eps, topdown) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, lo, mpp = topdown["grid"], topdown["lower"], topdown["mpp"]
    fig, ax = plt.subplots(figsize=(8, 8 * grid.shape[0] / max(grid.shape[1], 1)))
    ax.imshow(grid, cmap="Greys_r", alpha=0.4, origin="upper")
    colors = {"arrived": "tab:green", "stuck": "tab:red", "timeout": "tab:orange"}

    def to_px(x, y_map):
        return (x - lo[0]) / mpp, (-y_map - lo[2]) / mpp

    for e in eps:
        if not e.trajectory:
            continue
        pts = np.array([to_px(x, y) for x, y in e.trajectory])
        ax.plot(pts[:, 0], pts[:, 1], color=colors[e.outcome], lw=1)
        if e.stuck_at:
            px = to_px(e.stuck_at[0], -e.stuck_at[2])
            ax.plot(*px, "x", color="black", ms=8)
    ax.set_title(f"{eps[0].scene if eps else ''}: green arrived, red stuck (x), orange timeout")
    ax.set_axis_off()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation-config", default=str(REPO / "configs/validation/mesh_drive.yaml"))
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = compose_config(["embodiment=wheeled_vacuum", *args.overrides])
    vcfg = yaml.safe_load(Path(args.validation_config).read_text(encoding="utf-8"))
    if args.episodes:
        vcfg["episodes_per_scene"] = args.episodes
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / "mesh_drive" / stamp
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(vcfg["seed"]))
    all_eps, lines = [], ["# Mesh drive test (D-003)", "",
                          "| scene | n | arrived | stuck | timeout | driven m | collisions/100 m | stuck/100 m | median clearance at stuck (m) |",
                          "|---|---|---|---|---|---|---|---|---|"]
    infos = []
    for rel in vcfg["scenes"]:
        scene_path = str(Path(cfg["paths"]["data_root"]) / rel)
        eps, info = run_scene(cfg, vcfg, scene_path, out / "work" / Path(rel).stem, rng)
        s = summarize(eps)
        plot(out / f"{Path(rel).stem}.png", eps, info.pop("topdown"))
        infos.append({**info, **s})
        all_eps += eps
        lines.append(f"| {Path(rel).stem} | {s['n']} | {s.get('arrived')} | {s.get('stuck')} | {s.get('timeout')} | "
                     f"{s.get('driven_m')} | {s.get('collision_events_per_100m')} | {s.get('stuck_episodes_per_100m')} | "
                     f"{s.get('median_stuck_clearance_m')} |")
        print(lines[-1], flush=True)
    s = summarize(all_eps)
    lines.append(f"| **all** | {s['n']} | {s['arrived']} | {s['stuck']} | {s['timeout']} | {s['driven_m']} | "
                 f"{s['collision_events_per_100m']} | {s['stuck_episodes_per_100m']} | {s['median_stuck_clearance_m']} |")
    lines += ["", "Failed episodes:", ""]
    for e in all_eps:
        if not e.arrived:
            lines.append(f"- {e.scene} #{e.index}: {e.outcome}, geodesic {e.geodesic_m} m, final dist {e.final_dist_m} m, "
                         f"stuck clearance {e.stuck_clearance_m}, links {e.stuck_links}, max tilt {e.max_tilt_deg} deg")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "episodes.json").write_text(json.dumps({"scenes": infos, "episodes": episode_dicts(all_eps)}, indent=1), encoding="utf-8")
    write_manifest(out / "manifest.json", build_manifest(run_id=f"mesh_drive_{stamp}", config=cfg, seed=int(vcfg["seed"]),
                                                        repo=REPO, extra={"validation_config": vcfg}))
    print("\n".join(lines[-(len(all_eps) - s['arrived'] + 3):]))
    print(f"summary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
