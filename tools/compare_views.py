"""Render the wheeled robot's camera next to the standard Habitat agent's from the same spots.

    bash tools/wsl.sh python tools/compare_views.py --scene <glb> [--n 3] [--out views.png]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.config import compose_config  # noqa: E402
from kznav.sim.env import HabitatEnv, SceneSpec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="views.png")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One Habitat simulator per process at a time (two live GL contexts abort on close).
    rng = np.random.default_rng(args.seed)
    wheeled = HabitatEnv(compose_config(["embodiment=wheeled_vacuum"]), SceneSpec(args.scene), Path("/tmp/views_w"), seed=0)
    wheeled.sim.pathfinder.seed(args.seed)
    poses, low = [], []
    for _ in range(args.n):
        p = wheeled.random_navigable_point()
        yaw = float(rng.uniform(-math.pi, math.pi))
        low.append(wheeled.reset(p + np.array([0, 0.005, 0]), yaw).rgb)
        poses.append((p, yaw))
    wheeled.close()
    agent = HabitatEnv(compose_config(["embodiment=habitat_agent"]), SceneSpec(args.scene), Path("/tmp/views_a"), seed=0)
    high = [agent.reset(agent.snap(p), yaw).rgb for p, yaw in poses]
    agent.close()
    fig, axes = plt.subplots(args.n, 2, figsize=(10, 3.8 * args.n), squeeze=False)
    for i in range(args.n):
        axes[i, 0].imshow(low[i]); axes[i, 0].set_title("wheeled robot, camera 0.15 m, HFOV 69")
        axes[i, 1].imshow(high[i]); axes[i, 1].set_title("Habitat agent, camera 0.88 m, HFOV 79")
        for a in axes[i]:
            a.set_axis_off()
    fig.tight_layout()
    fig.savefig(args.out, dpi=90)
    print("saved", args.out)


if __name__ == "__main__":
    main()
