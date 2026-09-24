"""Demo video: the wheeled robot drives navmesh routes in a scene (stage-3 simulator demo).

    bash tools/wsl.sh python tools/demo_drive.py [--scene GLB] [--routes 3] [--speedup 2]

This is NOT an ObjectNav episode: there is no goal object, no perception and no language model
yet. The robot follows the shortest path on its own navmesh (privileged information) with the
pure-pursuit controller, under full Bullet physics. The video shows what the simulator produces:
the low camera, depth, and the top-down map with the planned path and the driven trajectory.

Output: runs/demo/<UTC timestamp>/{demo.mp4, preview.png, manifest.json}
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.policy.controller import PurePursuit, PurePursuitParams  # noqa: E402
from kznav.report.video import Mp4Writer, TopDownMap, compose, depth_to_rgb  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402
from kznav.sim.env import HabitatEnv, SceneSpec  # noqa: E402
from kznav.sim.frames import habitat_to_map  # noqa: E402
from kznav.sim.mesh_drive import sample_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="scene_datasets/habitat-test-scenes/apartment_1.glb",
                    help="GLB path, absolute or relative to paths.data_root")
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--min-m", type=float, default=4.0)
    ap.add_argument("--max-m", type=float, default=9.0)
    ap.add_argument("--speedup", type=float, default=2.0, help="playback speed relative to simulated time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=360, help="panel height in the video, px")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = compose_config(["embodiment=wheeled_vacuum", *args.overrides])
    vcfg = yaml.safe_load((REPO / "configs/validation/mesh_drive.yaml").read_text(encoding="utf-8"))
    scene = Path(args.scene).expanduser()
    if not scene.is_absolute():
        scene = Path(cfg["paths"]["data_root"]).expanduser() / scene
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / "demo" / stamp
    emb = cfg["embodiment"]
    ctrl_params = PurePursuitParams.from_config(cfg["policy"]["controller"])
    max_depth = float(emb["sensors"]["rgbd"]["depth"]["max_m"])
    radius = float(emb["body"]["radius_m"])

    env = HabitatEnv(cfg, SceneSpec(str(scene)), out / "work", seed=args.seed)
    writer = Mp4Writer(out / "demo.mp4", fps=args.speedup / float(emb["physics"]["control_dt_s"]))
    summary = []
    try:
        plan_pf = env.make_pathfinder(float(vcfg["planning_margin_m"]))
        lo, _ = env.sim.pathfinder.get_bounds()
        floor_y = float(env.snap(env.random_navigable_point())[1])
        mpp = 0.03
        topdown = TopDownMap(np.asarray(env.sim.pathfinder.get_topdown_view(mpp, floor_y), dtype=bool),
                             float(lo[0]), float(lo[2]), mpp)
        rng = np.random.default_rng(args.seed)
        pairs = sample_pairs(env, args.routes, args.min_m, args.max_m, rng, plan_pf)
        for ri, (start, goal, geo) in enumerate(pairs):
            path3 = env.shortest_path(start, goal, plan_pf)
            path2 = [habitat_to_map(p) for p in path3]
            d = np.asarray(path2[1]) - np.asarray(path2[0])
            obs = env.reset(start + np.array([0.0, float(vcfg["spawn_lift_m"]), 0.0]), math.atan2(d[1], d[0]))
            ctrl = PurePursuit(ctrl_params)
            ctrl.set_path(path2)
            goal2 = habitat_to_map(goal)
            traj, hits, n_coll, prev = [obs.pose[:2]], [], 0, False
            budget = geo / float(vcfg["timeout_speed_mps"]) + float(vcfg["timeout_slack_s"])
            v = w = 0.0
            while env.sim_time < budget and not ctrl.done(obs.pose):
                tk = env.drive(*ctrl.command(obs.pose), 1)[0]
                v, w = tk.v_cmd, tk.w_cmd
                if tk.collision and not prev:
                    n_coll += 1
                    hits.append(tk.pose[:2])
                prev = tk.collision
                obs = env.observe()
                traj.append(obs.pose[:2])
                text = [
                    f"kznav stage-3 simulator demo | {scene.stem} | route {ri + 1}/{len(pairs)} | "
                    f"x{args.speedup:g} playback",
                    f"t={env.sim_time:5.1f} s   v={v:4.2f} m/s   w={w:+5.2f} rad/s   tilt={tk.tilt_deg:4.1f} deg   "
                    f"collisions={n_coll}",
                    "robot: 0.34 m wide, camera at 0.15 m | NOT an ObjectNav episode: following its navmesh path",
                ]
                frame = compose(
                    [obs.rgb, depth_to_rgb(obs.depth, max_depth),
                     topdown.render(path=path2, trajectory=traj, pose=obs.pose, robot_radius_m=radius,
                                    goal=goal2, collisions=hits)],
                    args.height, text,
                )
                writer.write(frame)
                if writer.n_frames == 40:
                    import imageio

                    imageio.imwrite(out / "preview.png", frame)
            arrived = ctrl.done(obs.pose)
            summary.append({"route": ri + 1, "geodesic_m": round(geo, 2), "arrived": bool(arrived),
                            "sim_time_s": round(env.sim_time, 1), "collisions": n_coll})
            print(summary[-1], flush=True)
    finally:
        writer.close()
        env.close()
    write_manifest(out / "manifest.json", build_manifest(run_id=f"demo_{stamp}", config=cfg, seed=args.seed, repo=REPO,
                                                        extra={"scene": str(scene), "routes": summary}))
    print(f"video: {out / 'demo.mp4'} ({writer.n_frames} frames)")


if __name__ == "__main__":
    main()
