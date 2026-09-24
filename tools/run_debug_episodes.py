"""Run ObjectNav episodes on the DEBUG dataset (test scenes + placed chairs with known ids).

    bash tools/wsl.sh python tools/run_debug_episodes.py [--videos 2] [hydra overrides ...]
    e.g.  ... detector=oracle_semantic embodiment=habitat_agent dataset.episodes_per_scene=10

Numbers from this tool are pipeline checks (role: debug), never paper results.
Output: runs/debug/<UTC timestamp>/{episodes.csv, summary.md, episodes.json, vlm_calls.jsonl, events.jsonl,
        manifest.json, video_*.mp4}
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.aggregate import summarize  # noqa: E402
from kznav.eval.records import column_names  # noqa: E402
from kznav.perception.base import create_detector, create_segmenter  # noqa: E402
from kznav.perception.camera import Intrinsics  # noqa: E402
from kznav.policy.objectnav import ObjectNavPolicy  # noqa: E402
from kznav.report.video import Mp4Writer, compose, depth_to_rgb, render_policy_map  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.run.episode import Executor, run_episode  # noqa: E402
from kznav.run.logging import RunLogger  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402
from kznav.sim.env import HabitatEnv, SceneSpec  # noqa: E402
from kznav.sim.episodes import ObjectSpec, compute_view_points, placement_populate, sample_episodes, save_episodes  # noqa: E402


class VideoRecorder:
    def __init__(self, path: Path, cfg, label: str):
        self.w = Mp4Writer(path, fps=4.0)
        self.max_depth = float(cfg["embodiment"]["sensors"]["rgbd"]["depth"]["max_m"])
        self.radius = float(cfg["embodiment"]["body"]["radius_m"])
        self.label = label

    def __call__(self, *, obs, decision, policy, trace, step, episode):
        fr_cells = [f.cell for f in policy.last_frontiers]
        mp = render_policy_map(policy.map, pose=obs.pose, radius_m=self.radius,
                               trajectory=[p[:2] for p in trace.poses], path=decision.path_xy,
                               frontier_cells=fr_cells, candidates=policy.memory.candidates,
                               gt_boxes=[(g.aabb_min, g.aabb_max) for g in episode.goals])
        text = [f"{self.label} | {episode.episode_id} | goal: {episode.goal_category} | step {step}  t={obs.sim_time:.1f}s",
                f"decision: {decision.kind} - {decision.reason}",
                f"detections: {len(policy.last_detections)}  candidates: {len(policy.memory.candidates)}  "
                f"frontiers: {len(policy.last_frontiers)}   map: free/occupied/unknown, cyan=frontier, green=GT goal"]
        self.w.write(compose([obs.rgb, depth_to_rgb(obs.depth, self.max_depth), mp], 360, text))

    def close(self):
        self.w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, default=2, help="record the first N episodes")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = compose_config(["dataset=debug_chairs", "detector=oracle_semantic", *args.overrides])
    ds, emb = cfg["dataset"], cfg["embodiment"]
    if ds["role"] != "debug":
        raise SystemExit("this tool is for debug datasets only")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"debug_{emb['name']}_{cfg['detector']['type']}_{stamp}"
    out = REPO / "runs" / "debug" / stamp
    logger = RunLogger(out)
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    rgbd = emb["sensors"]["rgbd"]
    K = Intrinsics.from_hfov(float(rgbd["hfov_deg"]), int(rgbd["resolution_hw"][1]), int(rgbd["resolution_hw"][0]))
    rows, all_eps = [], []
    for rel in ds["scenes"]:
        scene = str(Path(cfg["paths"]["data_root"]) / rel)
        placed = []
        specs = [ObjectSpec(o["category"], str(o["config_path"]), int(o["count"])) for o in ds["objects"]]
        populate = placement_populate(specs, rng, placed, float(ds["min_separation_m"]))
        env = HabitatEnv(cfg, SceneSpec(scene, populate=populate), out / "work" / Path(rel).stem,
                         seed=int(cfg["run"]["seed"]))
        try:
            vp = cfg["eval"]["view_points"]
            for g in placed:
                g.view_points = compute_view_points(env, g, success_radius_m=float(vp["success_radius_m"]),
                                                    grid_step_m=float(vp["grid_step_m"]),
                                                    min_pixels=int(vp["min_pixels"]), max_snap_m=float(vp["max_snap_m"]),
                                                    success_distance_m=float(ds["success_distance_m"]))
                print(f"{g.object_id}: {len(g.view_points)} view points", flush=True)
            lo, hi = env.sim.pathfinder.get_bounds()
            fy = float(env.snap(env.random_navigable_point())[1])
            np.savez_compressed(out / f"navmesh_{Path(rel).stem}.npz", grid=np.asarray(
                env.sim.pathfinder.get_topdown_view(0.05, fy), dtype=bool), lower=np.asarray(lo), mpp=0.05)
            eps = []
            for cat in sorted({g.category for g in placed}):
                eps += sample_episodes(env, placed, cat, int(ds["episodes_per_scene"]), rng,
                                       min_geodesic_m=float(ds["min_geodesic_m"]), max_geodesic_m=float(ds["max_geodesic_m"]),
                                       scene_id=rel, dataset=ds["name"], split=ds["split"], embodiment=emb["name"])
            all_eps += eps
            policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]),
                                     queries=[e for e in [eps[0].goal_category]] if eps else [], intrinsics=K,
                                     seed=int(cfg["run"]["seed"]))
            ex = Executor(env, cfg)
            for i, ep in enumerate(eps):
                policy.queries = [ep.goal_category]
                rec = VideoRecorder(out / f"video_{ep.episode_id}.mp4", cfg,
                                    f"DEBUG {emb['name']} / detector={cfg['detector']['type']}") \
                    if i < args.videos else None
                try:
                    res, trace = run_episode(env, policy, ex, ep, cfg, run_id=run_id,
                                             use_oracle_aux=cfg["detector"]["type"] == "oracle_semantic",
                                             on_frame=rec, logger=logger)
                finally:
                    if rec:
                        rec.close()
                rows.append(res.to_row())
                np.savez_compressed(out / f"map_{ep.episode_id}.npz", log_odds=policy.map.log_odds,
                                    observed=policy.map.observed, origin=np.array([policy.map.x0, policy.map.y0]),
                                    res=policy.map.res, pose=np.array(trace.poses[-1]))
                print(f"{ep.episode_id}: {res.outcome:13s} SR={int(res.success)} SPL={res.spl:.3f} "
                      f"l={res.start_geodesic:.2f} p={res.path_length:.2f} d_T={res.distance_to_goal:.2f} "
                      f"steps={res.num_decision_steps} coll={res.collisions} stuck={res.stuck_time_s:.1f}s "
                      f"wall={res.wall_time_s:.1f}s", flush=True)
                pd.DataFrame(rows, columns=column_names()).to_csv(out / "episodes.csv", index=False)
        finally:
            env.close()
    logger.close()
    save_episodes(out / "episodes.json", all_eps)
    df = pd.DataFrame(rows, columns=column_names())
    summ = summarize(df, ["success", "spl", "soft_spl", "dts", "collisions", "stuck_time_s", "path_length",
                          "num_decision_steps", "goal_ever_in_view", "goal_ever_detected"], n_boot=2000)
    lines = [f"# DEBUG run {run_id}", "", "role: debug — pipeline check, NOT a paper result.", "",
             summ.to_markdown(index=False), "", "Outcomes:", "", df["outcome"].value_counts().to_markdown()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(out / "manifest.json", build_manifest(run_id=run_id, config=cfg, seed=int(cfg["run"]["seed"]),
                                                        repo=REPO, extra={"role": "debug"}))
    print(f"\nsummary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
