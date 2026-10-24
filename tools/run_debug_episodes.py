"""Run ObjectNav episodes on a DEBUG dataset (test scenes + placed objects with known ids).

    bash tools/wsl.sh python tools/run_debug_episodes.py [--videos 2] [--lang kk] [--no-verify] [hydra overrides ...]
    e.g.  ... detector=mm_grounding_dino_base value_encoder=siglip2_base --lang kk   (needs the Qolda server)

--lang none (default): the detector gets the English category name, no language model.
--lang kk | kk_ru | en: the command (commands/<dataset>.yaml templates, round-robin, or --command "...") is parsed by the
    configured VLM backend; its hypotheses are the detector queries and value-map weights, and the
    VLM verifies each candidate before the stop (unless --no-verify).
    goal=top1 | mt_word | mt_command (with --lang): the goal ablations of D-038 / D-039.
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
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.aggregate import summarize  # noqa: E402
from kznav.eval.records import column_names  # noqa: E402
from kznav.language.base import ListSink, TeeSink  # noqa: E402
from kznav.language.goal import make_goal_cache, with_target_name  # noqa: E402
from kznav.language.lexicon import Lexicon  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402
from kznav.perception.base import create_detector, create_segmenter, create_value_encoder  # noqa: E402
from kznav.perception.camera import Intrinsics  # noqa: E402
from kznav.policy.objectnav import ObjectNavPolicy  # noqa: E402
from kznav.policy.verification import load_stop_calibration, make_verifier  # noqa: E402
from kznav.report.episode_video import EpisodeVideo as VideoRecorder  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.run.episode import Executor, run_episode  # noqa: E402
from kznav.run.gpu import memory_report  # noqa: E402
from kznav.run.logging import RunLogger  # noqa: E402
from kznav.run.manifest import build_manifest, write_manifest  # noqa: E402
from kznav.sim.env import HabitatEnv, SceneSpec  # noqa: E402
from kznav.sim.episodes import ObjectSpec, compute_view_points, placement_populate, sample_episodes, save_episodes  # noqa: E402

# Language of the object name inside the command templates (commands/<dataset>.yaml: kk_ru names it in Russian).
# Only goal=mt_word uses the name: it gets it the way standard ObjectNav gets the category (D-039).
from kznav.language.commands import WORD_LANG  # noqa: E402


def command_for(templates: dict, lexicon: Lexicon, lang: str, category: str, i: int):
    t = templates[lang][i % len(templates[lang])]
    names = lexicon.categories[category].names
    text = t["text"].format(**{k: (names.get(k) or [category])[0] for k in ("kk", "ru", "en")})
    return t["id"], text[0].upper() + text[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, default=2, help="record the first N episodes")
    ap.add_argument("--lang", default="none", choices=["none", "kk", "kk_ru", "en"])
    ap.add_argument("--no-verify", action="store_true", help="with --lang: stop without VLM verification")
    ap.add_argument("--only", default="", help="comma-separated episode ids to run (after sampling)")
    ap.add_argument("--command", default="", help="with --lang: your own command text for every episode run "
                    "(pick the episode whose object it names with --only)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    use_lang = args.lang != "none"
    cfg = compose_config(["dataset=debug_ycb", "detector=oracle_semantic",
                          *(["backend=qolda"] if use_lang else []), *args.overrides])
    ds, emb = cfg["dataset"], cfg["embodiment"]
    if ds["role"] != "debug":
        raise SystemExit("this tool is for debug datasets only")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    venc_name = cfg["value_encoder"]["type"] if cfg["value_encoder"].get("enabled", False) else "novalue"
    goal_name = str(cfg["goal"]["name"])
    if goal_name != "multi" and not use_lang:
        raise SystemExit(f"goal={goal_name} needs --lang kk | kk_ru | en")
    goal_tag = "" if goal_name == "multi" else f"-{goal_name}"
    lang_tag = ("nolang" if not use_lang else
                f"{cfg['backend']['name']}-{args.lang}{goal_tag}-{'noverify' if args.no_verify else 'verify'}")
    run_id = f"debug_{ds['name']}_{emb['name']}_{cfg['detector']['type']}_{venc_name}_{lang_tag}_{stamp}"
    out = REPO / "runs" / "debug" / stamp
    logger = RunLogger(out)
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    rgbd = emb["sensors"]["rgbd"]
    K = Intrinsics.from_hfov(float(rgbd["hfov_deg"]), int(rgbd["resolution_hw"][1]), int(rgbd["resolution_hw"][0]))
    lexicon = Lexicon.load(REPO / ds["lexicon"])
    backend = goals = templates = None
    calls = ListSink()
    if args.command and not use_lang:
        raise SystemExit("--command needs --lang kk | kk_ru | en")
    extra = {"role": "debug", "lang": args.lang, "verify": use_lang and not args.no_verify,
             "custom_command": args.command or None}
    if use_lang:
        backend = create_backend(cfg["backend"], sink=TeeSink(logger.call_sink, calls))
        backend.context.run_id = run_id
        backend.warmup(int(cfg["backend"].get("warmup_calls", 0)))
        goals = make_goal_cache(cfg["goal"], backend, calls, lexicon, translator_sink=logger.call_sink)
        cmd_file = yaml.safe_load((REPO / "commands" / f"{ds['name']}.yaml").read_text(encoding="utf-8"))
        templates = cmd_file["templates"]
        _, thr, base, calibrated = load_stop_calibration(cfg["policy"])
        extra.update(backend=backend.describe_config(), goal=cfg["goal"],
                     translator=goals.translator.describe_config() if goals.translator is not None else None,
                     commands_review_status=cmd_file["review_status"],
                     stop_threshold=thr, base_rate=base, calibrated=calibrated,
                     note="" if calibrated else "verifier uncalibrated (identity + debug threshold): debug only")
    only = {x for x in args.only.split(",") if x}
    rows, all_eps, lang_rows = [], [], []
    n_video = 0
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
            venc = create_value_encoder(cfg["value_encoder"]) if cfg["value_encoder"].get("enabled", False) else None
            policy = ObjectNavPolicy(cfg, create_detector(cfg["detector"]), create_segmenter(cfg["segmenter"]),
                                     queries=[], intrinsics=K, seed=int(cfg["run"]["seed"]), value_encoder=venc)
            ex = Executor(env, cfg)
            per_cat = {}
            for ep in eps:
                i_cat = per_cat.get(ep.goal_category, 0)
                per_cat[ep.goal_category] = i_cat + 1
                if only and ep.episode_id not in only:
                    continue
                lg = None
                if use_lang:
                    cid, text = ("custom", args.command) if args.command else                         command_for(templates, lexicon, args.lang, ep.goal_category, i_cat)
                    word_lang = WORD_LANG[args.lang]
                    names = lexicon.categories[ep.goal_category].names
                    lg = goals.get(command_id=cid, text=text, lang=args.lang, category=ep.goal_category,
                                   word_type=lexicon.categories[ep.goal_category].word_type,
                                   word=(names.get(word_lang) or [ep.goal_category])[0], word_lang=word_lang)
                    lang_rows.append({"episode_id": ep.episode_id, **lg.summary()})
                    logger.event("goal", episode_id=ep.episode_id, step=-1, **lg.summary())
                    texts, weights = list(lg.hypotheses), list(lg.weights)
                    if cfg["perception"]["detection"].get("add_target_name", False):      # D-072
                        texts, weights = with_target_name(texts, weights, lg.target_en)
                    policy.queries = texts
                    policy.query_weights = weights if texts else None
                    policy.rooms = list(lg.rooms)
                    backend.context.episode_id = ep.episode_id
                    policy.verifier = None if args.no_verify or not lg.parse_ok else make_verifier(
                        cfg, backend, lg.target, lg.effective_lang, texts, weights, lg.target_en)
                else:
                    policy.queries, policy.query_weights, policy.verifier = [ep.goal_category], None, None
                rec = None
                if n_video < args.videos:
                    n_video += 1
                    rec = VideoRecorder(out / f"video_{ep.episode_id}.mp4", cfg,
                                        f"DEBUG {emb['name']} / {cfg['detector']['type']} / {venc_name} / {lang_tag.split('_')[0]}",
                                        language=lg)
                try:
                    res, trace = run_episode(env, policy, ex, ep, cfg, run_id=run_id, cell_id=lang_tag,
                                             backend=cfg["backend"]["name"] if use_lang else "none",
                                             model_id=cfg["backend"]["model_id"] if use_lang else "none",
                                             command_lang=args.lang if use_lang else "en",
                                             command_text=lg.text if lg else "", command_id=lg.command_id if lg else "",
                                             word_type=lg.word_type if lg else "universal",
                                             use_oracle_aux=cfg["detector"]["type"] == "oracle_semantic",
                                             on_frame=rec, logger=logger, language=lg, calls=calls)
                finally:
                    if rec:
                        rec.close()
                rows.append(res.to_row())
                np.savez_compressed(out / f"map_{ep.episode_id}.npz", log_odds=policy.map.log_odds,
                                    observed=policy.map.observed, origin=np.array([policy.map.x0, policy.map.y0]),
                                    res=policy.map.res, pose=np.array(trace.poses[-1]))
                vtxt = ""
                if policy.verifier is not None:
                    evs = policy.verifier.events
                    vtxt = f" verif={sum(e.called for e in evs)} rej={sum(e.decision == 'reject' for e in evs)}"
                print(f"{ep.episode_id}: {res.outcome:13s} SR={int(res.success)} SPL={res.spl:.3f} "
                      f"l={res.start_geodesic:.2f} p={res.path_length:.2f} d_T={res.distance_to_goal:.2f} "
                      f"steps={res.num_decision_steps} coll={res.collisions} stuck={res.stuck_time_s:.1f}s "
                      f"vlm={res.vlm_calls}/{res.vlm_latency_total_s:.1f}s{vtxt} wall={res.wall_time_s:.1f}s", flush=True)
                pd.DataFrame(rows, columns=column_names()).to_csv(out / "episodes.csv", index=False)
        finally:
            env.close()
    logger.close()
    save_episodes(out / "episodes.json", all_eps)
    if lang_rows:
        pd.DataFrame(lang_rows).to_csv(out / "goals.csv", index=False)
    df = pd.DataFrame(rows, columns=column_names())
    metrics = ["success", "spl", "soft_spl", "dts", "collisions", "stuck_time_s", "path_length", "num_decision_steps",
               "goal_ever_in_view", "goal_ever_detected"]
    if use_lang:
        metrics += ["goal_ever_verified_positive", "parse_ok", "vlm_calls", "vlm_calls_verify", "vlm_latency_total_s"]
    summ = summarize(df, metrics, n_boot=2000)
    lines = [f"# DEBUG run {run_id}", "", "role: debug — pipeline check, NOT a paper result.", ""]
    if use_lang:
        extra["gpu_memory"] = mem = memory_report(cfg["backend"]["base_url"])
        if mem["server_peak_exceeds_gpu"]:
            lines += [f"WARNING: model server peak {mem['server_stats']['max_allocated_gb']:.2f} GB > GPU "
                      f"{mem['gpu_total_gb']:.2f} GB: paged to shared memory, latencies invalid (D-006).", ""]
    if use_lang and not extra["calibrated"]:
        lines += [f"verifier: UNCALIBRATED (identity, stop threshold {extra['stop_threshold']}); "
                  f"commands: {extra['commands_review_status']}", ""]
    lines += [summ.to_markdown(index=False), "", "Outcomes:", "", df["outcome"].value_counts().to_markdown()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(out / "manifest.json", build_manifest(run_id=run_id, config=cfg, seed=int(cfg["run"]["seed"]),
                                                        repo=REPO, extra=extra))
    print(f"\nsummary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
