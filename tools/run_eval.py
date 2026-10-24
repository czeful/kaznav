"""Run ObjectNav on a benchmark split (HM3D ObjectNav v1 val, HM3D-OVON) with any embodiment, model and goal mode.

    bash tools/wsl.sh python tools/run_eval.py [--lang none|kk|kk_ru|en] [--no-verify] [--shard i/n]
        [--resume runs/eval/<dir>] [hydra overrides ...]
    e.g.  ... --lang kk embodiment=wheeled_vacuum backend=qolda detector=grounding_dino_tiny value_encoder=siglip2_base
          ... run.episodes.subset_file=splits/hm3d_val_500.txt          (a fixed subset)
          ... policy.stop.calibration_file=calib/qolda_dev.json          (required for reported runs)

--lang none: the detector gets the English category name (the VLFM-style baseline, no language model).
The dataset's episodes are prepared for the embodiment (kznav/run/benchmark.py, D-041): standard agent = the
dataset's navmesh and view points (table (a)); wheeled robot = its own navmesh, recomputed view points (cached in
runs/cache/view_points), starts snapped to its navmesh, unreachable episodes listed in exclusions.csv (table (b)).
Output: runs/eval/<UTC timestamp>/{episodes.csv, exclusions.csv, failures.csv, goals.csv, events.jsonl,
        vlm_calls.jsonl, summary.md, manifest.json, video_*.mp4}. --resume continues a run directory with the same
        configuration and retries episodes that failed (D-046). --out names a new run directory (tools/run_matrix.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.aggregate import summarize  # noqa: E402
from kznav.language.lexicon import Lexicon  # noqa: E402
from kznav.report.episode_video import EpisodeVideo  # noqa: E402
from kznav.run.benchmark import BenchmarkRunner, LanguageRun, RunAborted, word_types_from  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from kznav.run.logging import RunLogger  # noqa: E402
from kznav.run.manifest import build_manifest, config_hash, write_manifest  # noqa: E402
from kznav.sim.objectnav_dataset import load_objectnav_v1, read_id_list, scenes_of_ids, select_episodes  # noqa: E402

METRICS = ["success", "spl", "soft_spl", "dts", "collisions", "stuck_time_s", "path_length", "num_decision_steps",
           "goal_ever_in_view", "goal_ever_detected"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="none", choices=["none", "kk", "kk_ru", "en"])
    ap.add_argument("--no-verify", action="store_true", help="with --lang: stop without VLM verification")
    ap.add_argument("--shard", default="", help="i/n: run every n-th episode starting at i")
    ap.add_argument("--resume", default="", help="continue this run directory (same configuration)")
    ap.add_argument("--out", default="", help="run directory for a new run (default runs/eval/<UTC timestamp>)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args()      # hydra overrides may follow the options
    use_lang = args.lang != "none"
    if args.resume and args.out:
        raise SystemExit("--resume names the run directory; --out is for a new run")
    out = Path(args.resume or args.out or REPO / "runs" / "eval" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if args.out and (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a run; continue it with --resume {out}")
    overrides = list(args.overrides)
    prev = None
    if args.resume:      # a resumed run keeps the configuration it started with (manifest extra fields)
        prev = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if overrides and overrides != prev["overrides"]:
            raise SystemExit(f"--resume keeps the run's overrides {prev['overrides']}; got {overrides}")
        overrides, args.lang, args.shard = prev["overrides"], prev["lang"], prev["shard"]
        args.no_verify = prev["lang"] != "none" and not prev["verify"]
        use_lang = args.lang != "none"
    cfg = compose_config(overrides)
    ds = cfg["dataset"]
    if ds["role"] not in ("benchmark", "dev", "smoke"):
        raise SystemExit(f"dataset {ds['name']} has role {ds['role']!r}; debug sets run with tools/run_debug_episodes.py")
    if cfg["goal"]["name"] != "multi" and not use_lang:
        raise SystemExit(f"goal={cfg['goal']['name']} needs --lang kk | kk_ru | en")
    sel = cfg["run"]["episodes"]
    subset = read_id_list(Path(sel["subset_file"])) if sel.get("subset_file") else None
    eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"],
                            scenes=scenes_of_ids(subset) if subset else None)    # only the subset's scenes
    eps = select_episodes(eps, subset=subset, limit=sel.get("limit"), shard=args.shard or None)
    out.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out)
    run_id = f"eval_{ds['name']}_{cfg['embodiment']['name']}_{out.name}"
    language = None
    extra = {"role": ds["role"], "overrides": overrides, "lang": args.lang, "verify": use_lang and not args.no_verify,
             "shard": args.shard, "n_selected": len(eps), "resumes": []}
    if prev is not None:
        if prev["config_sha256"] != config_hash(cfg):
            raise SystemExit("the composed configuration differs from the run's manifest (code or config changed); "
                             "not resuming")
        extra["resumes"] = prev.get("resumes", []) + [datetime.now(timezone.utc).isoformat(timespec="seconds")]
        extra["first_created_utc"] = prev.get("first_created_utc", prev["created_utc"])
    if use_lang:
        language = LanguageRun(cfg, args.lang, verify=not args.no_verify, call_sink=logger.call_sink, repo=REPO,
                               eval_episode_ids=[e.episode_id for e in eps])
        language.backend.context.run_id = run_id
        language.warmup()
        extra.update(language.manifest_extra())
    write_manifest(out / "manifest.json", build_manifest(run_id=run_id, config=cfg, seed=int(cfg["run"]["seed"]),
                                                        repo=REPO, extra=extra))

    def env_factory(scene_path: str):
        from kznav.sim.env import HabitatEnv, SceneSpec

        return HabitatEnv(cfg, SceneSpec(scene_path, ds.get("scene_dataset_config")), out / "work" / Path(scene_path).stem,
                          seed=int(cfg["run"]["seed"]))

    lexicon = Lexicon.load(REPO / ds["lexicon"])
    label = f"{ds['name']} / {cfg['embodiment']['name']} / {cfg['detector']['type']}"
    runner = BenchmarkRunner(
        cfg, out, run_id=run_id, env_factory=env_factory, logger=logger, language=language,
        video_factory=lambda path, ep, lg: EpisodeVideo(path, cfg, label, language=lg),
        cache_root=REPO / "runs" / "cache" / "view_points" / ds["name"] / cfg["embodiment"]["name"],
        word_types=word_types_from(lexicon), progress=lambda s: print(s, flush=True))
    aborted = ""
    try:
        counts = runner.run(eps)
    except RunAborted as e:
        aborted, counts = str(e), dict(runner.counts)
    logger.close()
    df = pd.read_csv(runner.episodes_csv) if runner.episodes_csv.exists() else pd.DataFrame()
    exc = pd.read_csv(runner.exclusions_csv) if runner.exclusions_csv.exists() else pd.DataFrame(columns=["reason"])
    fail = pd.read_csv(runner.failures_csv) if runner.failures_csv.exists() else pd.DataFrame(columns=["episode_id"])
    open_fail = fail[~fail.episode_id.isin(df.episode_id if len(df) else [])].drop_duplicates("episode_id", keep="last")
    lines = [f"# Run {run_id}", "", f"dataset {ds['name']} ({ds['split']}), embodiment {cfg['embodiment']['name']}, "
             f"lang {args.lang}, goal {cfg['goal']['name']}, verify {extra['verify']}", "",
             f"selected {len(eps)}; run {len(df)}; excluded {len(exc)}; failed (not completed) {len(open_fail)}; "
             f"this invocation: {counts}", ""]
    if aborted:
        lines += [f"ABORTED: {aborted}", ""]
    if len(open_fail):
        lines += ["Episodes that did not complete (failures.csv; resume retries them up to run.max_episode_attempts):",
                  "", open_fail[["episode_id", "attempt", "kind", "error_type", "message"]].to_markdown(index=False), ""]
    if ds["role"] == "smoke":
        lines += ["SMOKE TEST on val scenes: checks the chain, never a result, never used to tune anything.", ""]
    if language is not None and not language.calibrated and extra["verify"]:
        lines += [f"verifier: UNCALIBRATED (identity, stop threshold {language.threshold}) - not a reported run", ""]
    if len(exc):
        lines += ["Exclusions by reason:", "", exc["reason"].value_counts().to_markdown(), ""]
        if cfg["embodiment"]["navmesh"]["source"] == "dataset":
            lines += ["WARNING: the standard agent should run every dataset episode; exclusions break the comparison "
                      "with the literature (table (a)).", ""]
    if len(df):
        metrics = METRICS + (["goal_ever_verified_positive", "parse_ok", "vlm_calls", "vlm_latency_total_s"] if use_lang else [])
        lines += [summarize(df, metrics, n_boot=2000).to_markdown(index=False), "", "Outcomes:", "",
                  df["outcome"].value_counts().to_markdown()]
        if "failure_stage" in df:
            lines += ["", "Failure stages (D-045):", "", df["failure_stage"].value_counts().to_markdown()]
    if extra["resumes"]:
        lines += ["", f"Resumed {len(extra['resumes'])} time(s): resumed episodes ran in a new simulator process (D-036)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary: {out / 'summary.md'}")
    if aborted:
        raise SystemExit(f"run aborted: {aborted}")


if __name__ == "__main__":
    main()
