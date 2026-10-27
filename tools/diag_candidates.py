"""Why does the robot see the goal but not reach it? (D-065, dev only.) Re-runs chosen episodes of a dev matrix cell
with the cell's configuration and logs every object candidate at every step; ``report`` then explains each episode.

    bash tools/wsl.sh python tools/diag_candidates.py run --cell runs/matrix/dev_calibration/qolda_kk/seed0 \
        --stage not_reached --limit 20 --out runs/diag/candidates/qolda_kk [extra hydra overrides ...]
    bash tools/wsl.sh python tools/diag_candidates.py report runs/diag/candidates/qolda_kk

run: the cell's manifest gives the overrides and the language; extra overrides are appended (e.g. a calibration
file). Episodes: the cell's episodes with the given failure stage(s), in a seeded hash order (not the first scenes),
or --episodes. Output: a normal run directory (episodes.csv, events.jsonl, ...) plus cands_<episode>.jsonl:
a header line, then one line per step with the decision and every candidate:
    [cid, fused conf, best detector score, detections, misses, flags, distance to the goal's AABB, distance to the
     robot, goal fraction of its box in this frame or null]     flags: 1 reliable, 2 rejected, 4 lost, 8 unreachable
Benchmark-role datasets (val) are refused: the test set stays untouched.

report: per episode, the goal candidates (within --goal-radius of a goal AABB, or a box that is mostly goal
pixels) and a reason why the episode failed; then fused confidences of goal and non-goal candidates, the numbers
behind a reliable-confidence threshold for verifier arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.sim.episodes import footprint_distance  # noqa: E402

REASONS = {
    "no_goal_candidate": "goal detected, but no candidate was ever placed on it",
    "never_reliable": "goal candidate below the reliable confidence, never approached; exploration never exhausted",
    "suspected_too_late": "goal candidate only suspected, never approached; exploration exhausted too late",
    "reliable_not_chosen": "goal candidate reliable, never approached (other candidate or no path)",
    "goal_rejected": "goal candidate rejected by the verifier",
    "became_suspected": "approached, then its fused confidence fell below the bar: suspected, the robot left",
    "approached_lost": "approached, then given up as lost (not seen again from close by)",
    "switched": "approached, then the robot went elsewhere while the goal candidate was still reliable",
    "approach_cut_by_budget": "still approaching the goal candidate when the episode ended",
    "other": "other",
}


class CandidateLog:
    """Frame hook (the video interface of BenchmarkRunner): one JSON line per step."""

    def __init__(self, path: Path, ep: Any, policy_cfg: Dict[str, Any], overlap: float, video: Any = None):
        self.f = open(path, "w", encoding="utf-8")
        self.goal_ids = np.asarray(ep.goal_semantic_ids())
        self.overlap = overlap
        self.video = video
        self.header = {"episode_id": ep.episode_id, "category": ep.goal_category,
                       "goals": [[list(map(float, g.aabb_min)), list(map(float, g.aabb_max))] for g in ep.goals],
                       "stop_distance_m": float(policy_cfg["approach"]["stop_distance_m"])}
        self.wrote_header = False

    def _fraction(self, gp: np.ndarray, box) -> Optional[float]:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        b = gp[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        return float(b.mean()) if b.size else None

    def __call__(self, *, obs, decision, policy, trace, step, episode):
        """Log one decision step: goal visibility, detections, every candidate's state and the verification."""
        if self.video is not None:
            self.video(obs=obs, decision=decision, policy=policy, trace=trace, step=step, episode=episode)
        if not self.wrote_header:
            self.header.update(queries=list(policy.queries), reliable_conf=float(policy.memory.p.reliable_conf))
            self.f.write(json.dumps(self.header, ensure_ascii=False) + "\n")
            self.wrote_header = True
        sem = obs.semantic
        gp = np.isin(sem, self.goal_ids) if sem is not None and sem.size > 1 else np.zeros((1, 1), bool)
        frame = policy.step_idx - 1
        goal_det = any((self._fraction(gp, d.box_xyxy) or 0.0) >= self.overlap for d in policy.last_detections)
        mem = policy.memory
        cands = []
        for c in mem.candidates:
            flags = (1 if mem.is_reliable(c) else 0) | (2 if c.rejected else 0) | (4 if c.lost else 0) \
                | (8 if c.unreachable else 0)
            dg = min(footprint_distance(c.position, g.aabb_min, g.aabb_max) for g in episode.goals)
            fr = self._fraction(gp, c.last_box) if c.last_step == frame and c.last_box is not None else None
            cands.append([c.cid, round(float(c.conf), 3), round(float(c.best_score), 3), int(c.n_detections),
                          int(c.misses), flags, round(float(dg), 2), round(c.nearest_distance(obs.pose[:2]), 2),
                          None if fr is None else round(fr, 2)])
        v = policy.last_verify
        rec = {"step": step, "kind": decision.kind, "cand": decision.candidate_id, "exhausted": bool(policy.exhausted),
               "goal_in_view": int(gp.sum()), "goal_det": bool(goal_det), "n_det": len(policy.last_detections),
               "verify": None if v is None or v.step != frame else [v.cid, v.decision, round(float(v.p_candidate), 3)],
               "cands": cands}
        self.f.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        """Close the log file and the optional video."""
        self.f.close()
        if self.video is not None:
            self.video.close()


# ------------------------------------------------------------------------------------------------------------ run
def _hash_order(ids: List[str], seed: int = 0) -> List[str]:
    return sorted(ids, key=lambda x: hashlib.sha256(f"{seed}|diag|{x}".encode()).hexdigest())


def cmd_run(args: argparse.Namespace) -> None:
    """Re-run selected episodes of a dev matrix cell with the per-step candidate log."""
    from kznav.language.lexicon import Lexicon
    from kznav.report.episode_video import EpisodeVideo
    from kznav.run.benchmark import BenchmarkRunner, LanguageRun, word_types_from
    from kznav.run.config import compose_config
    from kznav.run.logging import RunLogger
    from kznav.run.manifest import build_manifest, write_manifest
    from kznav.sim.objectnav_dataset import load_objectnav_v1, scenes_of_ids, select_episodes

    cell = Path(args.cell)
    man = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
    if args.episodes:
        ids = args.episodes.split(",")
    else:
        df = pd.read_csv(cell / "episodes.csv")
        ids = _hash_order(df.loc[df.failure_stage.isin(args.stage.split(",")), "episode_id"].tolist())[:args.limit]
    if not ids:
        raise SystemExit("no episodes selected")
    out = Path(args.out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a diagnostic run")
    out.mkdir(parents=True, exist_ok=True)
    subset = out / "episodes.txt"
    subset.write_text(f"# diag_candidates: {len(ids)} episodes of {cell.as_posix()} ({args.stage})\n"
                      + "\n".join(ids) + "\n", encoding="utf-8")
    overrides = [o for o in man["overrides"] if not o.startswith("run.episodes.subset_file=")] \
        + [f"run.episodes.subset_file={subset.as_posix()}", *args.overrides]
    cfg = compose_config(overrides)
    ds = cfg["dataset"]
    if ds["role"] != "dev":
        raise SystemExit(f"diagnostics run on dev episodes only, not {ds['name']} ({ds['role']})")
    lang, verify = man["lang"], bool(man["verify"])
    eps = load_objectnav_v1(Path(ds["episodes_path"]), Path(ds["scene_root"]), dataset=ds["name"], split=ds["split"],
                            scenes=scenes_of_ids(ids))
    eps = select_episodes(eps, subset=ids)
    logger = RunLogger(out)
    run_id = f"diag_candidates_{out.name}"
    language = None
    extra = {"role": "diag", "overrides": overrides, "lang": lang, "verify": verify, "source_cell": cell.as_posix(),
             "n_selected": len(eps)}
    if lang != "none":
        language = LanguageRun(cfg, lang, verify=verify, call_sink=logger.call_sink, repo=REPO,
                               eval_episode_ids=[e.episode_id for e in eps])
        language.backend.context.run_id = run_id
        language.warmup()
        extra.update(language.manifest_extra())
    write_manifest(out / "manifest.json", build_manifest(run_id=run_id, config=cfg, seed=int(cfg["run"]["seed"]),
                                                        repo=REPO, extra=extra))
    cfg["run"]["video"]["fraction"] = 1.0           # the frame hook runs on every episode; videos only with --video
    overlap = float(cfg["eval"]["funnel"]["detection_goal_overlap"])
    label = f"DIAG {ds['name']} / {cfg['embodiment']['name']} / {cfg['detector']['type']}"

    def hook(path: Path, ep: Any, lg: Any) -> CandidateLog:
        video = EpisodeVideo(path, cfg, label, language=lg) if args.video else None
        return CandidateLog(out / f"cands_{ep.episode_id.replace(':', '_')}.jsonl", ep, cfg["policy"], overlap, video)

    def env_factory(scene_path: str):
        from kznav.sim.env import HabitatEnv, SceneSpec

        return HabitatEnv(cfg, SceneSpec(scene_path, ds.get("scene_dataset_config")), out / "work" / Path(scene_path).stem,
                          seed=int(cfg["run"]["seed"]))

    runner = BenchmarkRunner(cfg, out, run_id=run_id, env_factory=env_factory, logger=logger, language=language,
                             video_factory=hook,
                             cache_root=REPO / "runs" / "cache" / "view_points" / ds["name"] / cfg["embodiment"]["name"],
                             word_types=word_types_from(Lexicon.load(REPO / ds["lexicon"])),
                             progress=lambda s: print(s, flush=True))
    print(runner.run(eps))
    logger.close()


# --------------------------------------------------------------------------------------------------------- report
def explain(lines: List[Dict[str, Any]], header: Dict[str, Any], goal_radius: float, overlap: float) -> Dict[str, Any]:
    """One episode's goal candidates and the reason it failed (``REASONS``): what became of the goal candidate the
    robot approached last (D-065), or why it never approached one."""
    goal: Dict[int, Dict[str, Any]] = {}
    other_conf: Dict[int, float] = {}
    last_state: Dict[int, list] = {}
    goal_det_steps = 0
    for r in lines:
        goal_det_steps += r["goal_det"]
        for c in r["cands"]:
            cid, conf, flags, dg, fr = c[0], c[1], c[5], c[6], c[8]
            last_state[cid] = c
            if dg <= goal_radius or (fr is not None and fr >= overlap):
                g = goal.setdefault(cid, {"first": r["step"], "max_conf": 0.0, "reliable": False})
                g.update(max_conf=max(g["max_conf"], conf), reliable=g["reliable"] or bool(flags & 1))
    for cid, c in last_state.items():
        if cid not in goal:
            other_conf[cid] = c[1]
    exhausted = any(r["exhausted"] for r in lines)
    appr = [k for k, r in enumerate(lines) if r["cand"] in goal and r["kind"] in ("approach", "face")]
    if not goal:
        reason = "no_goal_candidate" if goal_det_steps else "other"
    elif any(last_state[cid][5] & 2 for cid in goal):
        reason = "goal_rejected"
    elif not appr:
        reason = "reliable_not_chosen" if any(g["reliable"] for g in goal.values()) else (
            "suspected_too_late" if exhausted else "never_reliable")
    elif appr[-1] == len(lines) - 1:
        reason = "approach_cut_by_budget"
    else:
        cid = lines[appr[-1]]["cand"]
        after = next(c for c in lines[appr[-1] + 1]["cands"] if c[0] == cid)
        reason = "approached_lost" if after[5] & 4 else ("became_suspected" if not after[5] & 1 else "switched")
    return {"episode_id": header["episode_id"], "category": header["category"], "steps": len(lines),
            "goal_det_steps": goal_det_steps, "goal_cands": len(goal),
            "goal_max_conf": max((g["max_conf"] for g in goal.values()), default=np.nan),
            "goal_final_conf": max((last_state[cid][1] for cid in goal), default=np.nan),
            "first_goal_step": min((g["first"] for g in goal.values()), default=np.nan),
            "exhausted": exhausted, "reason": reason, "_goal_conf": [last_state[cid][1] for cid in goal],
            "_other_conf": list(other_conf.values())}


def cmd_report(args: argparse.Namespace) -> None:
    """Explain each logged episode's failure and write ``report.md`` with confidence thresholds."""
    d = Path(args.dir)
    eps = pd.read_csv(d / "episodes.csv").set_index("episode_id") if (d / "episodes.csv").exists() else None
    rows, thr = [], float("nan")
    for f in sorted(d.glob("cands_*.jsonl")):
        raw = [json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not raw:
            continue
        thr = float(raw[0]["reliable_conf"])
        rows.append(explain(raw[1:], raw[0], args.goal_radius, args.overlap))
    if not rows:
        raise SystemExit(f"no cands_*.jsonl in {d}")
    rep = pd.DataFrame(rows)
    if eps is not None:
        rep["stage"] = rep.episode_id.map(eps["failure_stage"])
        rep["success"] = rep.episode_id.map(eps["success"])
    lines = [f"# Candidates in {d.as_posix()} (goal radius {args.goal_radius} m, reliable conf {thr})", "",
             rep.drop(columns=["_goal_conf", "_other_conf"]).round(3).to_markdown(index=False), "", "Reasons:", ""]
    lines += [f"- {k}: {n} - {REASONS[k]}" for k, n in rep.reason.value_counts().items()]
    gc = np.array([c for cs in rep._goal_conf for c in cs])
    oc = np.array([c for cs in rep._other_conf for c in cs])
    lines += ["", "Final fused confidence per candidate (goal vs non-goal), share at or above each threshold:", "",
              "| threshold | goal candidates | non-goal candidates |", "|---|---|---|"]
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
        lines.append(f"| {t:.2f} | {np.mean(gc >= t) if len(gc) else float('nan'):.2f} (n={len(gc)}) | "
                     f"{np.mean(oc >= t) if len(oc) else float('nan'):.2f} (n={len(oc)}) |")
    text = "\n".join(lines) + "\n"
    (d / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    """Command-line entry point: ``run`` or ``report``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cell", required=True, help="a dev matrix cell (its manifest gives the configuration)")
    r.add_argument("--stage", default="not_reached", help="failure stage(s) to re-run, comma-separated")
    r.add_argument("--episodes", default="", help="explicit episode ids, comma-separated (instead of --stage)")
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--out", required=True)
    r.add_argument("--video", action="store_true", help="also write the debug video of every episode")
    r.add_argument("overrides", nargs="*")
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.add_argument("--goal-radius", type=float, default=0.3, help="a candidate this close to a goal AABB is the goal")
    p.add_argument("--overlap", type=float, default=0.3, help="... or one whose box is this much goal pixels")
    args = ap.parse_args()                  # hydra overrides go last (subparsers rule out intermixed parsing)
    cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
