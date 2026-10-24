"""Arms x seeds on a benchmark split (D-050): runs every cell with tools/run_eval.py, resumes unfinished cells,
starts and stops the model servers (one at a time on a 12 GB GPU), and writes the make_paper_tables spec.

    bash tools/wsl.sh python tools/run_matrix.py plan paper/matrices/hm3d_v1_main.yaml
    bash tools/wsl.sh python tools/run_matrix.py run  paper/matrices/hm3d_v1_main.yaml [--only qolda_kk,qwen_kk]
    bash tools/wsl.sh python tools/run_matrix.py spec paper/matrices/hm3d_v1_main.yaml   # -> paper/specs/<name>.yaml

Matrix file:
    name: hm3d_v1_main
    title: "HM3D ObjectNav v1 val (test subset), wheeled robot"
    out_root: runs/matrix                     # cells in <out_root>/<name>/<arm key>/seed<k>
    seeds: [0]
    common: [dataset=hm3d_objectnav_v1_val, embodiment=wheeled_vacuum, detector=grounding_dino_tiny, ...]
    servers:                                  # started before the first cell that needs one, stopped after the last
      qolda: {start: bash tools/start_qolda.sh, pid_file: runs/servers/qolda.pid,
              health: "http://127.0.0.1:23333/v1/models"}
    arms:
      - {key: vlfm_en, label: "VLFM-style, English category", lang: none, overrides: []}
      - {key: qolda_kk, label: "Qolda, kk", lang: kk, server: qolda, accent: true,
         overrides: [backend=qolda, policy.stop.calibration_file=calib/qolda_dev.json]}
      - {key: qolda_kk_noverify, label: "Qolda, kk, no verification", lang: kk, verify: false, server: qolda, ...}
    contrasts: [[qolda_kk, vlfm_en]]          # the analysis plan's primary contrasts (docs/analysis_plan.md)
    secondary_contrasts: [[qolda_kkru, qolda_kk]]   # exploratory, on the nested subset of the secondary arms
    secondary_subset_file: splits/hm3d_v1_test_secondary.txt   # arms with secondary: true override the subset to this
    decompositions: [[qolda_en, qolda_kk]]
    breakdowns: [goal_category]

A cell is complete when every selected episode is in episodes.csv, exclusions.csv, or failed
run.max_episode_attempts times; an unfinished cell is resumed (same configuration, D-041). Cells run in groups by
server, servers in the order of first use. A cell that exits with an error does not stop the matrix; ``plan``
shows it as unfinished. ``spec`` lists only complete cells unless --allow-partial (never for the paper).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.run.benchmark import read_failures  # noqa: E402


@dataclass
class Cell:
    arm: dict
    seed: int
    path: Path

    @property
    def key(self) -> str:
        return f"{self.arm['key']}/seed{self.seed}"


def _rel(path: Path) -> str:
    """Repository-relative path when inside the repository (portable specs), else absolute."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def load_matrix(path: Path) -> dict:
    m = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = [a["key"] for a in m["arms"]]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate arm keys")
    for a in m["arms"]:
        if a.get("server") and a["server"] not in (m.get("servers") or {}):
            raise SystemExit(f"arm {a['key']}: unknown server {a['server']!r}")
        if a.get("lang", "none") == "none" and a.get("server"):
            raise SystemExit(f"arm {a['key']}: lang none uses no model server")
    for pair in (m.get("contrasts") or []) + (m.get("decompositions") or []):
        for k in pair:
            if k not in keys:
                raise SystemExit(f"contrast names unknown arm {k!r}")
    return m


def cells(m: dict) -> List[Cell]:
    root = REPO / m.get("out_root", "runs/matrix") / m["name"]
    return [Cell(a, int(s), root / a["key"] / f"seed{int(s)}") for a in m["arms"] for s in m.get("seeds", [0])]


def cell_args(m: dict, c: Cell) -> List[str]:
    a = c.arm
    args = ["--lang", str(a.get("lang", "none"))]
    if a.get("lang", "none") != "none" and not a.get("verify", True):
        args.append("--no-verify")
    return args + [*m.get("common", []), *a.get("overrides", []), f"run.seed={c.seed}"]


def status(c: Cell) -> Dict[str, object]:
    """new | partial | complete, with counts from the cell's files."""
    man = c.path / "manifest.json"
    if not man.exists():
        return {"state": "new", "done": 0, "selected": None, "failed": 0}
    mf = json.loads(man.read_text(encoding="utf-8"))
    sel = int(mf.get("n_selected", 0))
    max_att = int(((mf.get("config") or {}).get("run") or {}).get("max_episode_attempts", 2))
    ids = set()
    for f in ("episodes.csv", "exclusions.csv"):
        if (c.path / f).exists():
            ids |= set(pd.read_csv(c.path / f, usecols=["episode_id"]).episode_id)
    fails = read_failures(c.path / "failures.csv")
    given_up = {e for e, n in fails.items() if n >= max_att and e not in ids}
    done = len(ids | given_up)
    return {"state": "complete" if sel and done >= sel else "partial", "done": done, "selected": sel,
            "failed": len(given_up)}


def plan(m: dict) -> pd.DataFrame:
    rows = []
    for c in cells(m):
        s = status(c)
        rows.append({"cell": c.key, "server": c.arm.get("server", "—"), "state": s["state"],
                     "done / selected": f"{s['done']}/{s['selected'] if s['selected'] is not None else '?'}",
                     "given up": s["failed"], "dir": _rel(c.path)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- servers
def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return 200 <= r.status < 300
    except Exception:          # noqa: BLE001 - any failure means "not up"
        return False


class Servers:
    def __init__(self, spec: Dict[str, dict], dry: bool):
        self.spec, self.dry = spec or {}, dry
        self.started: Optional[str] = None

    def ensure(self, name: Optional[str]) -> None:
        if name == self.started:
            return
        self.stop()
        if name is None:
            return
        sp = self.spec[name]
        others = [n for n, s in self.spec.items() if n != name and s.get("health") and _healthy(s["health"])]
        if others:
            raise SystemExit(f"server(s) {others} still running; stop them first (one model fits on the GPU)")
        if sp.get("health") and _healthy(sp["health"]):
            print(f"[matrix] server {name} already up (left running afterwards)", flush=True)
            self.started = None
            return
        print(f"[matrix] starting server {name}: {sp['start']}", flush=True)
        if not self.dry:
            subprocess.run(sp["start"], shell=True, check=True, cwd=REPO, executable="/bin/bash")
            if sp.get("health"):
                for _ in range(int(sp.get("wait_s", 600))):
                    if _healthy(sp["health"]):
                        break
                    time.sleep(1)
                else:
                    raise SystemExit(f"server {name} did not become healthy")
        self.started = name

    def stop(self) -> None:
        if self.started is None:
            return
        sp = self.spec[self.started]
        print(f"[matrix] stopping server {self.started}", flush=True)
        if not self.dry and sp.get("pid_file") and (REPO / sp["pid_file"]).exists():
            pid = int((REPO / sp["pid_file"]).read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(60):
                    os.kill(pid, 0)
                    time.sleep(1)
            except ProcessLookupError:
                pass
        self.started = None


def run(m: dict, only: Optional[List[str]], dry: bool) -> int:
    todo = [c for c in cells(m) if (not only or c.arm["key"] in only) and status(c)["state"] != "complete"]
    order = list(dict.fromkeys([None] + [c.arm.get("server") for c in todo]))      # no-server cells first
    servers = Servers(m.get("servers"), dry)
    failed = []
    try:
        for srv in order:
            group = [c for c in todo if c.arm.get("server") == srv]
            if not group:
                continue
            servers.ensure(srv)
            for c in group:
                resume = (c.path / "manifest.json").exists()
                cmd = [sys.executable, str(REPO / "tools" / "run_eval.py")]
                cmd += ["--resume", str(c.path)] if resume else ["--out", str(c.path), *cell_args(m, c)]
                print(f"[matrix] {c.key}: {'resume' if resume else 'start'}\n  {' '.join(shlex.quote(x) for x in cmd)}",
                      flush=True)
                if dry:
                    continue
                c.path.parent.mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(cmd, cwd=REPO).returncode
                if rc != 0:
                    failed.append(c.key)
                    print(f"[matrix] {c.key}: exited with {rc}; continuing", flush=True)
    finally:
        servers.stop()
    if failed:
        print(f"[matrix] unfinished cells: {failed}")
    return 1 if failed else 0


def write_spec(m: dict, out: Path, allow_partial: bool, part: str = "primary") -> dict:
    """``part``: primary (the arms without ``secondary: true``, contrasts) or secondary (every arm, restricted to the
    secondary arms' nested subset ``secondary_subset_file``, secondary contrasts)."""
    arms, missing = [], []
    chosen = [a for a in m["arms"] if part == "secondary" or not a.get("secondary")]
    for a in chosen:
        runs = []
        for c in [c for c in cells(m) if c.arm["key"] == a["key"]]:
            st = status(c)["state"]
            if st == "complete" or (allow_partial and st == "partial"):
                runs.append(_rel(c.path))
            else:
                missing.append(c.key)
        if runs:
            arms.append({"key": a["key"], "label": a["label"], "runs": runs, **({"accent": True} if a.get("accent") else {}),
                         **({"group": a["group"]} if a.get("group") else {})})
    if missing and not allow_partial:
        raise SystemExit(f"cells not complete: {missing} (run them, or --allow-partial for a preview)")
    have = {a["key"] for a in arms}
    ckey = "contrasts" if part == "primary" else "secondary_contrasts"
    spec = {"title": m.get("title", m["name"]) + (" [secondary arms]" if part == "secondary" else "")
                     + (" [PARTIAL]" if missing else ""), "arms": arms,
            "contrasts": [p for p in m.get(ckey) or [] if set(p) <= have],
            "decompositions": [p for p in m.get("decompositions") or [] if set(p) <= have],
            "breakdowns": m.get("breakdowns") or [], "sensitivity": part == "primary"}
    if part == "secondary":
        if not m.get("secondary_subset_file"):
            raise SystemExit("secondary arms need secondary_subset_file (the nested subset they ran on)")
        spec["subset_file"] = m["secondary_subset_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Written by tools/run_matrix.py from {m['name']}; regenerate instead of editing.\n"
                   + yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "spec"):
        p = sub.add_parser(name)
        p.add_argument("matrix")
        if name == "run":
            p.add_argument("--only", default="", help="comma-separated arm keys")
            p.add_argument("--dry-run", action="store_true")
        if name == "spec":
            p.add_argument("--out", default="")
            p.add_argument("--allow-partial", action="store_true")
            p.add_argument("--part", choices=["primary", "secondary"], default="primary")
    args = ap.parse_args()
    m = load_matrix(Path(args.matrix))
    if args.cmd == "plan":
        print(plan(m).to_markdown(index=False))
    elif args.cmd == "run":
        raise SystemExit(run(m, [k for k in args.only.split(",") if k], args.dry_run))
    else:
        suffix = "" if args.part == "primary" else "_secondary"
        out = Path(args.out) if args.out else REPO / "paper" / "specs" / f"{m['name']}{suffix}.yaml"
        spec = write_spec(m, out, args.allow_partial, args.part)
        print(f"{len(spec['arms'])} arms -> {out}")


if __name__ == "__main__":
    main()
