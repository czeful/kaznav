"""Compare two runs of the reproducibility check (docker/check/repro.yaml) call by call (D-089).

    python docker/check/compare.py runs/docker_check/host runs/docker_check/docker [--reference]

Per arm: episodes whose metrics differ (success, SPL, path length, steps, termination, ...) and VLM calls whose
input (``input_key``: the rendered image's hash and the prompt) or output (answer, P(yes)) differ, with the first
difference. ``--reference`` instead compares run A with each arm's original dev run (``reference`` in the matrix) on
the same episodes. Also compares the test frames of tools/check_render.py. Writes <B>/compare.md (or
<A>/compare_reference.md); exit status 1 if anything differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
METRICS = ["success", "spl", "soft_spl", "path_length", "num_decision_steps", "termination", "distance_to_goal",
           "collisions", "vlm_calls", "stop_candidate_is_goal"]


def calls(cell: Path, episodes: set) -> Tuple[Dict[str, List[Tuple]], Dict[str, str]]:
    """VLM calls of a cell: per episode in order (method, step, input_key, output as JSON), and the command parses
    (input_key -> output) of the whole cell - a parse is cached by command, so a full run holds it in whichever
    episode used the command first."""
    seq: Dict[str, List[Tuple]] = {}
    parses: Dict[str, str] = {}
    with open(cell / "vlm_calls.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx = r.get("context") or {}
            if r.get("type", "vlm_call") != "vlm_call" or ctx.get("warmup"):
                continue
            out = json.dumps(r.get("output"), sort_keys=True)
            if r["method"] == "parse_goal":
                parses[r.get("input_key")] = out
            elif ctx.get("episode_id") in episodes:
                seq.setdefault(ctx["episode_id"], []).append((r["method"], ctx.get("step"), r.get("input_key"), out))
    return seq, parses


def p_target(output_json: str):
    """P(yes) of a verification output, else None."""
    o = json.loads(output_json)
    return o.get("p_target") if isinstance(o, dict) else None


def compare_cell(a: Path, b: Path, lines: List[str]) -> int:
    """Append the comparison of two cells to ``lines``; return the number of differences."""
    ea, eb = pd.read_csv(a / "episodes.csv").set_index("episode_id"), pd.read_csv(b / "episodes.csv").set_index("episode_id")
    eps = sorted(ea.index)                    # the check's episodes; B (a full dev run) may hold more
    diffs = 0
    missing = [e for e in eps if e not in eb.index]
    if missing:
        lines.append(f"- episodes missing in B: {missing}")
        diffs += len(missing)
    rows = []
    for e in eps:
        if e not in eb.index:
            continue
        bad = [m for m in METRICS if m in ea.columns and str(ea.at[e, m]) != str(eb.at[e, m])]
        rows.append(f"| {e} | {ea.at[e, 'success']} | {ea.at[e, 'num_decision_steps']} | "
                    f"{'same' if not bad else 'DIFF: ' + ', '.join(f'{m} {ea.at[e, m]} vs {eb.at[e, m]}' for m in bad)} |")
        diffs += bool(bad)
    lines += ["", "| episode | success (A) | steps (A) | metrics A vs B |", "|---|---|---|---|", *rows, ""]
    (ca, pa), (cb, pb) = calls(a, set(eps)), calls(b, set(eps))
    n_parse = len(pa)
    same_parse = sum(pb.get(k) == v for k, v in pa.items())
    lines.append(f"- command parses identical (same request, same answer): **{same_parse} of {n_parse}**")
    n = same = 0
    dp = 0.0
    first = ""
    for e in eps:
        la, lb = ca.get(e, []), cb.get(e, [])
        for i in range(max(len(la), len(lb))):
            n += 1
            x, y = (la[i] if i < len(la) else None), (lb[i] if i < len(lb) else None)
            if x == y:
                same += 1
                continue
            if x and y and x[:2] == y[:2] and p_target(x[3]) is not None and p_target(y[3]) is not None:
                dp = max(dp, abs(p_target(x[3]) - p_target(y[3])))
            if not first:
                which = "missing call" if not (x and y) else "image/prompt (input_key)" if x[2] != y[2] else "output"
                first = f"{e}, call {i + 1} ({(x or y)[0]}, step {(x or y)[1]}): {which}; A={x and x[3][:160]} B={y and y[3][:160]}"
    lines.append(f"- other VLM calls identical (input image and prompt hash, output): **{same} of {n}**"
                 + (f"; max |dP(yes)| = {dp:.2e}" if dp else ""))
    if first:
        lines.append(f"- first difference: {first}")
    return diffs + (n_parse - same_parse) + (n - same)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b", nargs="?", default="")
    ap.add_argument("--reference", action="store_true", help="compare A with the original dev runs")
    args = ap.parse_args()
    a = Path(args.a)
    m = yaml.safe_load((REPO / "docker/check/repro.yaml").read_text(encoding="utf-8"))
    lines = [f"# Reproducibility check: {a}" + (" vs the original dev runs" if args.reference else f" vs {args.b}"), ""]
    total = 0
    if not args.reference:
        b = Path(args.b)
        for p in (a, b):
            t = (p / "render.txt").read_text(encoding="utf-8").strip() if (p / "render.txt").exists() else "-"
            h = hashlib.sha256((p / "render.png").read_bytes()).hexdigest()[:16] if (p / "render.png").exists() else "-"
            lines.append(f"- {p}: test frame sha256 {h}; " + "; ".join(t.splitlines()))
        lines.append("")
    for arm in m["arms"]:
        ca = a / "repro" / arm["key"] / "seed0"
        cb = REPO / arm["reference"] if args.reference else Path(args.b) / "repro" / arm["key"] / "seed0"
        lines += [f"## {arm['key']}: {arm['label']}", f"A = {ca}", f"B = {cb}"]
        total += compare_cell(ca, cb, lines)
        lines.append("")
    lines.append(f"**{'IDENTICAL' if total == 0 else f'{total} differences'}**")
    text = "\n".join(lines) + "\n"
    out = (a / "compare_reference.md") if args.reference else (Path(args.b) / "compare.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
