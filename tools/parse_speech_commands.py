"""Parse the natural commands with Qolda-AVL - from their recordings or their text - before the robot runs (D-077).

Runs in the Qolda-AVL venv (transformers 5.x, the whole GPU for ~10.5 GB; nothing else on it):

    source ~/avl-env/bin/activate
    python tools/parse_speech_commands.py --input speech --speaker tts_mms --out runs/speech/dev_tts_native

Writes into ``--out``:

* ``vlm_calls.jsonl`` - the parse calls, the replay source of the robot runs (``backend=replay``,
  ``backend.prompts.parse_goal`` = the prompt used here, ``backend.replay.sources=[<out>/vlm_calls.jsonl]``);
* ``commands.jsonl`` - one row per (command, language): parse ok, target in dictionary form, an English hypothesis
  naming the object, target_en naming the object (the same lexicon checks as the robot runs' goal events), the
  attempts and the latency; with ``--transcribe`` also the model's transcript of the recording and its character
  error rate against the command text (a diagnostic call, not in vlm_calls.jsonl);
* ``summary.md`` - the same per language.

Resumes: (command, language) rows already in commands.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.language.base import ListSink, TeeSink  # noqa: E402
from kznav.language.commands import NaturalCommands  # noqa: E402
from kznav.language.goal import GoalSettings, parse_command  # noqa: E402
from kznav.language.lexicon import normalize  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402

PROMPTS = {"speech": "speech/parse_goal/v1", "text": "default/parse_goal/v4"}


def cer(hyp: str, ref: str) -> float:
    """Character error rate after the lexicon's normalisation (case, punctuation); spaces count."""
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return float("nan")
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def set_dotted(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def summarize(rows, title: str) -> str:
    langs = sorted({r["lang"] for r in rows}, key=["kk", "kk_ru", "en"].index)
    cols = [("parsed", "parse_ok"), ("target in dictionary form", "target_match"),
            ("a hypothesis names the object", "hypothesis_match"), ("target_en names the object", "target_en_ok")]
    lines = [f"# {title}", "", "| | " + " | ".join(langs) + " | all |", "|---" * (len(langs) + 2) + "|"]
    for label, key in cols:
        vals = [sum(bool(r[key]) for r in rows if r["lang"] == lang) for lang in langs]
        ns = [sum(1 for r in rows if r["lang"] == lang) for lang in langs]
        lines.append(f"| {label} | " + " | ".join(f"{v}/{n}" for v, n in zip(vals, ns))
                     + f" | {sum(vals)}/{sum(ns)} |")
    for label, key, fmt in (("mean attempts", "attempts", "{:.2f}"), ("mean parse latency, s", "latency_s", "{:.1f}"),
                            ("mean CER of the transcript", "cer", "{:.3f}")):
        vals = []
        for lang in langs + [None]:
            xs = [r[key] for r in rows if (lang is None or r["lang"] == lang) and r.get(key) is not None]
            vals.append(fmt.format(sum(xs) / len(xs)) if xs else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    bad = [r for r in rows if not (r["parse_ok"] and r["hypothesis_match"])]
    if bad:
        lines += ["", "Commands without a hypothesis naming the object:", ""]
        lines += [f"- {r['command_id']} {r['lang']}: target={r['target']!r}, target_en={r['target_en']!r}, "
                  f"hypotheses={r['hypotheses']}" + (f", heard: {r['transcript']!r}" if r.get("transcript") else "")
                  for r in bad]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["speech", "text"], required=True)
    ap.add_argument("--speaker", default=None, help="speech: data/speech/<speaker>/ (default: the config's)")
    ap.add_argument("--thinking", choices=["native", "skip"], default=None, help="default: the config's")
    ap.add_argument("--langs", nargs="+", default=["kk", "kk_ru", "en"])
    ap.add_argument("--commands", default="commands/natural/hm3d_objectnav_v1.yaml")
    ap.add_argument("--role", default="dev", help="dev: the dev commands (default) | benchmark: approved test ones")
    ap.add_argument("--config", default="configs/backend/qolda_avl.yaml")
    ap.add_argument("--transcribe", action="store_true", help="speech: also ask for a transcript (diagnostic)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    spec["input"] = args.input
    set_dotted(spec, "prompts.parse_goal", PROMPTS[args.input])
    if args.speaker:
        set_dotted(spec, "speech.speaker", args.speaker)
    if args.thinking:
        spec["thinking"] = args.thinking
    set_dotted(spec, "speech.commands", args.commands)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "commands.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["command_id"], r["lang"])] = r
    calls_f = open(out / "vlm_calls.jsonl", "a", encoding="utf-8")
    rows_f = open(rows_path, "a", encoding="utf-8")

    def write_call(rec) -> None:
        calls_f.write(json.dumps(rec.to_json_dict(), ensure_ascii=False, default=str) + "\n")
        calls_f.flush()

    calls = ListSink()
    backend = create_backend(spec, sink=TeeSink(calls, write_call))
    backend.context.run_id = out.name
    bank = NaturalCommands(REPO / args.commands, REPO, role=args.role)
    lex = bank.lexicon
    usable = sorted((c for c in bank.commands if bank.usable(c)), key=lambda c: (c.category, c.id))
    todo = [(c, lang) for c in usable for lang in args.langs if (c.id, lang) not in done]
    print(f"[speech] {args.input}, thinking={backend.thinking}, speaker={spec.get('speech', {}).get('speaker')}: "
          f"{len(todo)} to parse, {len(done)} done", flush=True)
    t_start = time.time()
    for i, (c, lang) in enumerate(todo, 1):
        text = c.texts[lang].strip()
        backend.context.episode_id = f"{c.id}|{lang}"
        g = parse_command(backend, calls, command_id=c.id, text=text, lang=lang, category=c.category,
                          word_type="universal", lexicon=lex, settings=GoalSettings())
        rec = g.parse_record
        row = {"command_id": c.id, "category": c.category, "lang": lang, "text": text, "input": args.input,
               "parse_ok": g.parse_ok, "target": g.target, "target_en": g.target_en,
               "target_match": bool(g.target_match), "hypothesis_match": bool(g.hypothesis_match),
               "target_en_ok": bool(g.target_en) and bool(lex.match_any_en(c.category, [g.target_en])),
               "hypotheses": list(g.hypotheses), "rooms": list(g.rooms), "attempts": g.parse_attempts,
               "latency_s": None if rec is None else round(rec.latency_total_s, 2), "error": g.error,
               "audio": None if rec is None else rec.inputs.get("audio"), "transcript": None, "cer": None}
        if args.transcribe and args.input == "speech":
            clip = backend.bank.clip(text, lang)
            transcript, _ = backend.transcribe(clip)
            row["transcript"], row["cer"] = transcript, round(cer(transcript, text), 4)
        rows_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_f.flush()
        done[(c.id, lang)] = row
        el = time.time() - t_start
        print(f"[speech] {i}/{len(todo)} {c.id} {lang}: ok={g.parse_ok} obj={row['hypothesis_match']} "
              f"target_en={g.target_en!r} {row['latency_s']}s" + (f" cer={row['cer']}" if row["cer"] is not None
                                                                   else "") + f" (elapsed {el / 60:.1f} min)",
              flush=True)
    calls_f.close()
    rows_f.close()
    title = (f"Qolda-AVL parse, input={args.input}, thinking={backend.thinking}"
             + (f", speaker {spec['speech']['speaker']}" if args.input == "speech" else ""))
    rows = [done[k] for k in sorted(done)]
    (out / "summary.md").write_text(summarize(rows, title), encoding="utf-8")
    print(summarize(rows, title))
    print("SPEECH_PARSE_DONE", flush=True)


if __name__ == "__main__":
    main()
