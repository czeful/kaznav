"""The cascade arm of spoken commands (D-078): Qolda-AVL's transcript of a recording, parsed by the text model.

    bash tools/start_qolda.sh
    bash tools/wsl.sh python tools/parse_transcripts.py --speech runs/speech/tts_mms_skip \
        --out runs/speech/cascade_tts_mms_qolda -- backend=qolda backend.prompts.parse_goal=default/parse_goal/v4

``--speech`` is a run of tools/parse_speech_commands.py made with ``--transcribe`` (its commands.jsonl holds the
transcripts). Each transcript is parsed like a typed command (the same backend and prompt as the text arms), and the
call is re-keyed to the command's own text (``rekey_parse_record``), so a robot run replays it with
``++backend.parse_replay.prompt=default/parse_goal/v4 "++backend.parse_replay.sources=[<out>/vlm_calls.jsonl]"``.
No transcript, or a failed parse, leaves no usable record: the episode counts as "did not understand".

Writes <out>/vlm_calls.jsonl, <out>/commands.jsonl (understood or not, per command) and <out>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.language.backends.parse_replay import rekey_parse_record  # noqa: E402
from kznav.language.base import ListSink  # noqa: E402
from kznav.language.goal import GoalSettings, parse_command  # noqa: E402
from kznav.language.lexicon import Lexicon  # noqa: E402
from kznav.language.registry import create_backend  # noqa: E402
from kznav.run.config import compose_config  # noqa: E402
from parse_speech_commands import summarize  # noqa: E402  (tools/ is the script's own directory)


def main() -> None:
    """Command-line entry point: parse each transcript with the text model and write the summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speech", required=True, help="a parse_speech_commands.py run made with --transcribe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="+", default=["kk", "kk_ru", "en"])
    ap.add_argument("--lexicon", default="lexicon/hm3d_objectnav.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = compose_config(args.overrides)
    if cfg["backend"].get("parse_replay"):
        raise SystemExit("the cascade parses live: no parse_replay in the backend config")
    lex = Lexicon.load(REPO / args.lexicon)
    calls = ListSink()
    backend = create_backend(cfg["backend"], sink=calls)
    backend.context.run_id = Path(args.out).name
    rows_in = [json.loads(line) for line in (Path(args.speech) / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_in = [r for r in rows_in if r["lang"] in args.langs and r.get("input") == "speech"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "vlm_calls.jsonl", "w", encoding="utf-8") as fc:
        for r in sorted(rows_in, key=lambda r: (r["command_id"], r["lang"])):
            transcript = (r.get("transcript") or "").strip()
            row = {"command_id": r["command_id"], "category": r["category"], "lang": r["lang"], "text": r["text"],
                   "transcript": transcript, "cer": r.get("cer"), "parse_ok": False, "target": "", "target_en": "",
                   "target_match": False, "hypothesis_match": False, "target_en_ok": False, "hypotheses": [],
                   "attempts": 0, "latency_s": None}
            if transcript:
                backend.context.episode_id = f"{r['command_id']}|{r['lang']}"
                g = parse_command(backend, calls, command_id=r["command_id"], text=transcript, lang=r["lang"],
                                  category=r["category"], word_type="universal", lexicon=lex, settings=GoalSettings())
                rec = g.parse_record
                if rec is not None:
                    d = rekey_parse_record(rec.to_json_dict(), r["text"], r["lang"], transcript_cer=r.get("cer"),
                                           asr=r.get("audio"), asr_model="qolda-avl-5b")
                    d["model_id"] = f"{rec.model_id} (ASR: qolda-avl-5b)"
                    fc.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                    row["latency_s"] = round(rec.latency_total_s, 2)
                en = g.target_en or ""
                row.update(parse_ok=g.parse_ok, target=g.target, target_en=en, target_match=bool(g.target_match),
                           hypothesis_match=bool(g.hypothesis_match), hypotheses=list(g.hypotheses),
                           attempts=g.parse_attempts,
                           target_en_ok=bool(en) and bool(lex.match_any_en(r["category"], [en])))
            rows.append(row)
            print(f"[cascade] {r['command_id']} {r['lang']}: heard {transcript!r} -> ok={row['parse_ok']} "
                  f"obj={row['hypothesis_match']} target_en={row['target_en']!r}", flush=True)
    with open(out / "commands.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    title = f"Cascade: Qolda-AVL transcript -> {cfg['backend']['name']} ({cfg['backend']['prompts']['parse_goal']})"
    (out / "summary.md").write_text(summarize(rows, title), encoding="utf-8")
    print(summarize(rows, title))
    print("CASCADE_DONE", flush=True)


if __name__ == "__main__":
    main()
