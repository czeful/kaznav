"""Synthetic recordings of the natural commands, to test the spoken-command chain before people record (D-077).

Meta's MMS-TTS (VITS; facebook/mms-tts-kaz for kk and kk_ru - the mixed commands are Cyrillic, read with Kazakh
phonetics - and facebook/mms-tts-eng for en; CC-BY-NC 4.0, research use). A pilot speaker only: clean synthetic
speech is easier than a person in a room, so its numbers are not the paper's. CPU, a few seconds per command:

    source ~/avl-env/bin/activate
    python tools/make_tts_commands.py --speaker tts_mms

Writes data/speech/<speaker>/<command id>__<lang>.wav (16 kHz mono, peak 0.9) and manifest.jsonl there (text,
model, letters the voice's alphabet lacks - dropped by its tokenizer). The noise of VITS is seeded per clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.language.commands import NaturalCommands  # noqa: E402
from kznav.language.speech import clip_path, write_wav  # noqa: E402

VOICES = {"kk": "facebook/mms-tts-kaz", "kk_ru": "facebook/mms-tts-kaz", "en": "facebook/mms-tts-eng"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default="tts_mms")
    ap.add_argument("--root", default="data/speech")
    ap.add_argument("--langs", nargs="+", default=["kk", "kk_ru", "en"])
    ap.add_argument("--commands", default="commands/natural/hm3d_objectnav_v1.yaml")
    ap.add_argument("--role", default="dev")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer, VitsModel

    bank = NaturalCommands(REPO / args.commands, REPO, role=args.role)
    usable = sorted((c for c in bank.commands if bank.usable(c)), key=lambda c: (c.category, c.id))
    voices = {}
    root = REPO / args.root
    manifest = []
    for lang in args.langs:
        repo = VOICES[lang]
        if repo not in voices:
            tok = AutoTokenizer.from_pretrained(repo)
            if getattr(tok, "is_uroman", False):
                raise SystemExit(f"{repo} expects romanised (uroman) text; not supported here")
            voices[repo] = (tok, VitsModel.from_pretrained(repo).eval())
        tok, model = voices[repo]
        vocab = set(tok.get_vocab())
        for c in usable:
            text = c.texts[lang].strip()
            norm = tok.normalize_text(text) if hasattr(tok, "normalize_text") else text.lower()
            missing = sorted({ch for ch in norm if ch not in vocab and not ch.isspace()})
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
            rate = int(model.config.sampling_rate)
            assert rate == 16000, rate
            wav = 0.9 * wav / max(1e-6, float(np.abs(wav).max()))
            path = clip_path(root, args.speaker, c.id, lang)
            write_wav(path, wav, rate)
            manifest.append({"command_id": c.id, "lang": lang, "text": text, "voice": repo, "seed": args.seed,
                             "duration_s": round(len(wav) / rate, 2), "letters_not_in_voice": missing,
                             "file": str(path.relative_to(REPO))})
            print(f"[tts] {c.id} {lang} {len(wav) / rate:.1f}s" + (f" missing {missing}" if missing else ""),
                  flush=True)
    with open(root / args.speaker / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"TTS_DONE {len(manifest)} clips", flush=True)


if __name__ == "__main__":
    main()
