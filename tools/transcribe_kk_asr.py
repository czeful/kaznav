"""Transcribe the spoken commands with the Kazakh ASR model (D-086): the cascade's first step without Qolda-AVL.

Runs in its own venv (NeMo at the git commit the model's package pins, torch cu130; runs/jobs/d086_install.sh),
alone on the GPU (about 1 GB):

    source ~/nemo-env/bin/activate
    python tools/transcribe_kk_asr.py --model ~/kznav_data/models/kk_asr/kk_punct_full_fixed.nemo \
        --speaker S01 --langs kk kk_ru --out runs/speech/d086/N_S01

The model, ``kk_punct_full_fixed.nemo``, is a streaming FastConformer-RNNT for Kazakh (625M parameters) that writes
punctuation and case; it is not public (received with its own README, outside this repository). It is Kazakh only
and needs its language prompt (kk -> 105). Every recording is decoded in one offline pass with the model's longest
lookahead (``att_context_size`` [56, 13], 1040 ms), greedy, fp16 on the GPU - the way its authors measured their WER.

Writes into ``--out``:

* ``commands.jsonl`` - the rows of ``parse_speech_commands.py --transcribe-only`` plus ``asr_model`` and ``asr``
  (file hash, lookahead, precision), so tools/parse_transcripts.py takes them as they are;
* ``summary.md`` - mean CER per language and every transcript.

``--check WAV TEXT`` first decodes a reference recording and stops if its CER against TEXT is above 0.1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.language.commands import NaturalCommands  # noqa: E402
from kznav.language.speech import LEVELS, SpeechBank, read_wav  # noqa: E402
from parse_speech_commands import cer, summarize_transcripts  # noqa: E402  (tools/ is the script's own directory)

ASR_MODEL = "kk-asr-fastconformer-rnnt"        # the name the cascade's parse records carry
KK_PROMPT_ID = 105
ATT_CONTEXT = {"0": [56, 0], "240": [56, 3], "480": [56, 6], "1040": [56, 13]}   # lookahead, ms -> att_context_size


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 16 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class KkASR:
    """The model, loaded once; calling it turns a mono 16 kHz float32 clip into text."""

    def __init__(self, path: Path, mode: str = "1040", device: str = "cuda"):
        import nemo.collections.asr as nemo_asr
        import torch

        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device == "cuda" else torch.float32    # the CPU needs fp32
        model = nemo_asr.models.ASRModel.restore_from(str(path), map_location=device)
        model.eval()
        model.freeze()
        if self.dtype == torch.float16:
            model = model.half()
        model.encoder.set_default_att_context_size(ATT_CONTEXT[mode])
        self.model = model
        self.info = {"file": path.name, "sha256": sha256_file(path), "lookahead_ms": int(mode),
                     "att_context_size": ATT_CONTEXT[mode], "prompt": "kk", "decoding": "greedy", "device": device,
                     "precision": "fp16" if self.dtype == torch.float16 else "fp32"}

    def __call__(self, audio: np.ndarray) -> str:
        torch = self.torch
        with torch.no_grad():
            sig = torch.from_numpy(np.ascontiguousarray(audio, np.float32)).to(self.device, self.dtype).unsqueeze(0)
            n = torch.tensor([sig.shape[1]], dtype=torch.long, device=self.device)
            prompt = torch.full((1,), KK_PROMPT_ID, dtype=torch.long, device=self.device)
            enc, enc_len = self.model.forward(input_signal=sig, input_signal_length=n, prompt_indices=prompt)
            hyps = self.model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                       return_hypotheses=False)
        h = hyps[0]
        if isinstance(h, list):                 # older NeMo returns (best, all)
            h = h[0]
        return (h if isinstance(h, str) else h.text).strip()


def transcribe_rows(items: Sequence[Tuple[str, str, str, str]], bank: SpeechBank,
                    recognize: Callable[[np.ndarray], str], asr: Dict[str, object]) -> List[Dict[str, object]]:
    """One row per (command id, category, lang, text): the transcript of its recording and its CER."""
    rows = []
    for i, (cid, category, lang, text) in enumerate(items, 1):
        clip = bank.clip(text, lang)
        t0 = time.time()
        transcript = recognize(clip.audio)
        rows.append({"command_id": cid, "category": category, "lang": lang, "text": text, "input": "speech",
                     "transcript": transcript, "cer": round(cer(transcript, text), 4), "audio": clip.describe(),
                     "asr_model": ASR_MODEL, "asr": asr, "transcribe_latency_s": round(time.time() - t0, 3)})
        print(f"[kk-asr] {i}/{len(items)} {cid} {lang}: heard {transcript!r} cer={rows[-1]['cer']}", flush=True)
    return rows


def main() -> None:
    """Command-line entry point: transcribe every recording of the speaker and write the rows and the summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="kk_punct_full_fixed.nemo")
    ap.add_argument("--speaker", required=True, help="data/speech/<speaker>/")
    ap.add_argument("--langs", nargs="+", default=["kk", "kk_ru"])
    ap.add_argument("--commands", default="commands/natural/hm3d_objectnav_v1.yaml")
    ap.add_argument("--role", default="dev", help="dev: the dev commands (default) | benchmark: approved test ones")
    ap.add_argument("--speech-root", default="data/speech")
    ap.add_argument("--level", choices=list(LEVELS), default="none", help="loudness normalisation (D-085)")
    ap.add_argument("--mode", choices=list(ATT_CONTEXT), default="1040", help="lookahead, ms")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", nargs=2, metavar=("WAV", "TEXT"), help="a reference recording and its text")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    asr = KkASR(Path(args.model).expanduser(), args.mode, args.device)
    print(f"[kk-asr] loaded {asr.info}", flush=True)
    if args.check:
        heard = asr(read_wav(Path(args.check[0]))[0])
        err = cer(heard, args.check[1])
        print(f"[kk-asr] check: heard {heard!r}, CER {err:.3f}", flush=True)
        if err > 0.1:
            raise SystemExit("the reference recording is not recognised: check the environment and the model file")
    bank = NaturalCommands(REPO / args.commands, REPO, role=args.role)
    usable = sorted((c for c in bank.commands if bank.usable(c)), key=lambda c: (c.category, c.id))
    speech = SpeechBank.from_commands(REPO / args.speech_root, args.speaker, usable, args.level)
    items = [(c.id, c.category, lang, c.texts[lang].strip()) for c in usable for lang in args.langs]
    rows = transcribe_rows(items, speech, asr, asr.info)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "commands.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    text = summarize_transcripts(rows, f"Kazakh ASR ({asr.info['file']}, {args.mode} ms), speaker {args.speaker}")
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print("KK_ASR_DONE", flush=True)


if __name__ == "__main__":
    main()
