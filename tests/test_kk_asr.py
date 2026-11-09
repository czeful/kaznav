"""D-086: the Kazakh ASR model's transcripts (a fake recogniser here) in the format the cascade reads."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kznav.language.speech import SAMPLE_RATE, SpeechBank, clip_path, write_wav

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import parse_transcripts  # noqa: E402
import transcribe_kk_asr as kk  # noqa: E402

TEXT = "Орындықты тауып берші"
ASR = {"file": "kk_punct_full_fixed.nemo", "sha256": "00e6", "lookahead_ms": 1040}


def rows(tmp_path, heard):
    t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    write_wav(clip_path(tmp_path, "S01", "chair-a01-01", "kk"), (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))
    bank = SpeechBank(tmp_path, "S01", {("chair-a01-01", "kk"): TEXT}, level="peak")
    got = []
    out = kk.transcribe_rows([("chair-a01-01", "chair", "kk", TEXT)], bank, lambda a: got.append(a) or heard, ASR)
    return out, got


def test_rows_carry_the_transcript_its_cer_and_the_model(tmp_path):
    (r,), got = rows(tmp_path, "Орындықты тауып берші.")
    assert np.abs(got[0]).max() == pytest.approx(0.9)                 # the clip as the bank's level makes it
    assert r["transcript"] == "Орындықты тауып берші." and r["cer"] == 0.0 and r["input"] == "speech"
    assert r["category"] == "chair" and r["asr_model"] == kk.ASR_MODEL and r["asr"] == ASR
    assert r["audio"]["level"] == "peak" and r["audio"]["command_id"] == "chair-a01-01"


def test_the_cascade_parses_them_and_names_the_asr_model(tmp_path, monkeypatch):
    out, _ = rows(tmp_path, "Орындық тауып берші")
    src = tmp_path / "asr"
    src.mkdir()
    (src / "commands.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                                        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["parse_transcripts.py", "--speech", str(src), "--out", str(tmp_path / "c"),
                                      "--langs", "kk", "--", "backend=mock"])
    parse_transcripts.main()
    (rec,) = [json.loads(line) for line in (tmp_path / "c/vlm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rec["inputs"]["text"] == TEXT and rec["inputs"]["parsed_text"] == "Орындық тауып берші"
    assert rec["inputs"]["asr_model"] == kk.ASR_MODEL and rec["inputs"]["asr_settings"] == ASR
    assert rec["model_id"].endswith(f"(ASR: {kk.ASR_MODEL})")
    assert kk.ASR_MODEL in (tmp_path / "c/summary.md").read_text(encoding="utf-8").splitlines()[0]
