"""D-077: spoken commands - audio files, the speech bank, the Qolda-AVL backend (with a fake model) and replay."""
import json
import wave

import numpy as np
import pytest
import yaml

from kznav.language.backends.qolda_avl import QoldaAVLBackend, split_thinking
from kznav.language.backends.replay import ReplayBackend
from kznav.language.base import BackendConfigError, ListSink
from kznav.language.schemas import VerifyQuery
from kznav.language.speech import SAMPLE_RATE, SpeechBank, clip_path, read_wav, resample, write_wav

GOAL = ('{"target": "орындық", "target_en": "chair", "hypotheses": [{"en": "wooden chair", "w": 0.6}, '
        '{"en": "office chair", "w": 0.4}], "attributes": {"material": [], "color": [], "size": []}, '
        '"rooms": ["living room"], "landmarks": []}')
TEXT = "Орындықты тауып бер"


def spec(repo_root, **kw):
    s = yaml.safe_load((repo_root / "configs/backend/qolda_avl.yaml").read_text(encoding="utf-8"))
    s.update({"thinking": "native", **kw})
    return s


class FakeRuntime:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, messages, audio, max_new_tokens, thinking):
        self.calls.append({"messages": messages, "audio": audio, "thinking": thinking})
        return self.outputs.pop(0)


def tone(freq, rate, seconds=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(x, rate=SAMPLE_RATE):
    return float(np.argmax(np.abs(np.fft.rfft(x))) * rate / len(x))


def test_resampling_keeps_the_pitch_and_the_length():
    y = resample(tone(1000, 48000), 48000)
    assert len(y) == SAMPLE_RATE // 2 and abs(peak_hz(y) - 1000) < 5


def test_wav_round_trip_stereo_and_24_bit(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, tone(440, SAMPLE_RATE))
    x, rate = read_wav(p)
    assert rate == SAMPLE_RATE and abs(peak_hz(x) - 440) < 5 and np.abs(x).max() == pytest.approx(0.5, abs=1e-3)
    s = (tone(440, 44100) * (1 << 23)).astype(np.int32)
    raw = np.stack([s, s], axis=1).reshape(-1)
    b = np.stack([raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF], axis=1).astype(np.uint8).tobytes()
    q = tmp_path / "b.wav"
    with wave.open(str(q), "wb") as w:
        w.setnchannels(2), w.setsampwidth(3), w.setframerate(44100), w.writeframes(b)
    y, rate = read_wav(q)
    assert rate == 44100 and len(y) == SAMPLE_RATE // 2 and abs(peak_hz(y) - 440) < 5


def bank(tmp_path, record=True):
    texts = {("chair-a01-01", "kk"): TEXT, ("chair-a01-01", "en"): "Find the chair."}
    if record:
        write_wav(clip_path(tmp_path, "S01", "chair-a01-01", "kk"), tone(300, SAMPLE_RATE))
    return SpeechBank(tmp_path, "S01", texts)


def test_bank_finds_the_recording_by_the_command_text(tmp_path):
    b = bank(tmp_path)
    c = b.clip("орындықты тауып бер!", "kk")             # the lexicon's normalisation: case, punctuation
    assert c.command_id == "chair-a01-01" and c.speaker == "S01" and c.duration_s == pytest.approx(0.5)
    with pytest.raises(FileNotFoundError):
        b.clip("Find the chair.", "en")
    with pytest.raises(KeyError):
        b.clip("Диванды тап", "kk")
    with pytest.raises(ValueError, match="same kk text"):
        SpeechBank(tmp_path, "S01", {("a", "kk"): TEXT, ("b", "kk"): TEXT})


def test_split_thinking():
    assert split_thinking("hmm\n</think>\n\n{}", "native") == ("hmm", "{}")
    assert split_thinking("still thinking {", "native") == ("still thinking {", "")
    assert split_thinking(" {} ", "skip") == ("", "{}")


def test_speech_parse_sends_the_audio_never_the_text(repo_root, prompt_lib, tmp_path):
    rt = FakeRuntime([f"the user wants a chair</think>\n\n{GOAL}"])
    sink = ListSink()
    be = QoldaAVLBackend(spec(repo_root), prompt_lib, sink, runtime=rt, speech_bank=bank(tmp_path))
    g = be.parse_goal(TEXT, "kk")
    assert g.target_en == "chair"
    msgs = rt.calls[0]["messages"]
    assert TEXT not in json.dumps(msgs, ensure_ascii=False)
    assert msgs[-1]["content"][0] == {"type": "audio"}
    assert msgs[-1]["content"][1]["text"] == prompt_lib.load("speech/parse_goal/v1").data["audio_instruction"]
    assert len(rt.calls[0]["audio"]) == 1 and len(rt.calls[0]["audio"][0]) == SAMPLE_RATE // 2
    rec = sink.records[-1]
    assert rec.ok and rec.inputs["audio"]["command_id"] == "chair-a01-01"
    assert rec.inputs["thinking"] == ["the user wants a chair"] and rec.inputs["text"] == TEXT


def test_unfinished_thinking_is_an_invalid_answer_and_is_retried(repo_root, prompt_lib, tmp_path):
    rt = FakeRuntime(["thinking forever {\"target\"", f"ok</think>{GOAL}"])
    sink = ListSink()
    be = QoldaAVLBackend(spec(repo_root), prompt_lib, sink, runtime=rt, speech_bank=bank(tmp_path))
    assert be.parse_goal(TEXT, "kk").target == "орындық"
    rec = sink.records[-1]
    assert rec.json_retries == 1 and [a.ok for a in rec.attempts] == [False, True]
    retry = rt.calls[1]["messages"]
    assert retry[-2]["role"] == "assistant" and TEXT not in json.dumps(retry, ensure_ascii=False)


def test_text_input_reads_the_command_with_the_text_arms_prompt(repo_root, prompt_lib):
    rt = FakeRuntime([GOAL])
    s = spec(repo_root, input="text", thinking="skip")
    s["prompts"] = dict(s["prompts"], parse_goal="default/parse_goal/v4")
    be = QoldaAVLBackend(s, prompt_lib, ListSink(), runtime=rt)
    be.parse_goal(TEXT, "kk")
    assert rt.calls[0]["messages"][-1] == {"role": "user", "content": TEXT} and rt.calls[0]["audio"] is None
    with pytest.raises(BackendConfigError, match="parse prompt"):
        QoldaAVLBackend(spec(repo_root, input="text"), prompt_lib, ListSink(), runtime=rt)
    with pytest.raises(BackendConfigError):
        be.verify_candidate(np.zeros((8, 8, 3), np.uint8), VerifyQuery("chair", "en"))


def test_robot_runs_replay_the_speech_parse(repo_root, prompt_lib, tmp_path):
    sink = ListSink()
    be = QoldaAVLBackend(spec(repo_root), prompt_lib, sink, runtime=FakeRuntime([f"</think>{GOAL}"]),
                         speech_bank=bank(tmp_path))
    g = be.parse_goal(TEXT, "kk")
    src = tmp_path / "vlm_calls.jsonl"
    src.write_text(json.dumps(sink.records[-1].to_json_dict(), ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    rp = yaml.safe_load((repo_root / "configs/backend/replay.yaml").read_text(encoding="utf-8"))
    rp["replay"]["sources"] = [str(src)]
    rp["prompts"] = dict(rp["prompts"], parse_goal="speech/parse_goal/v1")
    out = ReplayBackend(rp, prompt_lib, ListSink()).parse_goal(TEXT, "kk")
    assert out == g
    rp["prompts"]["parse_goal"] = "default/parse_goal/v4"            # the text arms' prompt: a different key
    with pytest.raises(Exception, match="no recorded"):
        ReplayBackend(rp, prompt_lib, ListSink()).parse_goal(TEXT, "kk")


def test_speech_prompt_is_v4_but_the_last_turn(prompt_lib):
    sp, v4 = prompt_lib.load("speech/parse_goal/v1").data, prompt_lib.load("default/parse_goal/v4").data
    for key in ("system", "fewshot", "fewshot_targets", "user_template", "retry_template"):
        assert sp[key] == v4[key], key
    assert sp["input"] == "audio" and sp["audio_instruction"]
