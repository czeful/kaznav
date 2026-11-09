"""Spoken commands (D-077): the audio of a command, for a model that hears the command instead of reading it.

Recordings live outside git (voices are personal data) in one folder per speaker::

    data/speech/<speaker>/<command id>__<lang>.wav      e.g. data/speech/S01/chair-a01-01__kk.wav

``speaker`` is an anonymised id (S01, ...; ``tts_mms`` for the synthetic pilot), never a name. Any WAV
(8/16/24/32-bit PCM or 32-bit float, mono or stereo, any sample rate) is read and turned into mono float32 at
16 kHz, what Whisper-style encoders expect. Only the standard library and numpy are used.

``level`` (D-085): ``none`` passes the recording on as recorded; ``peak`` scales it so that its peak is
``PEAK_LEVEL``, as a microphone's automatic gain control would (phone recordings of S01 peak at 0.07-0.21).
"""
from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .lexicon import normalize

SAMPLE_RATE = 16000
LEVELS = ("none", "peak")
PEAK_LEVEL = 0.9                # about -1 dBFS


@dataclass(frozen=True)
class SpeechClip:
    """One command recording, loaded and resampled, with its provenance."""
    path: str
    speaker: str
    command_id: str
    lang: str
    sha256: str                 # of the file's bytes: a re-recorded command is a different input
    audio: np.ndarray           # mono float32 in [-1, 1] at SAMPLE_RATE
    file_rate: int
    level: str = "none"         # loudness normalisation applied to ``audio`` (one of LEVELS)

    @property
    def duration_s(self) -> float:
        """Length of the clip in seconds."""
        return len(self.audio) / SAMPLE_RATE

    def describe(self) -> Dict[str, object]:
        """Provenance of the clip for the call record (path, speaker, command, hash, duration, file rate, level)."""
        d: Dict[str, object] = {"path": self.path, "speaker": self.speaker, "command_id": self.command_id,
                                "lang": self.lang, "sha256": self.sha256, "duration_s": round(self.duration_s, 3),
                                "file_rate": self.file_rate}
        if self.level != "none":                # records made before D-085 stay byte-identical
            d["level"] = self.level
        return d


def normalize_level(x: np.ndarray, level: str) -> np.ndarray:
    """Loudness normalisation of a clip (D-085).

    Args:
        x: Mono float32 audio.
        level: ``none`` (returned unchanged) or ``peak`` (scaled so that the peak is ``PEAK_LEVEL``).

    Returns:
        The clip as float32; silence stays silence.

    Raises:
        ValueError: If ``level`` is unknown.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if level == "none" or peak == 0.0:
        return x.astype(np.float32)
    return (x * (PEAK_LEVEL / peak)).astype(np.float32)


def resample(x: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited (FFT) resampling, as scipy.signal.resample; the clip is short, so one FFT is cheap."""
    if rate == target or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * target / rate))
    spec = np.fft.rfft(x.astype(np.float64))
    keep = n // 2 + 1
    out = np.zeros(keep, dtype=complex)
    m = min(keep, len(spec))
    out[:m] = spec[:m]
    return (np.fft.irfft(out, n) * (n / len(x))).astype(np.float32)


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """(mono float32 at SAMPLE_RATE, the file's own sample rate)."""
    with wave.open(str(path), "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, rate), rate


def write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """16-bit PCM mono."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def clip_path(root: Path, speaker: str, command_id: str, lang: str) -> Path:
    """Path of a recording: ``<root>/<speaker>/<command id>__<lang>.wav``."""
    return Path(root) / speaker / f"{command_id}__{lang}.wav"


class SpeechBank:
    """The recordings of one speaker, looked up by the command text the run hands to the parser.

    ``texts`` maps (command id, lang) -> command text (the natural command bank). The run calls the parser with
    (text, lang); the bank finds the command id of that text and loads its recording. ``level`` is the loudness
    normalisation applied to every clip (``normalize_level``).
    """

    def __init__(self, root: Path, speaker: str, texts: Dict[Tuple[str, str], str], level: str = "none"):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.root, self.speaker, self.level = Path(root), speaker, level
        self._ids: Dict[Tuple[str, str], str] = {}
        for (cid, lang), text in texts.items():
            key = (lang, normalize(text))
            if key in self._ids and self._ids[key] != cid:
                raise ValueError(f"commands {self._ids[key]} and {cid} have the same {lang} text")
            self._ids[key] = cid
        self._cache: Dict[Tuple[str, str], SpeechClip] = {}

    @classmethod
    def from_commands(cls, root: Path, speaker: str, commands: Iterable, level: str = "none") -> "SpeechBank":
        """``commands``: NaturalCommand objects (kznav.language.commands)."""
        return cls(root, speaker, {(c.id, lang): t for c in commands for lang, t in c.texts.items() if t.strip()},
                   level)

    def command_id(self, text: str, lang: str) -> str:
        """The command id whose text in ``lang`` equals ``text`` (after normalisation).

        Raises:
            KeyError: If no command has this text.
        """
        cid = self._ids.get((lang, normalize(text)))
        if cid is None:
            raise KeyError(f"no command with this {lang} text in the command bank: {text!r}")
        return cid

    def clip(self, text: str, lang: str) -> SpeechClip:
        """The recording of the command with this text, loaded once and cached.

        Raises:
            KeyError: If no command has this text.
            FileNotFoundError: If the speaker has no recording of the command.
        """
        cid = self.command_id(text, lang)
        if (cid, lang) not in self._cache:
            path = clip_path(self.root, self.speaker, cid, lang)
            if not path.is_file():
                raise FileNotFoundError(f"no recording {path} (speaker {self.speaker}, command {cid}, {lang})")
            audio, rate = read_wav(path)
            self._cache[(cid, lang)] = SpeechClip(str(path), self.speaker, cid, lang,
                                                  hashlib.sha256(path.read_bytes()).hexdigest(),
                                                  normalize_level(audio, self.level), rate, self.level)
        return self._cache[(cid, lang)]

    def available(self, command_id: str, lang: str) -> Optional[Path]:
        """Path of the recording of a command, or None if the speaker has not recorded it."""
        p = clip_path(self.root, self.speaker, command_id, lang)
        return p if p.is_file() else None
