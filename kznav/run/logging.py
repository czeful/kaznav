"""Append-only JSONL logs and a content-addressed image store for one run.

Layout of a run directory::

    runs/<run_id>/
      manifest.json
      vlm_calls.jsonl     every model request/response (raw), incl. failures and warm-up
      events.jsonl        decisions per step (detections, verifications, stop reasons, ...)
      episodes.csv        one row per finished episode (kznav.eval.records)
      images/<sha256>.png every image sent to a model, stored once

Lines are flushed after every write so that a crash loses at most the line
being written; resuming skips episodes already present in episodes.csv.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..language.base import CallRecord


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "value") and hasattr(o, "name"):  # enums
        return o.value
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _sanitize(o: Any) -> Any:
    """Replace NaN/inf by strings so every line is strict JSON (json.loads in other tools)."""
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return "NaN" if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


class JsonlWriter:
    """Appends JSON objects to a file, one per line, flushed after every write.

    Args:
        path: Output file (parent directories are created).
        fsync: Also force each line to disk.
    """
    def __init__(self, path: Path, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._fsync = fsync

    def write(self, obj: Dict[str, Any]) -> None:
        """Append one object as a line (NaN and infinities are written as null)."""
        self._f.write(json.dumps(_sanitize(obj), ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")
        self._f.flush()
        if self._fsync:
            os.fsync(self._f.fileno())

    def close(self) -> None:
        """Close the file."""
        self._f.close()


class ImageStore:
    """Content-addressed store of the images sent to models: each file is named by its SHA-256.

    Args:
        root: Directory of the stored images.
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, ext: str = "png") -> str:
        """Store image bytes once (atomically) and return their SHA-256."""
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / f"{digest}.{ext}"
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return digest


class JsonlCallSink:
    """``CallSink`` that writes every VLM call to vlm_calls.jsonl and its images to the image store."""

    def __init__(self, writer: JsonlWriter, images: ImageStore, image_ext: str = "png"):
        self.writer = writer
        self.images = images
        self.image_ext = image_ext
        self.n_calls = 0

    def __call__(self, rec: CallRecord) -> None:
        """Write a call record to the call log and its input images to the image store."""
        d = rec.to_json_dict()
        d["image_files"] = [f"images/{self.images.put(b, self.image_ext)}.{self.image_ext}" for b in rec.images]
        self.writer.write(d)
        self.n_calls += 1


class RunLogger:
    """Owns the run directory's JSONL writers."""

    def __init__(self, run_dir: Path, fsync: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.images = ImageStore(self.run_dir / "images")
        self.calls = JsonlWriter(self.run_dir / "vlm_calls.jsonl", fsync)
        self.events = JsonlWriter(self.run_dir / "events.jsonl", fsync)
        self.call_sink = JsonlCallSink(self.calls, self.images)
        #: Set by the benchmark runner per episode attempt; events of an attempt that failed or was interrupted
        #: are told apart from the completed one (``episode_done`` events name the completed attempts).
        self.attempt: Optional[str] = None

    def event(self, event_type: str, *, episode_id: str, step: int, **payload: Any) -> None:
        """Append one event of an episode to ``events.jsonl`` (tagged with the current attempt)."""
        extra = {"attempt": self.attempt} if self.attempt is not None else {}
        self.events.write({"type": event_type, "episode_id": episode_id, "step": step, **extra, **payload})

    def close(self) -> None:
        """Close the call log and the event log."""
        self.calls.close()
        self.events.close()


def read_jsonl(path: Path, type_filter: Optional[str] = None):
    """Yield the objects of a JSONL file, optionally only those whose ``type`` equals ``type_filter``."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if type_filter is None or obj.get("type") == type_filter:
                    yield obj


def completed_events(path: Path, type_filter: Optional[str] = None):
    """Events of completed episode attempts only (D-046): an event with an ``attempt`` id is kept if an
    ``episode_done`` event names that attempt; events without one (debug runs, older runs) are all kept."""
    events = list(read_jsonl(path))
    done = {e["attempt"] for e in events if e.get("type") == "episode_done" and e.get("attempt")}
    for e in events:
        if (type_filter is None or e.get("type") == type_filter) and (e.get("attempt") is None or e["attempt"] in done):
            yield e
