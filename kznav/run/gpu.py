"""GPU memory bookkeeping for runs that share the GPU with a local model server (D-006).

On Windows/WDDM an allocation beyond physical VRAM does not fail: the driver silently pages to
shared system memory and everything gets slower. A run whose server peak exceeded the physical
memory therefore still "works", but its latencies are invalid; the manifest must say so.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


def _root(base_url: str) -> str:
    return base_url.rstrip("/").rsplit("/v1", 1)[0]


def server_stats(base_url: str, reset_peak: bool = False) -> Optional[Dict[str, Any]]:
    """``GET /kznav/stats`` (or ``POST /kznav/reset_peak``) of servers/qolda_server.py; None if unavailable."""
    try:
        url = _root(base_url) + ("/kznav/reset_peak" if reset_peak else "/kznav/stats")
        r = httpx.post(url, timeout=5.0) if reset_peak else httpx.get(url, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError):
        return None


def gpu_total_gb() -> Optional[float]:
    """Total memory of CUDA device 0 in GiB, or None without torch or CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    except Exception:   # no torch / no CUDA: not measurable here
        pass
    return None


def memory_report(base_url: str) -> Dict[str, Any]:
    """Server peak since the last reset vs. physical GPU memory, for the run manifest."""
    st = server_stats(base_url)
    total = gpu_total_gb()
    out: Dict[str, Any] = {"server_stats": st, "gpu_total_gb": total, "server_peak_exceeds_gpu": None}
    if st is not None and total is not None:
        out["server_peak_exceeds_gpu"] = float(st["max_allocated_gb"]) > total
    return out
