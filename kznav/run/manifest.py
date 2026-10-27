"""Run manifest: everything needed to know what exactly produced a set of numbers.

Contents: resolved config and its hash, seed, code version (git commit and
dirty flag), library versions, GPU / renderer, backend + prompt hashes, UTC time.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

TRACKED_PACKAGES = (
    "kznav", "numpy", "scipy", "pandas", "pydantic", "httpx", "hydra-core", "omegaconf",
    "habitat-sim", "habitat-lab", "torch", "torchvision", "transformers", "pillow",
)


def config_hash(config: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a resolved config (identifies the experimental condition)."""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    """Installed versions of the tracked packages (None for a package that is not installed)."""
    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def git_state(repo: Path) -> Dict[str, Any]:
    """Current commit of the repository and whether tracked files have uncommitted changes."""
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def gpu_info() -> Dict[str, Any]:
    """GPU name, memory and driver from ``nvidia-smi``, or None if it is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return {"nvidia_smi": out}
    except (OSError, subprocess.SubprocessError):
        return {"nvidia_smi": None}


def build_manifest(
    *,
    run_id: str,
    config: Dict[str, Any],
    seed: int,
    repo: Path,
    components: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything needed to reproduce a run: config and its hash, code state, environment, components.

    Args:
        run_id: Identifier of the run.
        config: The resolved config.
        seed: The run seed.
        repo: Repository root, for the git state.
        components: Descriptions of the instantiated components (models, detector, ...).
        extra: Further entries merged into the manifest.

    Returns:
        The manifest as a dict.
    """
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "config_sha256": config_hash(config),
        "config": config,
        "code": git_state(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
        "components": components or {},
        **(extra or {}),
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Write a manifest as indented JSON (parent directories are created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
