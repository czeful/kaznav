"""Compose the Hydra config outside of a Hydra app (tools, tests, notebooks)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]


def compose_config(overrides: Sequence[str] = (), config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Resolved config as a plain dict (interpolations evaluated)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(config_dir or REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
