"""Shared pytest fixtures: the repository root, the prompt library and a loader of backend configs."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root directory of the repository."""
    return REPO


@pytest.fixture(scope="session")
def prompt_lib():
    """The repository's prompt library (``prompts/``)."""
    from kznav.language.prompts import PromptLibrary

    return PromptLibrary(REPO / "prompts")


@pytest.fixture
def backend_spec():
    """Load a backend config YAML as a plain dict (without Hydra)."""
    import yaml

    def load(name: str) -> dict:
        return yaml.safe_load((REPO / "configs" / "backend" / f"{name}.yaml").read_text(encoding="utf-8"))

    return load
