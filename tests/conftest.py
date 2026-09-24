import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def prompt_lib():
    from kznav.language.prompts import PromptLibrary

    return PromptLibrary(REPO / "prompts")


@pytest.fixture
def backend_spec():
    """Load a backend config YAML as a plain dict (without Hydra)."""
    import yaml

    def load(name: str) -> dict:
        return yaml.safe_load((REPO / "configs" / "backend" / f"{name}.yaml").read_text(encoding="utf-8"))

    return load
