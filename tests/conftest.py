from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def beauty_config():
    from budget_sid.config import load_config

    return load_config(ROOT / "configs" / "lightweight_beauty.yaml")
