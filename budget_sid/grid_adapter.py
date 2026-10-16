"""Talk to a pinned GRID checkout without copying it into this repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

from budget_sid.config import repo_root
from budget_sid.constants import EMBEDDING_DIM, GRID_COMMIT, GRID_REPO_URL, TEXT_ENCODER
from budget_sid.errors import ConfigRejected

FORBIDDEN_OVERRIDE_MARKERS = ("flan-t5-xl", "strategy=ddp", "devices=-1")


def grid_checkout() -> Path:
    return repo_root() / "third_party" / "GRID"


def single_device_trainer_overrides() -> list[str]:
    """Replace the DDP block shipped in GRID's flat experiment configs."""
    return [
        "trainer.strategy=auto",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.sync_batchnorm=false",
        "trainer.accelerator=gpu",
    ]


def assert_overrides_allowed(overrides: list[str]) -> None:
    blob = " ".join(overrides).lower().replace(" ", "")
    for marker in FORBIDDEN_OVERRIDE_MARKERS:
        if marker.replace(" ", "") in blob:
            raise ConfigRejected(
                f"GRID override contains {marker!r}. "
                f"Embeddings stay on frozen {TEXT_ENCODER}; training stays on one device."
            )


def rkmeans_overrides() -> list[str]:
    """Overrides for ``rkmeans_train_flat`` / ``rkmeans_inference_flat`` only."""
    overrides = [
        f"embedding_dim={EMBEDDING_DIM}",
        "num_hierarchies=3",
        "codebook_width=256",
        "data_dir=data/amazon_data/beauty",
        *single_device_trainer_overrides(),
    ]
    assert_overrides_allowed(overrides)
    return overrides


def tiger_train_overrides() -> list[str]:
    """Overrides for ``tiger_train_flat``. Hierarchy 4 counts the dedup digit."""
    overrides = [
        "num_hierarchies=4",
        "data_dir=data/amazon_data/beauty",
        "model.huggingface_model.config.d_model=128",
        "model.huggingface_model.config.num_layers=4",
        "model.decoder.config.num_layers=4",
        "trainer.accumulate_grad_batches=4",
        "data_loading.train_dataloader_config.dataloader.batch_size_per_device=16",
        *single_device_trainer_overrides(),
    ]
    assert_overrides_allowed(overrides)
    return overrides


def tiger_inference_overrides() -> list[str]:
    """Inference keys that exist on ``tiger_inference_flat`` and not the trainer."""
    overrides = [
        "num_hierarchies=4",
        "data_dir=data/amazon_data/beauty",
        "model.huggingface_model.config.d_model=128",
        "model.huggingface_model.config.num_layers=4",
        "model.decoder.config.num_layers=4",
        "model.top_k_for_generation=20",
        *single_device_trainer_overrides(),
    ]
    assert_overrides_allowed(overrides)
    return overrides


def grid_commands(config: dict | None = None) -> list[list[str]]:
    """Commands the GPU path runs from the pinned checkout. Nothing is vendored."""
    del config
    return [
        ["python", "-m", "src.train", "experiment=rkmeans_train_flat", *rkmeans_overrides()],
        ["python", "-m", "src.inference", "experiment=rkmeans_inference_flat", *rkmeans_overrides()],
        ["python", "-m", "src.train", "experiment=tiger_train_flat", *tiger_train_overrides()],
        ["python", "-m", "src.inference", "experiment=tiger_inference_flat", *tiger_inference_overrides()],
    ]


def read_pinned_commit(checkout: Path | None = None) -> str | None:
    root = checkout or grid_checkout()
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def pin_status() -> dict:
    commit = read_pinned_commit()
    return {
        "repo": GRID_REPO_URL,
        "expected": GRID_COMMIT,
        "checkout": str(grid_checkout()),
        "actual": commit,
        "pinned": commit == GRID_COMMIT,
        "vendored": False,
    }
