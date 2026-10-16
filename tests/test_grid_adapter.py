import subprocess

import pytest

from budget_sid.constants import GRID_COMMIT, PUBLIC_GITHUB_REPO
from budget_sid.errors import ConfigRejected
from budget_sid.grid_adapter import (
    assert_overrides_allowed,
    rkmeans_overrides,
    tiger_inference_overrides,
    tiger_train_overrides,
)


def test_pin_script_matches_constant(repo_root):
    pin_script = (repo_root / "scripts" / "pin_grid.sh").read_text(encoding="utf-8")
    config = (repo_root / "configs" / "lightweight_beauty.yaml").read_text(encoding="utf-8")
    assert GRID_COMMIT in pin_script
    assert GRID_COMMIT in config
    subprocess.run(["bash", "-n", str(repo_root / "scripts" / "pin_grid.sh")], check=True)
    subprocess.run(["bash", "-n", str(repo_root / "scripts" / "push_github.sh")], check=True)


def test_overrides_are_single_device_and_minilm_width():
    rkmeans = rkmeans_overrides()
    tiger = tiger_train_overrides()
    inference = tiger_inference_overrides()
    assert "embedding_dim=384" in rkmeans
    assert "num_hierarchies=3" in rkmeans
    assert "trainer.devices=1" in rkmeans
    assert "trainer.strategy=auto" in rkmeans
    assert "num_hierarchies=4" in tiger
    assert "data_loading.train_dataloader_config.dataloader.batch_size_per_device=16" in tiger
    assert "trainer.accumulate_grad_batches=4" in tiger
    assert "model.top_k_for_generation=20" in inference
    for overrides in (rkmeans, tiger, inference):
        blob = " ".join(overrides)
        assert "flan-t5-xl" not in blob
        assert "strategy=ddp" not in blob
        assert "devices=-1" not in blob


def test_adapter_rejects_flan_and_ddp():
    with pytest.raises(ConfigRejected):
        assert_overrides_allowed(["embedding_model=google/flan-t5-xl"])
    with pytest.raises(ConfigRejected):
        assert_overrides_allowed(["trainer.strategy=ddp"])
    with pytest.raises(ConfigRejected):
        assert_overrides_allowed(["trainer.devices=-1"])


def test_push_script_names_public_repo(repo_root):
    script = (repo_root / "scripts" / "push_github.sh").read_text(encoding="utf-8")
    assert PUBLIC_GITHUB_REPO in script
    assert "third_party/GRID" in script
