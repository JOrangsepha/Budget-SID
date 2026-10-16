import json
import subprocess
import sys

import pytest

from budget_sid.experiment import command_evaluate_budgets

COMMANDS = [
    "train_recommender",
    "build_semantic_ids",
    "train_benefit_predictor",
    "evaluate_budgets",
    "demo",
]


@pytest.mark.parametrize("command", COMMANDS)
def test_smoke_cli(command, repo_root):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "budget_sid",
            command,
            "--config",
            "configs/lightweight_beauty.yaml",
            "--smoke",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == command
    assert payload["scripted"] is True
    assert payload["exit_code"] == 0
    if command == "build_semantic_ids":
        assert payload["code_shape"] == [12, 4]
        assert payload["embedding_dim"] == 384
        assert payload["duplicate_prefix_matched"] is True
        assert payload["duplicate_dedup_differs"] is True
    if command == "train_recommender":
        assert payload["d_model"] == 128
        assert payload["batch_size"] == 16
        assert payload["grad_accum"] == 4
        assert payload["strategy"] == "single_device"
    if command == "train_benefit_predictor":
        assert payload["splits_disjoint"] is True
        assert payload["test_locked"] is True
        assert payload["n_split_a"] > 0
        assert payload["n_split_b"] > 0
    if command == "evaluate_budgets":
        assert payload["sampled_negatives"] is False
        assert payload["stop_gate"]["exit_code"] == 0
        assert set(payload["test_metrics"]) == {
            "fixed_20",
            "fixed_40",
            "fixed_60",
            "random_budget",
            "entropy",
            "benefit_aware",
        }
        assert payload["test_metrics"]["fixed_20"]["recall@20"] == pytest.approx(0.5)
        assert payload["test_metrics"]["fixed_60"]["recall@20"] == pytest.approx(1.0)
        assert payload["test_metrics"]["fixed_60"]["slices"]["long_tail"]["recall@20"] == pytest.approx(1.0)
        assert payload["test_metrics"]["fixed_20"]["slices"]["long_tail"]["recall@20"] == pytest.approx(0.0)
        assert payload["test_metrics"]["fixed_40"]["max_recommendations"] <= 20
    if command == "demo":
        assert payload["policy"] == "benefit_aware"
        assert payload["beam_width"] in (20, 60)
        assert 1 <= len(payload["recommendations"]) <= 20
        assert "predicted_delta_ndcg" in payload


def test_cli_without_smoke_does_not_score_the_test_split(repo_root, beauty_config):
    result = subprocess.run(
        [sys.executable, "-m", "budget_sid", "evaluate_budgets", "--config", "configs/lightweight_beauty.yaml"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "gpu_path_not_run"
    assert "test_metrics" not in payload
    library = command_evaluate_budgets(beauty_config, smoke=False)
    assert library["exit_code"] == 1
    assert "test_metrics" not in library
