"""CPU checks for the leak-safe first-step dump. No GPU and no invented logits."""

import json
import os
import stat
import subprocess
import sys

import numpy as np
import pytest

from budget_sid.errors import MissingFirstStepLogits
from budget_sid.eval_samples import synthetic_samples, write_jsonl
from budget_sid.features import FEATURE_NAMES, extract_features
from budget_sid.first_step_dump import (
    assert_leak_safe_manifest,
    assert_same_request_ids,
    command_dump_first_step_features,
    feature_rows_from_keyed_logits,
    require_logit_vector,
)
from budget_sid.grid_first_step import (
    HYDRA_DATA_FOLDER,
    HYDRA_TIMEOUT,
    HYDRA_WORKERS,
    first_step_candidate_logits,
    hydra_overrides,
    predict_keys_from_batch,
    rows_from_batches,
)
from budget_sid.leak_safe_predict import history_only_rows, write_predict_dataset
from budget_sid.replay import command_replay_budgets
from budget_sid.synthetic import concentrated_tail_population


class _Batch:
    def __init__(self, keys, width=4):
        self.user_id_list = list(keys)
        self.mask = np.ones((len(keys), 2))
        self.transformed_sequences = {
            "sequence_data": np.zeros((len(keys), 2)),
            "user_id": np.asarray(list(keys)),
        }


class _MLP:
    def __init__(self, row):
        self.row = np.asarray(row, dtype=np.float64)

    def __call__(self, hidden):
        return np.tile(self.row, (hidden.shape[0], 1))


class _Decoder:
    def __init__(self, row):
        self.decoder_mlp = [_MLP(row)]


class _Module:
    def __init__(self, row, *, prefix=False):
        self.decoder = _Decoder(row)
        self.should_check_prefix = prefix
        self.num_embeddings_per_hierarchy = len(row)
        self.calls = []

    def encoder_forward_pass(self, attention_mask, input_ids, user_id):
        self.calls.append("encoder")
        batch = int(np.asarray(input_ids).shape[0])
        return np.zeros((batch, 2, 3)), np.ones((batch, 2))

    def decoder_forward_pass(self, **kwargs):
        assert kwargs["future_ids"] is None
        self.calls.append("decoder")
        batch = int(np.asarray(kwargs["encoder_output"]).shape[0])
        return np.zeros((batch, 1, 3)), None

    def _check_valid_prefix(self, prefix):
        valid = np.zeros(int(np.asarray(prefix).shape[0]), dtype=bool)
        valid[0] = True
        return valid


def _index():
    return [
        {"predict_key": 0, "request_id": "val:a:2", "n_history": 2, "user_id": "a", "split": "val"},
        {"predict_key": 1, "request_id": "val:b:4", "n_history": 4, "user_id": "b", "split": "val"},
    ]


def test_missing_logits_are_an_error_and_flat_zeros_are_not_invented():
    with pytest.raises(MissingFirstStepLogits, match="invent zeros"):
        require_logit_vector(None)
    with pytest.raises(MissingFirstStepLogits, match="invent zeros"):
        require_logit_vector([np.nan, 1.0])
    with pytest.raises(MissingFirstStepLogits, match="invent zeros"):
        feature_rows_from_keyed_logits(_index(), {0: [1.0, 0.0]}, logits_source="test_mock")
    with pytest.raises(MissingFirstStepLogits, match="not in the leak-safe index"):
        feature_rows_from_keyed_logits(
            _index(),
            {0: [1.0, 0.0], 1: [0.0, 1.0], 9: [1.0, 0.0]},
            logits_source="test_mock",
        )
    uniform = extract_features(2, [0.0, 0.0, 0.0, 0.0])
    assert uniform["first_step_entropy"] == pytest.approx(np.log(4))
    rows = feature_rows_from_keyed_logits(
        _index(),
        {0: [0.0, 0.0, 0.0, 0.0], 1: [4.0, 0.0, 0.0, 0.0]},
        logits_source="test_mock",
    )
    assert rows[0]["first_step_entropy"] == pytest.approx(np.log(4))
    assert rows[1]["top1_prob"] > rows[0]["top1_prob"]
    assert rows[0]["history_length"] == 2.0
    assert rows[1]["history_length"] == 4.0
    assert rows[0]["raw_logits_stored"] is False
    assert rows[0]["includes_target_event"] is False


def test_dataloader_order_does_not_reorder_request_ids():
    module = _Module([3.0, 0.0, 0.0, 0.0])
    rows = rows_from_batches(
        module,
        [_Batch([1]), _Batch([0])],
        _index(),
    )
    assert [row["request_id"] for row in rows] == ["val:a:2", "val:b:4"]
    assert [row["predict_key"] for row in rows] == [0, 1]
    assert module.calls[0] == "encoder"
    expected = extract_features(2, [3.0, 0.0, 0.0, 0.0])
    assert rows[0]["first_step_entropy"] == pytest.approx(expected["first_step_entropy"])
    assert rows[0]["candidate_prob_gap"] == pytest.approx(expected["candidate_prob_gap"])
    assert rows[0]["top1_prob"] == pytest.approx(expected["top1_prob"])
    with pytest.raises(MissingFirstStepLogits, match="batch position"):
        predict_keys_from_batch(object())


def test_hook_fails_closed_when_the_grid_tensor_is_missing():
    class _Empty:
        pass

    bare = _Empty()
    with pytest.raises(MissingFirstStepLogits, match="invent"):
        first_step_candidate_logits(bare, attention_mask=np.ones((1, 1)), input_ids=np.zeros((1, 1)), user_id=None)

    class _NoMLP(_Module):
        def __init__(self):
            self.decoder = _Empty()
            self.should_check_prefix = False

    with pytest.raises(MissingFirstStepLogits, match="decoder.decoder_mlp"):
        first_step_candidate_logits(
            _NoMLP(),
            attention_mask=np.ones((1, 1)),
            input_ids=np.zeros((1, 1)),
            user_id=None,
        )

    class _NoneMLP(_MLP):
        def __call__(self, hidden):
            return None

    module = _Module([1.0, 0.0, 0.0, 0.0])
    module.decoder.decoder_mlp = [_NoneMLP([0.0])]
    with pytest.raises(MissingFirstStepLogits, match="invent zeros"):
        first_step_candidate_logits(
            module,
            attention_mask=np.ones((1, 2)),
            input_ids=np.zeros((1, 2)),
            user_id=None,
        )
    with pytest.raises(MissingFirstStepLogits, match="user_id_list"):
        rows_from_batches(module, [object()], _index())

    class _OneRow(_MLP):
        def __call__(self, hidden):
            return np.asarray([[1.0, 0.0]])

    short = _Module([1.0, 0.0])
    short.decoder.decoder_mlp = [_OneRow([1.0, 0.0])]
    with pytest.raises(MissingFirstStepLogits, match="zip"):
        rows_from_batches(short, [_Batch([0, 1])], _index())


def test_prefix_mask_uses_negative_inf_and_not_zeros():
    module = _Module([1.0, 3.0, 0.0, 0.0], prefix=True)
    matrix = first_step_candidate_logits(
        module,
        attention_mask=np.ones((1, 2)),
        input_ids=np.zeros((1, 2)),
        user_id=None,
    )
    assert matrix[0, 0] == pytest.approx(1.0)
    assert np.isneginf(matrix[0, 1])
    assert matrix[0, 1] != 0.0
    features = extract_features(2, matrix[0])
    assert features["top1_prob"] == pytest.approx(1.0)


def test_manifest_refuses_a_target_in_history_folder(tmp_path):
    samples = synthetic_samples()
    written = write_predict_dataset(tmp_path, history_only_rows(samples), sid_checked=False)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_leak_safe_manifest(manifest)
    manifest["includes_target_event"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refused = command_dump_first_step_features(
        smoke=False,
        predict_dir=str(tmp_path),
        ckpt=str(tmp_path / "missing.ckpt"),
        dry_run=True,
    )
    assert refused["exit_code"] == 1
    assert refused["invented_logits"] is False
    assert "target event" in refused["message"]
    assert "tfrecord" in written


def test_dry_run_checks_the_checkpoint_and_does_not_write_logits(tmp_path):
    samples = synthetic_samples()
    write_predict_dataset(tmp_path / "predict", history_only_rows(samples))
    missing = command_dump_first_step_features(
        smoke=False,
        predict_dir=str(tmp_path / "predict"),
        ckpt=str(tmp_path / "no-such.ckpt"),
        dry_run=True,
    )
    assert missing["exit_code"] == 1
    assert "Checkpoint not found" in missing["message"]
    assert "invent" in missing["message"]
    ckpt = tmp_path / "checkpoint_epoch=000_step=018800.ckpt"
    ckpt.write_text("not-a-real-checkpoint", encoding="utf-8")
    payload = command_dump_first_step_features(
        smoke=False,
        predict_dir=str(tmp_path / "predict"),
        ckpt=str(ckpt),
        semantic_id_path=str(tmp_path / "sid.pt"),
        data_dir=str(tmp_path / "beauty"),
        grid_dir=str(tmp_path / "GRID"),
        output=str(tmp_path / "first_step_features.jsonl"),
        dry_run=True,
    )
    assert payload["exit_code"] == 0, payload
    assert payload["ckpt_loaded"] is False
    assert payload["invented_logits"] is False
    assert payload["grid_first_step"] is False
    assert payload["p3_started"] is False
    assert not (tmp_path / "first_step_features.jsonl").exists()
    overrides = " ".join(payload["hydra_overrides"])
    assert "ckpt_path='" in overrides
    assert HYDRA_WORKERS in overrides
    assert HYDRA_TIMEOUT in overrides
    assert f"{HYDRA_DATA_FOLDER}=" in overrides
    assert "test_dataloader_config" not in overrides
    bare = command_dump_first_step_features(smoke=False)
    assert bare["exit_code"] == 1
    assert "invent" in bare["message"]


def test_hydra_overrides_quote_the_checkpoint_and_keep_the_history_only_folder():
    overrides = hydra_overrides(
        data_dir="/data/beauty",
        sid_path="/data/semantic_ids.pt",
        ckpt="/logs/checkpoint_epoch=000_step=018800.ckpt",
        predict_dir="/outputs/leak_safe_beams/predict",
    )
    text = " ".join(overrides)
    assert "ckpt_path='/logs/checkpoint_epoch=000_step=018800.ckpt'" in overrides
    assert HYDRA_WORKERS in overrides
    assert HYDRA_TIMEOUT in overrides
    assert f"{HYDRA_DATA_FOLDER}=/outputs/leak_safe_beams/predict" in overrides
    assert "experiment=tiger_inference_flat" in overrides
    assert "test_dataloader_config" not in text
    assert "testing" not in text
    with pytest.raises(MissingFirstStepLogits, match="="):
        hydra_overrides(
            data_dir="/data/beauty",
            sid_path="/data/semantic_ids.pt",
            ckpt="/logs/checkpoint_epoch=000_step=018800.ckpt",
            predict_dir="/outputs/leak_safe_beams/predict=bad",
        )


def test_feature_file_joins_replay_on_request_id(tmp_path):
    population = concentrated_tail_population(n_test=0)
    index = []
    logits = {}
    for position, row in enumerate(population):
        index.append(
            {
                "predict_key": position,
                "request_id": row["request_id"],
                "n_history": len(row["history"]),
                "user_id": row["user_id"],
                "split": row["split"],
            }
        )
        peaked = row["user_id"].startswith("positive")
        logits[position] = [0.0, 0.0, 0.0, 4.0] if peaked else [5.0, 0.0, 0.0, 0.0]
        row.pop("features")
    feature_rows = feature_rows_from_keyed_logits(index, logits, logits_source="test_mock")
    assert_same_request_ids(feature_rows, population, other_name="per_request.jsonl")
    with pytest.raises(ValueError, match="request_id order"):
        assert_same_request_ids(feature_rows, list(reversed(population)), other_name="per_request.jsonl")
    features_path = tmp_path / "first_step_features.jsonl"
    per_request_path = tmp_path / "per_request.jsonl"
    write_jsonl(features_path, feature_rows)
    write_jsonl(per_request_path, population)
    summary = {
        "ranking_source": "leak_safe_history",
        "auditable_for_delta_ndcg": True,
        "benefit_aware_claimed": False,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    joined = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(per_request_path),
        features=str(features_path),
    )
    assert joined["exit_code"] == 0, joined
    assert joined["features"]["full_p2_ready"] is True
    assert joined["features"]["feature_names"] == list(FEATURE_NAMES)
    assert joined["features"]["invented_logits"] is False
    assert joined["invented_logits"] is False
    assert joined["policies"]["entropy"]["status"] == "ran"
    assert joined["policies"]["benefit_aware"]["status"] == "ran"
    assert joined["latency_claimed"] is False
    assert joined["grid_recommender_started"] is False
    first = json.loads(features_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["request_id"] == population[0]["request_id"]
    assert set(FEATURE_NAMES).issubset(first)


def test_smoke_cli_does_not_claim_a_grid_dump():
    payload = command_dump_first_step_features(smoke=True)
    assert payload["exit_code"] == 0, payload
    assert payload["scripted"] is True
    assert payload["logits_source"] == "scripted_cpu"
    assert payload["grid_first_step"] is False
    assert payload["invented_logits"] is False
    assert payload["includes_target_event"] is False
    assert payload["p3_started"] is False
    assert payload["grid_pin_changed"] is False
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert payload["n"] == len(synthetic_samples())


def test_dump_script_dry_run_and_missing_checkpoint(tmp_path, repo_root):
    samples = synthetic_samples()
    sample_path = tmp_path / "eval_samples.jsonl"
    write_jsonl(sample_path, samples)
    sid_path = tmp_path / "semantic_ids.jsonl"
    mapping_lines = []
    items = sorted({item for sample in samples for item in [*sample["history"], sample["target"]]})
    for index, item in enumerate(items):
        mapping_lines.append(json.dumps({"item_id": item, "semantic_id": [index, 0, 0, 0]}))
    sid_path.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")
    ckpt = tmp_path / "checkpoint_epoch=000_step=018800.ckpt"
    ckpt.write_text("not-a-real-checkpoint", encoding="utf-8")
    out_dir = tmp_path / "leak_safe_beams"
    script = repo_root / "scripts" / "dump_first_step_features.sh"
    assert script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "ROOT_DIR": str(repo_root),
        "SAMPLES": str(sample_path),
        "SID_PATH": str(sid_path),
        "OUT_DIR": str(out_dir),
        "CKPT": str(ckpt),
        "DATA_DIR": str(tmp_path / "beauty"),
        "GRID_DIR": str(tmp_path / "GRID"),
        "DRY_RUN": "1",
    }
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "ckpt_path='" in result.stdout
    assert HYDRA_WORKERS in result.stdout
    assert "SKIP_SHUTDOWN" in result.stdout
    assert "poweroff" in result.stdout
    assert "replay_budgets" in result.stdout
    assert "--features" in result.stdout
    assert "P3 stays gated" in result.stdout
    assert not (out_dir / "first_step_features.jsonl").exists()
    assert (out_dir / "predict" / "manifest.json").is_file()
    manifest = json.loads((out_dir / "predict" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["includes_target_event"] is False

    missing = subprocess.run(
        ["bash", str(script)],
        env={**env, "CKPT": str(tmp_path / "absent.ckpt"), "SKIP_BUILD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1
    assert "invent" in missing.stderr
    assert "SKIP_SHUTDOWN" in missing.stdout

    refused = subprocess.run(
        ["bash", str(script)],
        env={**env, "LEGACY_BEAM_DIR": str(tmp_path / "old")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1
    assert "not leak-safe" in refused.stderr
    assert "SKIP_SHUTDOWN" in refused.stdout
