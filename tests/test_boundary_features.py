"""CPU checks for the hierarchy-0 logits cache and the beam-20/60 boundary cut."""

import json
import os
import stat
import subprocess
import sys

import numpy as np
import pytest

from budget_sid.errors import MissingFirstStepLogits
from budget_sid.eval_samples import synthetic_samples, write_jsonl
from budget_sid.features import (
    ROUTING_FEATURE_NAMES,
    extract_boundary_features,
    ranked_probabilities,
)
from budget_sid.first_step_logits import (
    assert_cached_request_coverage,
    boundary_rows_from_cache,
    command_dump_first_step_logits,
    command_extract_boundary_features,
    load_logits_cache,
    select_split_rows,
    stack_logit_rows,
    store_logits,
    take_logits_for_index,
    write_logits_cache,
)
from budget_sid.grid_first_step import _load_state_dict
from budget_sid.leak_safe_predict import history_only_rows, write_predict_dataset
from budget_sid.replay import command_replay_budgets


def _known_distribution():
    probs = np.zeros(60, dtype=np.float64)
    probs[:20] = 0.04
    probs[20] = 0.01
    probs[21:] = 0.19 / 39
    assert probs.sum() == pytest.approx(1.0)
    return np.log(probs)


def _index_row(key, request_id, split, n_history=3):
    return {
        "predict_key": key,
        "request_id": request_id,
        "user_id": request_id.split(":")[1],
        "split": split,
        "n_history": n_history,
    }


def test_boundary_math_uses_ranks_not_a_padded_short_vector():
    logits = _known_distribution()
    ranked = ranked_probabilities(logits)
    assert ranked[0] == pytest.approx(0.04)
    assert ranked[19] == pytest.approx(0.04)
    assert ranked[20] == pytest.approx(0.01)
    features = extract_boundary_features(4, logits)
    assert features["history_length"] == 4.0
    assert features["top1_prob"] == pytest.approx(0.04)
    assert features["top20_cum_prob"] == pytest.approx(0.80)
    assert features["prob_gap_p20_p21"] == pytest.approx(0.03)
    assert features["prob_sum_ranks_21_60"] == pytest.approx(0.20)
    assert set(ROUTING_FEATURE_NAMES) <= set(features)
    with pytest.raises(ValueError, match="pad"):
        extract_boundary_features(4, np.log(np.full(59, 1.0 / 59)))


def test_ties_keep_logit_order_and_invalid_codes_carry_no_mass():
    tied = np.full(60, 1.0 / 60)
    features = extract_boundary_features(1, np.log(tied))
    assert features["prob_gap_p20_p21"] == pytest.approx(0.0)
    assert features["top20_cum_prob"] == pytest.approx(20.0 / 60.0)
    assert features["prob_sum_ranks_21_60"] == pytest.approx(40.0 / 60.0)
    masked = np.full(60, -np.inf)
    masked[:20] = 0.0
    masked_features = extract_boundary_features(2, masked)
    assert masked_features["top20_cum_prob"] == pytest.approx(1.0)
    assert masked_features["prob_gap_p20_p21"] == pytest.approx(0.05)
    assert masked_features["prob_sum_ranks_21_60"] == pytest.approx(0.0)


def test_split_filter_keeps_the_original_predict_key():
    index = [
        _index_row(0, "train:a:1", "train", 1),
        _index_row(4, "val:b:2", "evaluation", 2),
        _index_row(9, "test:c:3", "test", 3),
    ]
    selected = select_split_rows(index, "val")
    assert [row["predict_key"] for row in selected] == [4]
    assert selected[0]["split"] == "val"
    assert selected[0]["request_id"] == "val:b:2"
    with pytest.raises(ValueError, match="no split"):
        select_split_rows([{"predict_key": 0, "request_id": "x", "n_history": 1}], "val")
    with pytest.raises(ValueError, match="empty"):
        select_split_rows([_index_row(0, "train:a:1", "train")], "val")


def test_full_then_filter_drops_other_splits_and_refuses_a_missing_val_key():
    selected = [_index_row(4, "val:b:2", "val", 2)]
    logits = {
        0: np.arange(60, dtype=np.float64),
        4: np.arange(60, dtype=np.float64) + 1.0,
        9: np.arange(60, dtype=np.float64) + 2.0,
    }
    kept = take_logits_for_index(selected, logits, allow_extra=True)
    assert list(kept) == [4]
    with pytest.raises(MissingFirstStepLogits, match="outside the logits selection"):
        take_logits_for_index(selected, logits, allow_extra=False)
    with pytest.raises(MissingFirstStepLogits, match="no first-step logits"):
        take_logits_for_index(selected, {0: logits[0]}, allow_extra=True)
    with pytest.raises(MissingFirstStepLogits, match="pad"):
        stack_logit_rows(selected, {4: np.zeros(59)})


def test_nan_and_overflow_do_not_become_a_stored_score(tmp_path):
    row = _index_row(4, "val:b:2", "val", 2)
    with pytest.raises(MissingFirstStepLogits, match="NaN"):
        write_logits_cache(
            tmp_path / "first_step_logits_val.npz",
            [row],
            {4: np.full(60, np.nan)},
            split="val",
            logits_pass="test",
            logits_source="test_mock",
        )
    assert not (tmp_path / "first_step_logits_val.npz").exists()
    wide = np.zeros((1, 60), dtype=np.float64)
    wide[0, 0] = 1.0e6
    stored, dtype_name = store_logits(wide)
    assert dtype_name == "float32"
    assert np.isfinite(stored).all()
    exact = np.arange(60, dtype=np.float64).reshape(1, 60)
    stored16, dtype16 = store_logits(exact)
    assert dtype16 == "float16"
    assert np.array_equal(stored16.astype(np.float64), exact)


def test_cache_roundtrip_joins_replay_on_request_id(tmp_path):
    users = ["aa", "bb", "cc", "dd"]
    index = []
    logits = {}
    per_request = []
    for position, user in enumerate(users):
        key = 10 + position
        request_id = f"val:{user}:2"
        index.append(_index_row(key, request_id, "val", 2))
        vector = np.linspace(-1.0, 3.0, 64, dtype=np.float64)
        vector[position] += 5.0
        logits[key] = vector
        ndcg20 = 0.1 * position
        ndcg60 = 0.8 if position < 2 else ndcg20
        per_request.append(
            {
                "request_id": request_id,
                "user_id": user,
                "split": "val",
                "target_position": 2,
                "beams": {
                    "20": {"ndcg@20": ndcg20},
                    "40": {"ndcg@20": ndcg20},
                    "60": {"ndcg@20": ndcg60},
                },
                "delta_ndcg@20": ndcg60 - ndcg20,
                "target": 99,
            }
        )
    cache_path = tmp_path / "first_step_logits_val.npz"
    written = write_logits_cache(
        cache_path,
        index,
        logits,
        split="val",
        logits_pass="val_only",
        logits_source="test_mock",
    )
    loaded = load_logits_cache(written["logits"])
    assert loaded["logits_dtype"] == "float16"
    assert [row["predict_key"] for row in loaded["rows"]] == [10, 11, 12, 13]
    assert loaded["manifest"]["join"]["request_id"].startswith("replay_budgets")
    assert loaded["manifest"]["targets_are_routing_features"] is False
    assert "top-20" in loaded["manifest"]["attribution_note"]
    rows = boundary_rows_from_cache(loaded, split="val")
    assert [row["request_id"] for row in rows] == [row["request_id"] for row in per_request]
    for position, (row, source) in enumerate(zip(rows, index)):
        assert set(ROUTING_FEATURE_NAMES) <= set(row)
        assert "target" not in row
        assert "target_in_top20" not in row
        original = logits[source["predict_key"]]
        cached = loaded["logits"][position]
        assert np.allclose(cached, original.astype(np.float16).astype(np.float64))
        from_cache = extract_boundary_features(2, cached)
        from_original = extract_boundary_features(2, original)
        for name in (*ROUTING_FEATURE_NAMES,):
            assert row[name] == pytest.approx(from_cache[name])
            assert row[name] == pytest.approx(from_original[name], rel=1e-3, abs=5e-4)
    features_path = tmp_path / "boundary_features.jsonl"
    per_path = tmp_path / "per_request.jsonl"
    write_jsonl(features_path, rows)
    poisoned = dict(rows[0])
    poisoned["target"] = 99
    poisoned["target_in_top20"] = 1
    write_jsonl(features_path, [poisoned, *rows[1:]])
    write_jsonl(per_path, per_request)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "ranking_source": "leak_safe_history",
                "auditable_for_delta_ndcg": True,
                "benefit_aware_claimed": False,
            }
        ),
        encoding="utf-8",
    )
    joined = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(per_path),
        features=str(features_path),
        fractions="0,0.5",
        seeds="42",
    )
    assert joined["exit_code"] == 0, joined
    assert joined["features"]["full_p2_ready"] is True
    assert joined["features"]["boundary_features_ready"] is True
    assert joined["boundary_features_dumped"] is True
    assert joined["features"]["feature_names"] == list(ROUTING_FEATURE_NAMES)
    assert joined["features"]["invented_logits"] is False
    assert "target" in joined["features"]["dropped_label_features"]
    assert "target_in_top20" not in joined["features"]["feature_names"]
    assert joined["predictor"]["hist_gradient_boosting"]["boundary_features"] is True
    assert joined["predictor"]["ridge"]["boundary_features"] is True
    assert {point["policies"]["entropy"]["status"] for point in joined["curve"]} == {"ran"}
    assert {point["policies"]["benefit_aware"]["status"] for point in joined["curve"]} == {"ran"}
    assert joined["p3_started"] is False
    assert joined["grid_pin_changed"] is False
    assert joined["invented_logits"] is False


def test_extract_command_filters_a_full_cache_down_to_val(tmp_path):
    index = [
        _index_row(0, "train:a:1", "train", 1),
        _index_row(1, "val:b:2", "val", 2),
    ]
    logits = {
        0: np.arange(60, dtype=np.float64),
        1: np.arange(60, dtype=np.float64) + 2.0,
    }
    cache_path = tmp_path / "first_step_logits.npz"
    write_logits_cache(
        cache_path,
        index,
        logits,
        split="all",
        logits_pass="all",
        logits_source="test_mock",
    )
    features_path = tmp_path / "boundary_features.jsonl"
    payload = command_extract_boundary_features(
        logits=str(cache_path),
        output=str(features_path),
        split="val",
    )
    assert payload["exit_code"] == 0, payload
    assert payload["n"] == 1
    assert payload["grid_first_step"] is False
    assert payload["p3_started"] is False
    written = [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["request_id"] == "val:b:2"
    assert written[0]["predict_key"] == 1
    dry = command_extract_boundary_features(logits=str(cache_path), split="val", dry_run=True)
    assert dry["exit_code"] == 0, dry
    assert dry["dry_run"] is True
    missing = command_extract_boundary_features(logits=str(tmp_path / "absent.npz"))
    assert missing["exit_code"] == 1
    assert "invent" in missing["message"]


def test_coverage_check_refuses_a_val_request_left_out_of_the_cache():
    cached = [_index_row(1, "val:b:2", "val")]
    other = [
        {"request_id": "val:b:2", "split": "val"},
        {"request_id": "val:c:2", "split": "val"},
    ]
    with pytest.raises(ValueError, match="missing"):
        assert_cached_request_coverage(cached, other, split="val", other_name="per_request.jsonl")
    with pytest.raises(ValueError, match="not in"):
        assert_cached_request_coverage(
            [_index_row(3, "val:z:2", "val")],
            other,
            split="val",
            other_name="per_request.jsonl",
        )


def test_checkpoint_load_is_strict(tmp_path, monkeypatch):
    calls = {}

    class _Torch:
        @staticmethod
        def load(path, map_location="cpu", weights_only=False):
            return {"state_dict": {"weight": 1}}

    class _Model:
        def load_state_dict(self, state, strict=False):
            calls["strict"] = strict
            calls["state"] = state
            raise RuntimeError("unexpected key")

    monkeypatch.setitem(sys.modules, "torch", _Torch)
    ckpt = tmp_path / "checkpoint.ckpt"
    ckpt.write_bytes(b"unused")
    with pytest.raises(MissingFirstStepLogits, match="partial load"):
        _load_state_dict(_Model(), str(ckpt))
    assert calls["strict"] is True


def test_dry_run_does_not_write_a_cache_or_load_the_model(tmp_path):
    samples = synthetic_samples()
    predict_dir = tmp_path / "predict"
    write_predict_dataset(predict_dir, history_only_rows(samples))
    ckpt = tmp_path / "checkpoint_epoch=000_step=018800.ckpt"
    ckpt.write_text("not-a-real-checkpoint", encoding="utf-8")
    output = tmp_path / "first_step_logits_val.npz"
    payload = command_dump_first_step_logits(
        predict_dir=str(predict_dir),
        output=str(output),
        ckpt=str(ckpt),
        semantic_id_path=str(tmp_path / "sid.pt"),
        data_dir=str(tmp_path / "beauty"),
        grid_dir=str(tmp_path / "GRID"),
        split="val",
        dry_run=True,
    )
    assert payload["exit_code"] == 0, payload
    assert payload["dry_run"] is True
    assert payload["logits_pass"] == "val_only"
    assert payload["n"] == 1
    assert payload["ckpt_loaded"] is False
    assert payload["grid_pin_changed"] is False
    assert payload["p3_started"] is False
    assert not output.exists()
    assert not (tmp_path / "predict_val").exists()
    assert "ckpt_path='" in " ".join(payload["hydra_overrides"])
    missing = command_dump_first_step_logits(
        predict_dir=str(predict_dir),
        ckpt=str(tmp_path / "absent.ckpt"),
        split="val",
        dry_run=True,
    )
    assert missing["exit_code"] == 1
    assert "invent" in missing["message"]


def test_dump_logits_script_dry_run_and_missing_checkpoint(tmp_path, repo_root):
    samples = synthetic_samples()
    sample_path = tmp_path / "eval_samples.jsonl"
    write_jsonl(sample_path, samples)
    sid_path = tmp_path / "semantic_ids.jsonl"
    items = sorted({item for sample in samples for item in [*sample["history"], sample["target"]]})
    mapping_lines = [
        json.dumps({"item_id": item, "semantic_id": [index, 0, 0, 0]})
        for index, item in enumerate(items)
    ]
    sid_path.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")
    ckpt = tmp_path / "checkpoint_epoch=000_step=018800.ckpt"
    ckpt.write_text("not-a-real-checkpoint", encoding="utf-8")
    out_dir = tmp_path / "leak_safe_beams"
    script = repo_root / "scripts" / "dump_first_step_logits.sh"
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
    assert "num_workers=0" in result.stdout
    assert "timeout=0" in result.stdout
    assert "SKIP_SHUTDOWN" in result.stdout
    assert "poweroff" in result.stdout
    assert "replay_budgets" in result.stdout
    assert "--features" in result.stdout
    assert "boundary_features.jsonl" in result.stdout
    assert "P3 stays gated" in result.stdout
    assert "request_id" in result.stdout
    assert "predict_key" in result.stdout
    assert not (out_dir / "first_step_logits_val.npz").exists()
    assert not (out_dir / "boundary_features.jsonl").exists()
    assert not (out_dir / "predict_val").exists()
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

    extract_only = subprocess.run(
        ["bash", str(script)],
        env={**env, "EXTRACT_ONLY": "1", "DRY_RUN": "0", "SKIP_BUILD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert extract_only.returncode == 1
    assert "invent" in extract_only.stderr
    assert "SKIP_SHUTDOWN" in extract_only.stdout
