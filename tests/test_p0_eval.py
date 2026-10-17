"""P0 exporter, leakage audit, and per-request NDCG@20."""

import json
import math
import pickle
import stat
import subprocess
import sys

import pytest

from budget_sid.beam_score import (
    codes_from_sid_matrix,
    command_score_beam_ndcg,
    load_rankings,
    load_sid_map,
    score_beams,
    score_ranking,
    synthetic_rankings,
    write_score_outputs,
)
from budget_sid.benefit import train_on_split_a
from budget_sid.data_audit import assert_target_event_not_in_history
from budget_sid.errors import HistoryLeakError, TestLeakError
from budget_sid.eval_audit import audit_samples, command_audit_eval_leakage
from budget_sid.eval_samples import (
    build_eval_sample,
    command_export_eval_samples,
    load_sequence_rows,
    read_jsonl,
    synthetic_samples,
    write_samples,
)
from budget_sid.grid_collate import describe_collate_inputs
from budget_sid.metrics import ndcg_at_k
from budget_sid.splits import assert_test_unused, describe_split_users
from budget_sid.synthetic import make_population

HASHES = {"checkpoint_hash": "ckpt", "semantic_id_hash": "sid"}


def test_repeated_purchase_is_kept_and_target_event_is_not():
    sample = build_eval_sample(
        user_id="a",
        split="testing",
        sequence=[1, 7, 2, 3, 4, 7],
        **HASHES,
    )
    assert sample["split"] == "test"
    assert sample["history"] == [1, 7, 2, 3, 4]
    assert sample["target"] == 7
    assert sample["target_position"] == 5
    assert sample["repeated_target_item"] is True
    assert sample["request_id"] == "test:a:5"
    assert 5 not in sample["history_positions"]
    assert_target_event_not_in_history(sample["history_positions"], sample["target_position"])
    required = {
        "request_id",
        "user_id",
        "split",
        "history",
        "target",
        "target_position",
        "checkpoint_hash",
        "semantic_id_hash",
    }
    assert required <= set(sample)


def test_regression_target_event_left_in_history():
    sequence = [10, 20, 30, 40]
    with pytest.raises(HistoryLeakError, match="target event"):
        build_eval_sample(
            user_id="u",
            split="test",
            history=sequence,
            target=40,
            target_position=3,
            **HASHES,
        )
    with pytest.raises(HistoryLeakError, match="target event"):
        build_eval_sample(
            user_id="u",
            split="test",
            sequence=sequence,
            history=sequence,
            target=40,
            target_position=3,
            **HASHES,
        )


def test_tampered_record_fails_the_audit_even_if_positions_were_not_updated():
    sample = build_eval_sample(user_id="u", split="test", sequence=[10, 20, 30, 40], **HASHES)
    sample["history"] = [10, 20, 30, 40]
    sample["history_positions"] = [0, 1, 2]
    report = audit_samples([sample])
    assert report["ok"] is False
    assert report["n_target_event_in_history"] == 1
    assert "target event" in report["leak_failures"][0]["error"]


def test_empty_history_is_allowed():
    sample = build_eval_sample(user_id="u", split="val", sequence=[4], **HASHES)
    assert sample["history"] == []
    assert sample["target"] == 4
    assert sample["target_position"] == 0


def test_inference_collate_keeps_the_target_train_mask_does_not():
    short = describe_collate_inputs(3)
    long = describe_collate_inputs(40)
    assert short["train_input_includes_target_event"] is False
    assert short["inference_input_includes_target_event"] is True
    assert short["reported_at5_at10_uses_masked_target"] is True
    assert "nDCG@5" in short["note"]
    assert long["inference_input_includes_target_event"] is True
    assert long["train_input_includes_target_event"] is False
    assert long["sequence_length_tokens"] == 120


def test_shared_users_are_not_label_leakage_and_disjoint_helper_still_raises():
    report = describe_split_users(
        [
            {"user_id": "a", "split": "train"},
            {"user_id": "a", "split": "val"},
            {"user_id": "a", "split": "test"},
        ]
    )
    assert report["user_overlap_is_label_leakage"] is False
    assert report["n_shared_val_test"] == 1
    assert report["n_shared_all_three"] == 1
    with pytest.raises(TestLeakError):
        assert_test_unused(["a"], ["a"])


def test_temporal_loo_allows_shared_users_and_rejects_reused_events():
    rows = []
    for row in make_population():
        copied = dict(row)
        copied["request_id"] = f"{row['split']}:{row['user_id']}:0"
        rows.append(copied)
    shared_user = rows[0]["user_id"]
    rows.append(
        {
            **rows[0],
            "split": "test",
            "request_id": f"test:{shared_user}:9",
            "target": 999,
        }
    )
    trained = train_on_split_a(rows, seed=42, user_protocol="temporal_loo")
    assert shared_user in trained.split_a | trained.split_b

    rows[0]["request_id"] = "test:someone:1"
    rows.append({**rows[-1], "split": "test", "request_id": "test:someone:1", "user_id": "someone"})
    with pytest.raises(TestLeakError, match="test events"):
        train_on_split_a(rows, seed=42, user_protocol="temporal_loo")
    with pytest.raises(ValueError, match="request_id"):
        train_on_split_a(make_population(), seed=42, user_protocol="temporal_loo")


def test_synthetic_audit_verifies_prefixes_and_does_not_call_train_metrics_leaked():
    report = audit_samples(synthetic_samples())
    assert report["ok"] is True
    assert report["protocol_verified"] is True
    assert report["n_target_event_in_history"] == 0
    assert report["n_repeated_target_item"] == 1
    assert report["users"]["n_shared_val_test"] == 1
    assert report["users"]["user_overlap_is_label_leakage"] is False
    assert report["collate"]["logged_ndcg_at5_at10_treated_as_leaked"] is False
    assert report["collate"]["stored_sequences_include_target_event"] is True
    assert report["tiger_inference_flat_rankings_leak_safe"] is False
    assert "NextKTokenMasking" in report["collate"]["note"]
    assert "collate_fn_inference_for_sequence" in report["collate"]["note"]
    assert "delta_ndcg@20" in report["p1_next_step"]
    assert report["benefit_aware_claimed"] is False
    assert report["n_checked"] == report["n_samples"]


def test_prefix_mismatch_fails_the_audit():
    samples = synthetic_samples()
    train = next(sample for sample in samples if sample["split"] == "train")
    train["source_sequence"] = [9, 9]
    train["history"] = [9]
    train["target"] = 9
    train["target_position"] = 1
    train["history_positions"] = [0]
    report = audit_samples(samples)
    assert report["ok"] is False
    assert report["prefixes"]["n_prefix_mismatch"] == 1
    assert report["n_target_event_in_history"] == 0


def test_jsonl_splits_round_trip_and_checkpoint_hash(tmp_path):
    for folder, sequence in (
        ("training", [1, 7, 2, 3]),
        ("evaluation", [1, 7, 2, 3, 4]),
        ("testing", [1, 7, 2, 3, 4, 7]),
    ):
        split_dir = tmp_path / folder
        split_dir.mkdir()
        (split_dir / "part.jsonl").write_text(
            json.dumps({"user_id": "a", "sequence": sequence}) + "\n",
            encoding="utf-8",
        )
    checkpoint = tmp_path / "checkpoint_epoch=000_step=018800.ckpt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    sid = tmp_path / "semantic_ids.pt"
    sid.write_bytes(b"sid-bytes")
    payload = command_export_eval_samples(
        smoke=False,
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        checkpoint=str(checkpoint),
        semantic_id_path=str(sid),
    )
    assert payload["exit_code"] == 0, payload
    assert payload["n"] == 3
    assert payload["n_repeated_target_item"] == 1
    loaded = read_jsonl(tmp_path / "out" / "eval_samples.jsonl")
    assert [row["split"] for row in loaded] == ["train", "val", "test"]
    assert loaded[-1]["history"] == [1, 7, 2, 3, 4]
    assert loaded[-1]["target"] == 7
    report = audit_samples(loaded)
    assert report["protocol_verified"] is True
    dry = command_export_eval_samples(
        smoke=False,
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "dry"),
        checkpoint_hash=payload["checkpoint_hash"],
        semantic_id_hash=payload["semantic_id_hash"],
        dry_run=True,
    )
    assert dry["exit_code"] == 0
    assert not (tmp_path / "dry" / "eval_samples.jsonl").exists()


def test_tfrecord_without_tensorflow_names_the_gpu_machine(tmp_path):
    split_dir = tmp_path / "testing"
    split_dir.mkdir()
    (split_dir / "part.tfrecord.gz").write_bytes(b"not-a-real-tfrecord")
    with pytest.raises(RuntimeError, match="tensorflow"):
        load_sequence_rows(tmp_path)


def test_ndcg_at_5_10_and_20_and_duplicate_hit_stays_at_one():
    target = 7
    ranked = [100, 101, 102, 103, 104, target, 106]
    scores = score_ranking(ranked, target)
    assert scores["ndcg@5"] == 0.0
    assert scores["recall@5"] == 0.0
    assert scores["ndcg@10"] == pytest.approx(1 / math.log2(7))
    assert scores["ndcg@20"] == pytest.approx(ndcg_at_k(ranked, [target], k=20))
    assert scores["recall@20"] == 1.0
    assert scores["rank"] == 6
    assert score_ranking([target, target, 1], target)["ndcg@20"] == pytest.approx(1.0)
    assert score_ranking([1, 2, 3], target)["ndcg@20"] == 0.0


def test_beam_scores_and_delta_are_not_auditable_for_the_inference_collate(tmp_path):
    samples = synthetic_samples()
    rankings = {beam: synthetic_rankings(samples, beam) for beam in (20, 40, 60)}
    per_request, summary = score_beams(samples, rankings, ranking_source="leak_safe_history")
    assert summary["auditable_for_delta_ndcg"] is True
    assert summary["benefit_aware_claimed"] is False
    assert summary["beams"]["20"]["ndcg@20"] == pytest.approx(0.0)
    assert summary["beams"]["60"]["ndcg@20"] == pytest.approx(1.0)
    assert summary["beams"]["40"]["ndcg@20"] == pytest.approx(1 / math.log2(7))
    assert summary["beams"]["40"]["ndcg@5"] == pytest.approx(0.0)
    assert summary["beams"]["40"]["recall@10"] == pytest.approx(1.0)
    assert summary["delta_ndcg@20"]["mean"] == pytest.approx(1.0)
    assert all(row["delta_ndcg@20"] == pytest.approx(1.0) for row in per_request)
    paths = write_score_outputs(tmp_path, per_request, summary)
    assert read_jsonl(paths["per_request_jsonl"])[0]["beams"]["20"]["ndcg@5"] == 0.0
    table = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "ndcg@20" in table
    assert "delta_60_minus_20" in table

    contaminated, contaminated_summary = score_beams(
        samples, rankings, ranking_source="grid_tiger_inference_flat"
    )
    assert contaminated_summary["auditable_for_delta_ndcg"] is False
    assert contaminated[0]["delta_ndcg@20"] == pytest.approx(1.0)
    unspecified = score_beams(samples, rankings, ranking_source="unspecified")[1]
    assert unspecified["auditable_for_delta_ndcg"] is False


def test_rankings_join_on_user_id_decode_semantic_ids_and_read_pickle(tmp_path):
    samples = synthetic_samples()
    only_b = [sample for sample in samples if sample["user_id"] == "b"]
    rankings = {20: [{"user_id": "b", "ranked_items": [8, 5, 6]}]}
    per_request, _summary = score_beams(only_b, rankings, ranking_source="leak_safe_history")
    assert per_request[0]["beams"]["20"]["rank"] == 1
    with pytest.raises(ValueError, match="matches"):
        score_beams(samples, {20: [{"user_id": "a", "ranked_items": [7]}]}, ranking_source="leak_safe_history")

    sid_path = tmp_path / "sids.jsonl"
    sid_path.write_text(
        "\n".join(
            [
                json.dumps({"item_id": 8, "semantic_id": [1, 2, 3, 0]}),
                json.dumps({"item_id": 5, "semantic_id": [4, 5, 6, 1]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = load_sid_map(sid_path)
    ranking_path = tmp_path / "beam_20.jsonl"
    ranking_path.write_text(
        json.dumps(
            {"request_id": "test:b:2", "semantic_ids": [[1, 2, 3, 0], [9, 9, 9, 9]]}
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_rankings(ranking_path, sid_to_item=mapping, beam=20)
    assert loaded[0]["ranked_items"][0] == 8
    assert loaded[0]["n_unknown_codes"] == 1
    matrix = [
        [1, 4, 0, 0, 0, 0],
        [2, 5, 0, 0, 0, 0],
        [3, 6, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ]
    assert codes_from_sid_matrix(matrix)[:2] == [(1, 2, 3, 0), (4, 5, 6, 1)]

    pickle_path = tmp_path / "beam_60.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump([{"user_id": "b", "ranked_items": [1, 8]}], handle)
    pickled = load_rankings(pickle_path, beam=60)
    assert pickled[0]["user_id"] == "b"
    assert pickled[0]["ranked_items"] == [1, 8]


def test_cli_smoke_writes_auditable_files_and_refuses_missing_inputs(tmp_path, repo_root):
    exported = command_export_eval_samples(smoke=True, output_dir=str(tmp_path / "export"))
    assert exported["exit_code"] == 0
    assert exported["n_target_event_in_history"] == 0
    assert (tmp_path / "export" / "eval_samples.jsonl").is_file()
    audited = command_audit_eval_leakage(
        smoke=False,
        samples=str(tmp_path / "export" / "eval_samples.jsonl"),
        output=str(tmp_path / "export" / "audit_report.json"),
    )
    assert audited["exit_code"] == 0
    assert audited["tiger_inference_flat_rankings_leak_safe"] is False
    assert audited["logged_ndcg_at5_at10_treated_as_leaked"] is False
    scored = command_score_beam_ndcg(smoke=True, output_dir=str(tmp_path / "score"))
    assert scored["exit_code"] == 0
    assert scored["auditable_for_delta_ndcg"] is True
    assert scored["mean_delta_ndcg@20"] == pytest.approx(1.0)
    assert scored["benefit_aware_claimed"] is False

    missing = subprocess.run(
        [sys.executable, "-m", "budget_sid", "export_eval_samples", "--config", "configs/lightweight_beauty.yaml"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1
    payload = json.loads(missing.stdout)
    assert payload["status"] == "error"
    assert "smoke" in payload["message"]
    assert "gpu_path_not_run" not in payload["status"]

    script = repo_root / "scripts" / "p0_eval_audit.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "/root/autodl-tmp/Budget-SID" in text
    assert "SKIP_SHUTDOWN" in text
    assert script.stat().st_mode & stat.S_IXUSR
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if "poweroff" in code or "shutdown" in code:
            assert "echo" in code
