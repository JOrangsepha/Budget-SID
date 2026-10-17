"""History-only predict inputs, ranking export, and the SeetaCloud beam script."""

import json
import os
import pickle
import stat
import subprocess
import sys

import pytest

from budget_sid.beam_score import command_score_beam_ndcg
from budget_sid.errors import HistoryLeakError
from budget_sid.eval_audit import audit_samples
from budget_sid.eval_samples import build_eval_sample, synthetic_samples, write_jsonl
from budget_sid.leak_safe_predict import (
    HYDRA_CKPT,
    HYDRA_PREDICT_FOLDER_KEY,
    HYDRA_TIMEOUT,
    HYDRA_WORKERS,
    PREDICT_FILENAME,
    RANKING_SOURCE,
    command_build_leak_safe_predict,
    command_export_leak_safe_rankings,
    encoder_item_window,
    history_only_rows,
    rankings_from_predictions,
    synthetic_sid_map,
    write_predict_dataset,
)
from budget_sid.tfrecord_io import crc32c, read_tfrecord_gz

HASHES = {"checkpoint_hash": "ckpt", "semantic_id_hash": "sid"}


def _sid_jsonl(path, samples):
    mapping = synthetic_sid_map(samples)
    lines = [
        json.dumps({"item_id": item, "semantic_id": list(code)})
        for code, item in mapping.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mapping


def test_crc32c_known_answer():
    assert crc32c(b"123456789") == 0xE3069283


def test_history_only_rows_keep_repeated_purchases_and_drop_the_target_event():
    samples = synthetic_samples()
    rows = history_only_rows(samples)
    by_request = {row["request_id"]: row for row in rows}
    test_row = by_request["test:a:5"]
    assert test_row["sequence_data"] == [1, 7, 2, 3, 4]
    assert test_row["sequence_data"].count(7) == 1
    assert test_row["n_history"] == 5
    assert [row["predict_key"] for row in rows] == list(range(len(rows)))
    assert encoder_item_window() == 30
    leaked = build_eval_sample(user_id="u", split="test", sequence=[1, 2, 3], **HASHES)
    leaked["history"] = [1, 2, 3]
    leaked["history_positions"] = [0, 1, 2]
    with pytest.raises(HistoryLeakError, match="target event"):
        history_only_rows([leaked])


def test_tfrecord_round_trip_does_not_append_the_target(tmp_path):
    samples = synthetic_samples()
    rows = history_only_rows(samples)
    written = write_predict_dataset(tmp_path, rows)
    loaded = read_tfrecord_gz(written["tfrecord"])
    assert len(loaded) == len(rows)
    for original, record in zip(rows, loaded):
        assert record["sequence_data"] == original["sequence_data"]
        assert record["user_id"] == [original["predict_key"]]
        assert len(record["sequence_data"]) == original["n_history"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["includes_target_event"] is False
    assert manifest["hydra_data_folder_key"] == HYDRA_PREDICT_FOLDER_KEY
    assert manifest["max_items_kept_by_collate"] == 30
    assert manifest["do_not_score_2026_10_17_tiger_inference_flat_pickles"] is True
    empty = build_eval_sample(user_id="e", split="test", sequence=[9], **HASHES)
    empty_rows = history_only_rows([empty])
    assert empty_rows[0]["sequence_data"] == []
    write_predict_dataset(tmp_path / "empty", empty_rows)
    assert read_tfrecord_gz(tmp_path / "empty" / PREDICT_FILENAME)[0]["sequence_data"] == []


def test_rankings_join_on_predict_key_and_score_as_leak_safe(tmp_path):
    samples = synthetic_samples()
    rows = history_only_rows(samples)
    sid = synthetic_sid_map(samples)
    item_to_code = {item: code for code, item in sid.items()}
    predictions = []
    for sample, row in zip(samples, rows):
        target_code = list(item_to_code[sample["target"]])
        other = next(code for code, item in sid.items() if item != sample["target"])
        predictions.append(
            {"user_id": row["predict_key"], "semantic_ids": [target_code, list(other), target_code]}
        )
    index = {row["predict_key"]: row for row in rows}
    ranked, n_unknown = rankings_from_predictions(predictions, index, sid, beam=60)
    assert n_unknown == 0
    assert ranked[0]["request_id"] == samples[0]["request_id"]
    assert ranked[0]["ranked_items"][0] == samples[0]["target"]
    assert ranked[0]["ranked_items"].count(samples[0]["target"]) == 1
    rank_dir = tmp_path / "rankings"
    rank_dir.mkdir()
    for beam, name in ((20, "beam_20.jsonl"), (40, "beam_40.jsonl"), (60, "beam_60.jsonl")):
        payload = []
        for sample, row in zip(samples, ranked):
            items = [sample["target"]] if beam == 60 else [row["ranked_items"][-1]]
            payload.append({"request_id": sample["request_id"], "ranked_items": items, "beam": beam})
        write_jsonl(rank_dir / name, payload)
    write_jsonl(tmp_path / "eval_samples.jsonl", samples)
    scored = command_score_beam_ndcg(
        smoke=False,
        samples=str(tmp_path / "eval_samples.jsonl"),
        rankings_dir=str(rank_dir),
        output_dir=str(tmp_path / "scores"),
        ranking_source=RANKING_SOURCE,
    )
    assert scored["exit_code"] == 0, scored
    assert scored["auditable_for_delta_ndcg"] is True
    assert scored["ranking_source"] == "leak_safe_history"
    assert scored["mean_ndcg@20_beam60"] == pytest.approx(1.0)


def test_export_reads_a_grid_pickle(tmp_path):
    samples = synthetic_samples()
    rows = history_only_rows(samples)
    sid_path = tmp_path / "sids.jsonl"
    sid = _sid_jsonl(sid_path, samples)
    item_to_code = {item: list(code) for code, item in sid.items()}
    pickle_path = tmp_path / "merged_predictions.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(
            [
                {"user_id": row["predict_key"], "semantic_ids": [item_to_code[sample["target"]]]}
                for sample, row in zip(samples, rows)
            ],
            handle,
        )
    write_jsonl(tmp_path / "predict_index.jsonl", rows)
    exported = command_export_leak_safe_rankings(
        smoke=False,
        predictions=str(pickle_path),
        index=str(tmp_path / "predict_index.jsonl"),
        sid_map=str(sid_path),
        beam=40,
        output=str(tmp_path / "beam_40.jsonl"),
    )
    assert exported["exit_code"] == 0, exported
    assert exported["n_unknown_codes"] == 0
    loaded = json.loads((tmp_path / "beam_40.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert loaded["request_id"] == rows[0]["request_id"]
    assert loaded["ranked_items"] == [samples[0]["target"]]
    assert "target" not in loaded


def test_truncation_window_does_not_fail_the_audit_or_count_as_leakage():
    full = list(range(12))
    cap = 4
    rows = []
    for split, sequence in (
        ("train", full[:-2][-cap:]),
        ("val", full[:-1][-cap:]),
        ("test", full[-cap:]),
    ):
        rows.append(build_eval_sample(user_id="long", split=split, sequence=sequence, **HASHES))
    for sample in synthetic_samples():
        if sample["user_id"] == "a":
            rows.append(sample)
    report = audit_samples(rows)
    assert report["ok"] is True
    assert report["status"] == "verified_with_truncation"
    assert report["protocol_exact"] is False
    assert report["protocol_verified"] is True
    assert report["n_target_event_in_history"] == 0
    assert report["prefixes"]["n_prefix_truncation_consistent"] == 1
    assert report["prefixes"]["n_prefix_exact"] == 1
    assert report["tiger_inference_flat_rankings_leak_safe"] is False
    assert "tail window" in report["tiger_inference_flat_rankings_reason"]

    broken = [
        build_eval_sample(user_id="z", split="train", sequence=[1], **HASHES),
        build_eval_sample(user_id="z", split="val", sequence=[2], **HASHES),
        build_eval_sample(user_id="z", split="test", sequence=[3], **HASHES),
    ]
    failed = audit_samples(broken)
    assert failed["ok"] is False
    assert failed["prefixes"]["n_prefix_mismatch"] == 1
    assert failed["n_target_event_in_history"] == 0


def test_build_smoke_and_missing_samples():
    payload = command_build_leak_safe_predict(smoke=True)
    assert payload["exit_code"] == 0
    assert payload["includes_target_event"] is False
    assert payload["n_target_event_in_history"] == 0
    assert payload["scripted"] is True
    missing = command_build_leak_safe_predict(smoke=False)
    assert missing["exit_code"] == 1
    assert "smoke" in missing["message"]


def test_script_dry_run_builds_history_only_and_prints_the_gpu_command(tmp_path, repo_root):
    samples = synthetic_samples()
    sample_path = tmp_path / "eval_samples.jsonl"
    write_jsonl(sample_path, samples)
    sid_path = tmp_path / "semantic_ids.jsonl"
    _sid_jsonl(sid_path, samples)
    out_dir = tmp_path / "leak_safe_beams"
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "ROOT_DIR": str(repo_root),
        "SAMPLES": str(sample_path),
        "SID_PATH": str(sid_path),
        "OUT_DIR": str(out_dir),
        "DRY_RUN": "1",
        "CKPT": str(tmp_path / "checkpoint_epoch=000_step=018800.ckpt"),
    }
    script = repo_root / "scripts" / "run_leak_safe_beams.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    records = read_tfrecord_gz(out_dir / "predict" / PREDICT_FILENAME)
    assert records[2]["sequence_data"] == [1, 7, 2, 3, 4]
    assert records[2]["user_id"] == [2]
    text = result.stdout
    assert HYDRA_CKPT in script.read_text(encoding="utf-8")
    assert "ckpt_path='" in text
    assert HYDRA_WORKERS in script.read_text(encoding="utf-8")
    assert HYDRA_TIMEOUT in script.read_text(encoding="utf-8")
    assert f"{HYDRA_PREDICT_FOLDER_KEY}={out_dir / 'predict'}" in text
    assert "--ranking-source leak_safe_history" in text
    assert "SKIP_SHUTDOWN" in text
    assert "2026-10-17" in script.read_text(encoding="utf-8")
    assert not (out_dir / "rankings" / "beam_20.jsonl").exists()

    refused = subprocess.run(
        ["bash", str(script)],
        env={**env, "LEGACY_BEAM_DIR": str(tmp_path / "old-pickles")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1
    assert "not leak-safe" in refused.stderr
    assert "SKIP_SHUTDOWN" in refused.stdout


def test_infer_script_marks_done_only_on_zero_and_passes_hydra_fixes(tmp_path, repo_root):
    grid = tmp_path / "GRID"
    grid.mkdir()
    ckpt = grid / "checkpoint_epoch=000_step=018800.ckpt"
    ckpt.write_text("not-a-real-checkpoint", encoding="utf-8")
    sid = tmp_path / "sid.pt"
    sid.write_text("sid", encoding="utf-8")
    predict = tmp_path / "predict"
    predict.mkdir()
    (predict / PREDICT_FILENAME).write_bytes(b"tfrecord-placeholder")
    (predict / "predict_index.jsonl").write_text("{}\n", encoding="utf-8")
    samples = tmp_path / "eval_samples.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args_out = tmp_path / "args.txt"
    (bindir / "python").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                'printf "%s\\n" "$@" >> "${ARGS_OUT}"',
                'printf "\\n---\\n" >> "${ARGS_OUT}"',
                'exit "${FAKE_RC:-0}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (bindir / "python").chmod(0o755)
    script = repo_root / "scripts" / "run_leak_safe_beams.sh"
    assert script.stat().st_mode & stat.S_IXUSR

    def run(rc, markers):
        env = {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
            "ROOT_DIR": str(repo_root),
            "GRID_DIR": str(grid),
            "CKPT": str(ckpt),
            "SID_PATH": str(sid),
            "DATA_DIR": str(grid / "data"),
            "SAMPLES": str(samples),
            "PREDICT_DIR": str(predict),
            "OUT_DIR": str(tmp_path / "out"),
            "RANK_DIR": str(tmp_path / "rankings"),
            "MARKER_DIR": str(markers),
            "TOP_KS": "20",
            "SKIP_BUILD": "1",
            "SKIP_EXPORT": "1",
            "SKIP_SCORE": "1",
            "ARGS_OUT": str(args_out),
            "FAKE_RC": str(rc),
        }
        return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)

    args_out.write_text("", encoding="utf-8")
    ok = run(0, tmp_path / "markers-ok")
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "markers-ok" / "leak_safe_beam_20.done").is_file()
    args = args_out.read_text(encoding="utf-8")
    assert "experiment=tiger_inference_flat" in args
    assert f"ckpt_path='{ckpt}'" in args
    assert HYDRA_WORKERS in args
    assert HYDRA_TIMEOUT in args
    assert f"{HYDRA_PREDICT_FOLDER_KEY}={predict}" in args
    assert "callbacks.pickle_writer.output_dir=" in args
    assert "test_dataloader_config" not in args
    assert "SKIP_SHUTDOWN" in ok.stdout
    assert "poweroff" in ok.stdout

    failed = run(7, tmp_path / "markers-fail")
    assert failed.returncode == 7
    assert not (tmp_path / "markers-fail" / "leak_safe_beam_20.done").exists()
    assert "SKIP_SHUTDOWN" in failed.stdout
