"""GPU entry points and the transcribed SeetaCloud Beauty log."""

import json
import os
import stat
import subprocess

from budget_sid.experiment import GPU_PLAN_MESSAGE, command_evaluate_budgets


REQUIRED_SNIPPETS = (
    "ckpt_path='$CKPT'",
    "data_loading.predict_dataloader_config.dataloader.num_workers=0",
    "data_loading.predict_dataloader_config.dataloader.timeout=0",
    "SKIP_SHUTDOWN",
)


def _script(repo_root, name):
    return (repo_root / "scripts" / name).read_text(encoding="utf-8")


def test_gpu_scripts_parse_and_keep_hydra_fixes(repo_root):
    for name in (
        "run_after_tiger.sh",
        "run_beauty_gpu.sh",
        "run_leak_safe_beams.sh",
        "dump_first_step_features.sh",
        "dump_first_step_logits.sh",
    ):
        path = repo_root / "scripts" / name
        text = path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(path)], check=True)
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, name
        for snippet in REQUIRED_SNIPPETS:
            assert snippet in text, name
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if "poweroff" in code or "shutdown" in code:
                assert "echo" in code, line


def test_readme_records_finished_gpu_work_and_open_gate(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "ckpt_path='$CKPT'" in readme
    assert "predict_dataloader_config" in readme
    assert "test_dataloader_config" in readme
    assert "SKIP_SHUTDOWN" in readme
    assert "0.0293" in readme
    assert "0.0366" in readme
    assert "0.0456" in readme
    assert "0.0682" in readme
    assert "0.0361" in readme
    assert "0.0453" in readme
    assert "0.059" in readme
    assert "- [x]" in readme
    assert "ΔNDCG@20" in readme
    assert "TIGER 论文" in readme


def test_beauty_artifacts_match_seeta_log(repo_root):
    results = json.loads((repo_root / "artifacts" / "BEAUTY_GPU_RESULTS.json").read_text(encoding="utf-8"))
    protocol = json.loads((repo_root / "artifacts" / "PROTOCOL_SUMMARY.json").read_text(encoding="utf-8"))
    assert results["source"] == "SeetaCloud"
    assert results["source_date"] == "2026-10-17"
    assert results["gpu_rerun_by_this_commit"] is False
    assert results["all_done"] == "2026-10-17T12:09:00+08:00"
    assert results["grid_commit"] == "2fe3475b2d369580234093f35d52b1a2f54d0472"
    assert results["test"] == {
        "ndcg@5": 0.0293,
        "ndcg@10": 0.0366,
        "recall@5": 0.0456,
        "recall@10": 0.0682,
    }
    assert results["val"] == {"ndcg@5": 0.0361, "ndcg@10": 0.0453}
    assert results["checkpoint"]["best_val_recall@5"] == 0.059
    assert results["checkpoint"]["stored_in_git"] is False
    assert results["checkpoint"]["filename"] == "checkpoint_epoch=000_step=018800.ckpt"
    assert [row["top_k"] for row in results["beam_inference"]] == [20, 40, 60]
    assert [row["exit_code"] for row in results["beam_inference"]] == [0, 0, 0]
    assert [row["rows_approximate"] for row in results["beam_inference"]] == [22363, 22363, 22363]
    assert "ndcg@20" not in results["test"]
    assert results["instance_left_on"] is True
    assert results["auto_shutdown"] is False

    assert protocol["source"] == "SeetaCloud"
    assert protocol["source_date"] == "2026-10-17"
    assert protocol["gpu_rerun_by_this_commit"] is False
    statuses = {stage["name"]: stage["status"] for stage in protocol["stages"]}
    assert statuses["MiniLM"] == "done"
    assert statuses["fixed-beam infer"] == "done"
    assert statuses["benefit-aware stop-gate"] == "not_wired"
    assert protocol["checkpoint_in_git"] is False
    assert protocol["instance"]["left_on"] is True


def test_nonsmoke_plan_points_at_recorded_results(beauty_config):
    payload = command_evaluate_budgets(beauty_config, smoke=False)
    assert payload["exit_code"] == 1
    assert payload["status"] == "gpu_path_not_run"
    assert "BEAUTY_GPU_RESULTS.json" in GPU_PLAN_MESSAGE
    assert "test_metrics" not in payload


def _fake_python(path, args_out):
    path.write_text(
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
    path.chmod(0o755)


def _run_infer(script, env):
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)


def test_infer_scripts_mark_done_only_on_zero(tmp_path, repo_root):
    grid = tmp_path / "GRID"
    grid.mkdir()
    ckpt = grid / "checkpoint_epoch=000_step=018800.ckpt"
    ckpt.write_text("not-a-real-checkpoint", encoding="utf-8")
    sid = tmp_path / "sid.pt"
    sid.write_text("sid", encoding="utf-8")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args_out = tmp_path / "args.txt"
    _fake_python(bindir / "python", args_out)

    base_env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GRID_DIR": str(grid),
        "CKPT": str(ckpt),
        "SID_PATH": str(sid),
        "DATA_DIR": str(grid / "data" / "amazon_data" / "beauty"),
        "TOP_KS": "20",
        "ARGS_OUT": str(args_out),
        "FAKE_RC": "0",
        "SKIP_PIN": "1",
        "SKIP_TRAIN": "1",
    }

    for script_name in ("run_after_tiger.sh", "run_beauty_gpu.sh"):
        markers = tmp_path / f"markers-{script_name}"
        env = dict(base_env)
        env["MARKER_DIR"] = str(markers)
        env["FAKE_RC"] = "0"
        args_out.write_text("", encoding="utf-8")
        result = _run_infer(repo_root / "scripts" / script_name, env)
        assert result.returncode == 0, result.stderr
        assert (markers / "tiger_infer_top20.done").is_file()
        args = args_out.read_text(encoding="utf-8")
        assert "experiment=tiger_inference_flat" in args
        assert f"ckpt_path='{ckpt}'" in args
        assert "data_loading.predict_dataloader_config.dataloader.num_workers=0" in args
        assert "data_loading.predict_dataloader_config.dataloader.timeout=0" in args
        assert "test_dataloader_config" not in args
        assert "SKIP_SHUTDOWN" in result.stdout
        assert "poweroff" in result.stdout

        fail_markers = tmp_path / f"fail-{script_name}"
        env["MARKER_DIR"] = str(fail_markers)
        env["FAKE_RC"] = "7"
        result = _run_infer(repo_root / "scripts" / script_name, env)
        assert result.returncode == 7, result.stderr
        assert not (fail_markers / "tiger_infer_top20.done").exists()
        assert "SKIP_SHUTDOWN" in result.stdout
