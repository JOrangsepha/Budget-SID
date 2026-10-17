import json
import subprocess
import sys

import pytest

from budget_sid.analyze_benefit import (
    analyze_per_request_rows,
    analyze_summary_only,
    command_analyze_benefit,
    to_per_request_record,
    _synthetic_summary,
)
from budget_sid.eval_samples import write_jsonl
from budget_sid.synthetic import selective_oracle_counterexample


def test_smoke_counterexample_reports_fractions_and_oracle():
    payload = command_analyze_benefit(smoke=True)
    assert payload["exit_code"] == 0
    assert payload["population"] == "synthetic_counterexample"
    assert payload["benefit_aware_claimed"] is False
    assert payload["grid_recommender_started"] is False
    assert payload["n"] == 10
    assert payload["fraction_positive"] == pytest.approx(0.2)
    assert payload["fraction_zero"] == pytest.approx(0.0)
    assert payload["fraction_negative"] == pytest.approx(0.8)
    assert payload["mean_delta_ndcg@20"] == pytest.approx(-0.6)
    assert payload["all_20_ndcg@20"] == pytest.approx(0.8)
    assert payload["all_40_ndcg@20"] == pytest.approx(0.8)
    assert payload["all_60_ndcg@20"] == pytest.approx(0.2)
    assert payload["oracle_ndcg@20"] == pytest.approx(1.0)
    assert payload["oracle_mean_beam"] == pytest.approx(28.0)
    assert payload["oracle_n_upgraded"] == 2
    assert payload["rejected_force_fill_ndcg@20"] == pytest.approx(0.7)
    assert payload["oracle_ndcg@20"] > payload["rejected_force_fill_ndcg@20"]
    assert payload["rejected_force_fill_ndcg@20"] < payload["all_40_ndcg@20"]
    assert payload["p2_worth_doing"] is True
    assert payload["decision_split"] == "val"


def test_per_request_schema_round_trip(tmp_path):
    rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
    rows.append(
        {
            "request_id": "val:flat-00:0",
            "user_id": "flat-00",
            "split": "val",
            "beams": {
                "20": {"ndcg@20": 0.5, "recall@20": 1.0},
                "40": {"ndcg@20": 0.5, "recall@20": 1.0},
                "60": {"ndcg@20": 0.5, "recall@20": 1.0},
            },
            "delta_ndcg@20": 0.0,
        }
    )
    path = tmp_path / "per_request.jsonl"
    write_jsonl(path, rows)
    summary = _synthetic_summary(rows)
    summary["ranking_source"] = "leak_safe_history"
    summary["auditable_for_delta_ndcg"] = True
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    payload = command_analyze_benefit(per_request=str(path), summary=str(summary_path))
    assert payload["exit_code"] == 0
    assert payload["population"] == "per_request"
    assert payload["n"] == 11
    assert payload["fraction_positive"] == pytest.approx(2 / 11)
    assert payload["fraction_zero"] == pytest.approx(1 / 11)
    assert payload["fraction_negative"] == pytest.approx(8 / 11)
    assert payload["summary_cross_check"]["ok"] is True
    assert payload["benefit_aware_claimed"] is False
    assert payload["p2_worth_doing"] is True


def test_summary_means_do_not_clear_p2(repo_root):
    path = repo_root / "artifacts" / "LEAK_SAFE_BEAMS_SUMMARY.json"
    payload = command_analyze_benefit(summary=str(path))
    assert payload["exit_code"] == 0
    assert payload["population"] == "summary_only"
    assert payload["per_request_analyzed"] is False
    assert payload["n"] == 67089
    assert payload["ndcg@20_beam20"] == pytest.approx(0.06356)
    assert payload["ndcg@20_beam40"] == pytest.approx(0.06425)
    assert payload["ndcg@20_beam60"] == pytest.approx(0.06418)
    assert payload["mean_delta_ndcg@20"] == pytest.approx(0.000618)
    assert payload["min_delta_ndcg@20"] == pytest.approx(-0.369)
    assert payload["max_delta_ndcg@20"] == pytest.approx(1.0)
    assert payload["recall@20_beam20"] == pytest.approx(0.1388)
    assert payload["recall@20_beam40"] == pytest.approx(0.1416)
    assert payload["recall@20_beam60"] == pytest.approx(0.1414)
    assert payload["gap_ndcg@20_beam40_minus_beam20"] == pytest.approx(0.00069)
    assert payload["uniform_beam60_beats_beam40"] is False
    assert payload["mean_delta_exceeds_beam40_gap"] is False
    assert payload["p2_identified_by_summary_means"] is False
    assert payload["p2_worth_doing"] is None
    assert payload["p2_reasons"] == ["per_request_labels_required"]
    assert payload["fraction_positive"] is None
    assert payload["benefit_aware_claimed"] is False
    assert payload["grid_recommender_started"] is False
    assert payload["one_max_request_mean_shift"] == pytest.approx(1.0 / 67089)
    assert payload["one_min_request_mean_shift"] == pytest.approx(-0.369 / 67089)
    direct = analyze_summary_only(json.loads(path.read_text(encoding="utf-8")))
    assert direct["p2_worth_doing"] is None


def test_p2_decision_uses_validation_not_the_pooled_file():
    val_rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
    test_rows = []
    for index in range(100):
        test_rows.append(
            {
                "request_id": f"test:flat-{index:03d}:0",
                "user_id": f"flat-{index:03d}",
                "split": "test",
                "beams": {
                    "20": {"ndcg@20": 0.0},
                    "40": {"ndcg@20": 0.0},
                    "60": {"ndcg@20": 1.0},
                },
                "delta_ndcg@20": 1.0,
            }
        )
    rows = val_rows + test_rows
    summary = _synthetic_summary(rows)
    summary["ranking_source"] = "leak_safe_history"
    summary["auditable_for_delta_ndcg"] = True
    payload = analyze_per_request_rows(rows, summary=summary)
    assert payload["n"] == 110
    assert payload["concentrated_gain"] is False
    assert payload["decision_split"] == "val"
    assert payload["decision_concentrated_gain"] is True
    assert payload["decision_oracle_beats_all_40"] is True
    assert payload["p2_worth_doing"] is True
    assert payload["by_split"]["test"]["n"] == 100
    assert payload["benefit_aware_claimed"] is False


def test_test_split_alone_cannot_clear_p2():
    rows = [
        {
            "request_id": "test:only:0",
            "user_id": "only",
            "split": "test",
            "beams": {"20": {"ndcg@20": 0.0}, "40": {"ndcg@20": 0.0}, "60": {"ndcg@20": 1.0}},
            "delta_ndcg@20": 1.0,
        }
    ]
    payload = analyze_per_request_rows(rows, summary={"auditable_for_delta_ndcg": True, "ranking_source": "leak_safe_history"})
    assert payload["p2_worth_doing"] is False
    assert "no_validation_rows" in payload["p2_reasons"]


def test_unconfirmed_audit_does_not_clear_p2():
    rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
    payload = analyze_per_request_rows(rows, summary=None)
    assert payload["oracle_beats_all_40"] is True
    assert payload["p2_worth_doing"] is None
    assert "summary_required_to_confirm_leak_safe_labels" in payload["p2_reasons"]


def test_non_auditable_summary_blocks_p2():
    rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
    summary = _synthetic_summary(rows)
    summary["ranking_source"] = "grid_tiger_inference_flat"
    summary["auditable_for_delta_ndcg"] = False
    payload = analyze_per_request_rows(rows, summary=summary)
    assert payload["p2_worth_doing"] is False
    assert "not_auditable_for_delta_ndcg" in payload["p2_reasons"]
    assert payload["oracle_ndcg@20"] == pytest.approx(1.0)


def test_cross_check_reports_a_mismatch_without_replacing_the_oracle():
    rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
    summary = _synthetic_summary(rows)
    summary["n_requests"] = 67089
    summary["auditable_for_delta_ndcg"] = True
    summary["ranking_source"] = "leak_safe_history"
    payload = analyze_per_request_rows(rows, summary=summary)
    assert payload["summary_cross_check"]["ok"] is False
    assert payload["n"] == 10
    assert payload["oracle_ndcg@20"] == pytest.approx(1.0)


def test_cli_per_request_path(tmp_path, repo_root):
    rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
    path = tmp_path / "per_request.jsonl"
    write_jsonl(path, rows)
    summary = _synthetic_summary(rows)
    summary["ranking_source"] = "leak_safe_history"
    summary["auditable_for_delta_ndcg"] = True
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "budget_sid",
            "analyze_benefit",
            "--per-request",
            str(path),
            "--summary",
            str(summary_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "analyze_benefit"
    assert payload["fraction_positive"] == pytest.approx(0.2)
    assert payload["oracle_ndcg@20"] == pytest.approx(1.0)
    assert payload["all_40_ndcg@20"] == pytest.approx(0.8)
    assert payload["all_60_ndcg@20"] == pytest.approx(0.2)
    assert payload["p2_worth_doing"] is True
    assert payload["benefit_aware_claimed"] is False


def test_benefit_doc_uses_the_seeta_summary_and_does_not_clear_p2(repo_root):
    text = (repo_root / "docs" / "P1_BENEFIT_ANALYSIS.md").read_text(encoding="utf-8")
    for snippet in (
        "The summary means do not clear P2.",
        "67089",
        "0.06356",
        "0.06425",
        "0.06418",
        "0.000618",
        "-0.369",
        "0.1388",
        "0.1416",
        "0.1414",
        "benefit-aware online",
        "GridRecommender is not started",
        "synthetic_counterexample",
    ):
        assert snippet in text
    assert "Beauty oracle beats" not in text
    assert "p2_worth_doing` from the summary alone is null" in text
    assert "docs/P2_OFFLINE_REPLAY.md" in text
    assert "p2_worth_doing=true" in text


def test_delta_field_must_match_beam_scores():
    row = to_per_request_record(selective_oracle_counterexample()[0])
    row["delta_ndcg@20"] = 0.25
    with pytest.raises(ValueError, match="delta_ndcg@20"):
        analyze_per_request_rows([row])
