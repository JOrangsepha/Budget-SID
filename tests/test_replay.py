import json
import subprocess
import sys

import pytest

from budget_sid.errors import TestLeakError
from budget_sid.eval_samples import write_jsonl
from budget_sid.features import extract_features
from budget_sid.replay import command_replay_budgets, replay_records, select_exact_quota
from budget_sid.synthetic import concentrated_tail_population

SUMMARY = {
    "ranking_source": "synthetic_concentrated_tail",
    "auditable_for_delta_ndcg": True,
    "benefit_aware_claimed": False,
}


def _selected(payload, flag):
    return {row["request_id"] for row in payload["assignments"] if row.get(flag)}


def test_exact_quota_breaks_ties_without_a_label_argument():
    chosen = select_exact_quota([0.0, 0.0, 0.0, 1.0], ["b", "a", "c", "d"], 2)
    assert chosen == ["a", "d"]
    assert select_exact_quota([0.2, 0.2, 0.2], ["c", "b", "a"], 2) == ["a", "b"]
    assert select_exact_quota([1.0, 2.0], ["a", "b"], 0) == []


def test_concentrated_tail_matches_quota_and_beats_random():
    payload = replay_records(concentrated_tail_population(), seed=42, summary=SUMMARY)
    assert payload["user_protocol"] == "disjoint_users"
    assert payload["test_lock"] == "disjoint_users"
    assert payload["splits_disjoint"] is True
    assert payload["validation_premise"]["p2_worth_doing"] is True
    assert payload["split_b"]["uniform_beam60_beats_beam40"] is False
    assert payload["validation_premise"]["uniform_beam60_beats_beam40"] is False
    assert payload["split_b"]["n_quota"] == payload["split_b"]["n_positive"]
    assert payload["split_b"]["n_quota"] < payload["split_b"]["n_budget_ceiling"]
    assert payload["features"]["full_p2_ready"] is True
    assert payload["features"]["invented_logits"] is False
    assert payload["features"]["full_p2_blockers"] == []
    assert payload["invented_logits"] is False
    assert payload["latency_claimed"] is False
    assert payload["benefit_aware_online_claimed"] is False
    assert payload["grid_recommender_started"] is False
    assert payload["mse_is_not_the_allocation_metric"] is True
    assert "split_b_mse" in payload["predictor"]
    ran = ["random", "entropy", "benefit_aware", "oracle"]
    beams = set()
    counts = set()
    for name in ran:
        policy = payload["policies"][name]
        assert policy["status"] == "ran"
        counts.add(policy["n_large"])
        beams.add(policy["mean_beam"])
        assert policy["n_large"] == payload["split_b"]["n_quota"]
    assert len(counts) == 1
    assert len(beams) == 1
    benefit = payload["policies"]["benefit_aware"]
    random = payload["policies"]["random"]
    entropy = payload["policies"]["entropy"]
    oracle = payload["policies"]["oracle"]
    assert benefit["uses_labels"] is False
    assert random["uses_labels"] is False
    assert entropy["uses_labels"] is False
    assert oracle["uses_labels"] is True
    assert entropy["positive_delta_mass"] == pytest.approx(0.0)
    assert benefit["positive_delta_mass_vs_oracle"] == pytest.approx(1.0)
    assert benefit["positive_delta_mass_vs_oracle"] > random["positive_delta_mass_vs_oracle"]
    assert benefit["mean_ndcg@20"] == pytest.approx(oracle["mean_ndcg@20"])
    assert benefit["mean_ndcg@20"] > random["mean_ndcg@20"]
    assert benefit["mean_ndcg@20"] > payload["split_b"]["all_40_ndcg@20"]
    assigned_users = {row["user_id"] for row in payload["assignments"]}
    assert assigned_users.isdisjoint({f"test-{index:02d}" for index in range(4)})


def test_benefit_selection_ignores_split_b_labels():
    rows = concentrated_tail_population()
    original = replay_records(rows, seed=42, summary=SUMMARY)
    b_users = {row["user_id"] for row in original["assignments"] if row["group"] == "B"}
    positive = next(row for row in rows if row["user_id"] in b_users and row["delta_ndcg@20"] > 0)
    decoy = next(row for row in rows if row["user_id"] in b_users and row["user_id"].startswith("decoy"))
    swapped = json.loads(json.dumps(rows))
    by_id = {row["request_id"]: row for row in swapped}
    left = by_id[positive["request_id"]]
    right = by_id[decoy["request_id"]]
    left["beams"], right["beams"] = right["beams"], left["beams"]
    left["delta_ndcg@20"], right["delta_ndcg@20"] = right["delta_ndcg@20"], left["delta_ndcg@20"]
    replayed = replay_records(swapped, seed=42, summary=SUMMARY)
    assert _selected(original, "selected_benefit_aware") == _selected(replayed, "selected_benefit_aware")
    assert _selected(original, "selected_entropy") == _selected(replayed, "selected_entropy")
    assert _selected(original, "selected_random") == _selected(replayed, "selected_random")
    assert _selected(original, "selected_oracle") != _selected(replayed, "selected_oracle")


def test_three_seeds_keep_one_quota_inside_each_seed():
    config = {"train": {"seed": 42}}
    payload = command_replay_budgets(config, smoke=True, seeds="42,43,44")
    assert payload["exit_code"] == 0
    assert payload["scripted"] is True
    assert [item["seed"] for item in payload["by_seed"]] == [42, 43, 44]
    for item in payload["by_seed"]:
        counts = {
            item["policies"][name]["n_large"]
            for name in ("random", "entropy", "benefit_aware", "oracle")
        }
        assert counts == {item["split_b"]["n_quota"]}
        assert item["policies"]["benefit_aware"]["positive_delta_mass_vs_oracle"] == pytest.approx(1.0)
        assert item["policies"]["entropy"]["positive_delta_mass"] == pytest.approx(0.0)
    assert payload["policies"]["benefit_aware"]["positive_delta_mass_vs_oracle"]["mean"] == pytest.approx(1.0)
    assert "std" in payload["policies"]["random"]["mean_ndcg@20"]


def test_missing_logits_are_a_blocker_not_filled_in(tmp_path):
    rows = concentrated_tail_population(n_test=0)
    for row in rows:
        row.pop("features")
        row.pop("history")
    path = tmp_path / "per_request.jsonl"
    write_jsonl(path, rows)
    summary = dict(SUMMARY)
    summary["ranking_source"] = "leak_safe_history"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    payload = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(path),
        output_dir=str(tmp_path / "out"),
    )
    assert payload["exit_code"] == 0
    assert payload["summary_source"] == "sibling"
    assert payload["invented_logits"] is False
    assert payload["features"]["history_length_source"] == "target_position"
    assert payload["features"]["feature_names"] == ["history_length"]
    assert payload["features"]["logit_features_present"] is False
    assert payload["features"]["full_p2_ready"] is False
    assert payload["policies"]["entropy"]["status"] == "skipped"
    assert payload["policies"]["benefit_aware"]["status"] == "ran"
    assert payload["policies"]["benefit_aware"]["uses_labels"] is False
    blocker_text = " ".join(payload["features"]["full_p2_blockers"])
    assert "first_step_entropy" in blocker_text
    assert "candidate_prob_gap" in blocker_text
    assert "top1_prob" in blocker_text
    assert "does not invent logits" in blocker_text
    assert payload["latency_claimed"] is False
    assert payload["grid_recommender_started"] is False
    assert (tmp_path / "out" / "replay_summary.json").is_file()
    assert (tmp_path / "out" / "assignments_seed42.jsonl").is_file()


def test_samples_and_logits_join_on_request_id(tmp_path):
    rows = concentrated_tail_population(n_test=0)
    samples = []
    feature_rows = []
    for row in rows:
        samples.append(
            {
                "request_id": row["request_id"],
                "user_id": row["user_id"],
                "split": "val",
                "history": list(row["history"]),
                "target": 999,
            }
        )
        feature_rows.append(
            {
                "request_id": row["request_id"],
                "history_length": len(row["history"]),
                "logits": [0.0, 0.0, 0.0, 4.0] if row["user_id"].startswith("positive") else [3.0, 0.0, 0.0, 0.0],
                "delta_ndcg@20": row["delta_ndcg@20"],
            }
        )
        row.pop("features")
        row.pop("target_position")
    write_jsonl(tmp_path / "per_request.jsonl", rows)
    write_jsonl(tmp_path / "samples.jsonl", samples)
    payload = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(tmp_path / "per_request.jsonl"),
        samples=str(tmp_path / "samples.jsonl"),
        summary=None,
    )
    assert payload["features"]["history_length_source"] == "eval_samples"
    assert payload["features"]["feature_names"] == ["history_length"]
    assert payload["policies"]["entropy"]["status"] == "skipped"
    assert "delta_ndcg@20" not in payload["features"]["feature_names"]

    write_jsonl(tmp_path / "features.jsonl", feature_rows)
    joined = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(tmp_path / "per_request.jsonl"),
        features=str(tmp_path / "features.jsonl"),
    )
    assert joined["features"]["full_p2_ready"] is True
    assert joined["features"]["dropped_label_features"] == ["delta_ndcg@20"]
    assert joined["policies"]["entropy"]["status"] == "ran"
    assert joined["invented_logits"] is False
    expected = extract_features(1, [0.0, 0.0, 0.0, 4.0])
    assert joined["features"]["logit_features_present"] is True
    assert expected["first_step_entropy"] > 0.0


def test_logits_use_history_length_from_samples():
    rows = concentrated_tail_population(n_positive=4, n_decoy=4, n_negative=4, n_zero=4, n_test=0)
    samples = []
    feature_rows = []
    for row in rows:
        row.pop("features")
        row.pop("target_position")
        samples.append({"request_id": row["request_id"], "history": list(row["history"]), "target": 1})
        feature_rows.append(
            {
                "request_id": row["request_id"],
                "logits": [1.0, 0.0, 0.0, 0.0],
                "delta_ndcg@20": row["delta_ndcg@20"],
            }
        )
    payload = replay_records(
        rows,
        seed=42,
        summary=SUMMARY,
        feature_rows=feature_rows,
        sample_rows=samples,
    )
    assert payload["features"]["history_length_source"] == "eval_samples"
    assert payload["features"]["full_p2_ready"] is True
    assert payload["features"]["dropped_label_features"] == ["delta_ndcg@20"]
    assert payload["policies"]["entropy"]["status"] == "ran"
    assert payload["invented_logits"] is False


def test_history_length_prefers_samples_over_target_position():
    rows = concentrated_tail_population(n_positive=4, n_decoy=4, n_negative=4, n_zero=4, n_test=0)
    for row in rows:
        row.pop("features")
        row["target_position"] = 9
    samples = [
        {
            "request_id": row["request_id"],
            "history": [1, 2, 3],
            "target": 8,
        }
        for row in rows
    ]
    payload = replay_records(rows, seed=42, summary=SUMMARY, sample_rows=samples)
    assert payload["features"]["history_length_source"] == "eval_samples"
    assert payload["features"]["history_length_disagreement_n"] == len(rows)
    assert payload["features"]["feature_names"] == ["history_length"]


def test_predict_index_supplies_history_length():
    rows = concentrated_tail_population(n_positive=4, n_decoy=4, n_negative=4, n_zero=4, n_test=0)
    for row in rows:
        row.pop("features")
        row.pop("history")
        row.pop("target_position")
    index_rows = [{"request_id": row["request_id"], "n_history": 4} for row in rows]
    payload = replay_records(rows, seed=42, summary=SUMMARY, index_rows=index_rows)
    assert payload["features"]["history_length_source"] == "predict_index"
    assert payload["policies"]["entropy"]["status"] == "skipped"
    assert payload["policies"]["random"]["status"] == "ran"


def test_temporal_loo_allows_shared_users_and_locks_events():
    rows = concentrated_tail_population(n_test=0)
    shared = rows[0]["user_id"]
    rows.append(
        {
            "request_id": f"test:{shared}:99",
            "user_id": shared,
            "split": "test",
            "target_position": 1,
            "features": dict(rows[0]["features"]),
            "beams": {
                "20": {"ndcg@20": 0.0},
                "40": {"ndcg@20": 0.0},
                "60": {"ndcg@20": 1.0},
            },
            "delta_ndcg@20": 1.0,
        }
    )
    payload = replay_records(rows, seed=42, summary=SUMMARY)
    assert payload["user_protocol"] == "temporal_loo"
    assert payload["test_lock"] == "temporal_loo"
    assert shared in {
        row["user_id"] for row in payload["assignments"]
    }
    assert all(row["group"] != "test" for row in payload["assignments"])
    with pytest.raises(TestLeakError):
        replay_records(rows, seed=42, summary=SUMMARY, user_protocol="disjoint_users")
    leaked = json.loads(json.dumps(rows))
    leaked[-1]["request_id"] = leaked[0]["request_id"]
    with pytest.raises((TestLeakError, ValueError)):
        replay_records(leaked, seed=42, summary=SUMMARY, user_protocol="temporal_loo")


def test_unauditable_summary_refuses_the_fit(tmp_path):
    rows = concentrated_tail_population(n_test=0)
    path = tmp_path / "per_request.jsonl"
    write_jsonl(path, rows)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"auditable_for_delta_ndcg": False, "ranking_source": "grid_tiger_inference_flat"}), encoding="utf-8")
    payload = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(path),
        summary=str(summary),
    )
    assert payload["exit_code"] == 1
    assert payload["ok"] is False
    assert "not leak-safe" in payload["message"]
    assert payload["invented_logits"] is False
    assert payload["grid_recommender_started"] is False


def test_cli_smoke_and_missing_path(repo_root, beauty_config):
    smoke = command_replay_budgets(beauty_config, smoke=True)
    assert smoke["exit_code"] == 0
    assert smoke["population"] == "synthetic_concentrated_tail"
    assert smoke["policies"]["benefit_aware"]["mean_beam"] == pytest.approx(
        smoke["policies"]["random"]["mean_beam"]
    )
    missing = command_replay_budgets(beauty_config, smoke=False)
    assert missing["exit_code"] == 1
    assert "per_request" in missing["message"]
    result = subprocess.run(
        [sys.executable, "-m", "budget_sid", "replay_budgets", "--smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["command"] == "replay_budgets"
    assert body["scripted"] is True
    assert body["latency_claimed"] is False
    assert body["benefit_aware_online_claimed"] is False
    assert "mean_ndcg@20" in body["allocation_metrics"]
    assert "positive_delta_mass_vs_oracle" in body["allocation_metrics"]
