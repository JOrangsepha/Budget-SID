import json
import subprocess
import sys

import numpy as np
import pytest

from budget_sid.errors import TestLeakError
from budget_sid.eval_samples import write_jsonl
from budget_sid.features import extract_features
from budget_sid.benefit import n_large_for_budget
from budget_sid.replay import (
    DEFAULT_UPGRADE_FRACTIONS,
    command_replay_budgets,
    declared_mean_beam,
    n_quota_for_fraction,
    n_quota_from_positive_count,
    parse_upgrade_fractions,
    realized_mean_beam,
    replay_records,
    score_allocation,
    select_exact_quota,
)
from budget_sid.synthetic import concentrated_tail_population

SUMMARY = {
    "ranking_source": "synthetic_concentrated_tail",
    "auditable_for_delta_ndcg": True,
    "benefit_aware_claimed": False,
}


def _selection_map(payload, flag):
    mapping = {}
    for row in payload["assignments"]:
        if row["group"] != "B":
            continue
        mapping[row["request_id"]] = {
            key: (None if flags is None else flags.get(flag))
            for key, flags in (row.get("by_fraction") or {}).items()
        }
    return mapping


def _status(payload, name):
    statuses = {point["policies"][name]["status"] for point in payload["curve"]}
    assert len(statuses) == 1
    return statuses.pop()


def test_exact_quota_breaks_ties_without_a_label_argument():
    chosen = select_exact_quota([0.0, 0.0, 0.0, 1.0], ["b", "a", "c", "d"], 2)
    assert chosen == ["a", "d"]
    assert select_exact_quota([0.2, 0.2, 0.2], ["c", "b", "a"], 2) == ["a", "b"]
    assert select_exact_quota([1.0, 2.0], ["a", "b"], 0) == []


def test_concentrated_tail_matches_quota_and_beats_random():
    payload = replay_records(concentrated_tail_population(), seed=42, summary=SUMMARY)
    assert payload["protocol"] == "p2.1"
    assert payload["quota_mode"] == "fixed_fraction"
    assert payload["quota_derived_from_positive_count"] is False
    assert payload["seeds_retrain_recommender"] is False
    assert payload["p3_started"] is False
    assert payload["user_protocol"] == "disjoint_users"
    assert payload["test_lock"] == "disjoint_users"
    assert payload["splits_disjoint"] is True
    assert payload["validation_premise"]["p2_worth_doing"] is True
    assert payload["split_b"]["uniform_beam60_beats_beam40"] is False
    assert payload["validation_premise"]["uniform_beam60_beats_beam40"] is False
    assert payload["features"]["full_p2_ready"] is True
    assert payload["features"]["invented_logits"] is False
    assert payload["features"]["full_p2_blockers"] == []
    assert payload["invented_logits"] is False
    assert payload["latency_claimed"] is False
    assert payload["benefit_aware_online_claimed"] is False
    assert payload["grid_recommender_started"] is False
    assert payload["mse_is_not_the_allocation_metric"] is True
    assert payload["explanatory_metrics_are_not_primary"] is True
    assert "split_b_mse" in payload["predictor"]["ridge"]
    base = payload["split_b"]
    positive_quota = n_quota_from_positive_count(base["n"], base["n_positive"])
    assert any(point["n_quota"] != positive_quota for point in payload["curve"])
    point = next(candidate for candidate in payload["curve"] if 0 < candidate["n_quota"] <= base["n_positive"])
    beams = set()
    for name in ("random", "entropy", "benefit_aware"):
        policy = point["policies"][name]
        assert policy["status"] == "ran"
        assert policy["uses_labels"] is False
        assert policy["n_large"] == point["n_quota"]
        beams.add(policy["mean_beam"])
        assert "n_positive_captured" not in policy["primary"]
        assert "negative_loss" not in policy["primary"]
    assert len(beams) == 1
    oracle = point["oracle_upper_bound"]
    benefit = point["policies"]["benefit_aware"]
    random = point["policies"]["random"]
    entropy = point["policies"]["entropy"]
    assert oracle["uses_labels"] is True
    assert oracle["n_large"] == point["n_quota"]
    assert oracle["forced_negative_delta_upgrades"] == 0
    assert oracle["explanatory"]["negative_loss"] == pytest.approx(0.0)
    assert entropy["explanatory"]["n_positive_captured"] == 0
    assert benefit["explanatory"]["n_positive_captured"] == point["n_quota"]
    assert benefit["primary"]["mean_ndcg@20"] == pytest.approx(oracle["primary"]["mean_ndcg@20"])
    assert benefit["primary"]["mean_ndcg@20"] > random["primary"]["mean_ndcg@20"]
    assert benefit["primary"]["delta_vs_all_beam20"] > random["primary"]["delta_vs_all_beam20"]
    wide = next(candidate for candidate in payload["curve"] if candidate["upgrade_fraction"] == pytest.approx(0.5))
    assert wide["n_quota"] == n_quota_for_fraction(base["n"], 0.5)
    assert wide["n_quota"] > base["n_positive"]
    assert wide["oracle_upper_bound"]["n_large"] == base["n_positive"]
    assert wide["oracle_upper_bound"]["same_compute_as_policies"] is False
    assert wide["oracle_upper_bound"]["mean_beam"] < wide["realized_mean_beam"]
    assert wide["policies"]["random"]["n_large"] == wide["n_quota"]
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
    assert _selection_map(original, "selected_benefit_aware") == _selection_map(replayed, "selected_benefit_aware")
    assert _selection_map(original, "selected_benefit_aware_hgb") == _selection_map(
        replayed, "selected_benefit_aware_hgb"
    )
    assert _selection_map(original, "selected_entropy") == _selection_map(replayed, "selected_entropy")
    assert _selection_map(original, "selected_random") == _selection_map(replayed, "selected_random")
    assert _selection_map(original, "selected_oracle") != _selection_map(replayed, "selected_oracle")


def test_three_seeds_keep_one_quota_inside_each_seed():
    config = {"train": {"seed": 42}}
    payload = command_replay_budgets(config, smoke=True, seeds="42,43,44")
    assert payload["exit_code"] == 0
    assert payload["scripted"] is True
    assert payload["quota_derived_from_positive_count"] is False
    assert payload["seeds_retrain_recommender"] is False
    assert [item["seed"] for item in payload["by_seed"]] == [42, 43, 44]
    for item in payload["by_seed"]:
        assert item["quota_mode"] == "fixed_fraction"
        for point in item["curve"]:
            counts = {
                point["policies"][name]["n_large"]
                for name in ("random", "entropy", "benefit_aware")
            }
            assert counts == {point["n_quota"]}
            assert point["quota_derived_from_positive_count"] is False
            assert point["n_quota"] == n_quota_for_fraction(item["split_b"]["n"], point["upgrade_fraction"])
            assert point["oracle_upper_bound"]["n_large"] == min(item["split_b"]["n_positive"], point["n_quota"])
            assert point["oracle_upper_bound"]["explanatory"]["negative_loss"] == pytest.approx(0.0)
            if 0 < point["n_quota"] <= item["split_b"]["n_positive"]:
                assert point["policies"]["benefit_aware"]["explanatory"]["n_positive_captured"] == point["n_quota"]
                assert point["policies"]["entropy"]["explanatory"]["n_positive_captured"] == 0
    aggregate = payload["curve"][-1]["policies"]["benefit_aware"]
    assert "mean" in aggregate["primary"]["mean_ndcg@20"]
    assert "std" in aggregate["primary"]["delta_vs_all_beam20"]
    assert aggregate["explanatory"]["role"] == "explanatory_only"
    assert "mean" in aggregate["explanatory"]["n_positive_captured"]


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
    assert _status(payload, "entropy") == "skipped"
    assert _status(payload, "benefit_aware") == "ran"
    assert payload["curve"][0]["policies"]["benefit_aware"]["uses_labels"] is False
    blocker_text = " ".join(payload["features"]["full_p2_blockers"])
    assert "first_step_entropy" in blocker_text
    assert "candidate_prob_gap" in blocker_text
    assert "top1_prob" in blocker_text
    assert "does not invent logits" in blocker_text
    assert payload["latency_claimed"] is False
    assert payload["grid_recommender_started"] is False
    assert (tmp_path / "out" / "replay_summary.json").is_file()
    assert (tmp_path / "out" / "curve_summary.json").is_file()
    assert (tmp_path / "out" / "assignments_seed42.jsonl").is_file()
    curve = json.loads((tmp_path / "out" / "curve_summary.json").read_text(encoding="utf-8"))
    assert curve["quota_derived_from_positive_count"] is False
    assert curve["primary_metrics"] == ["mean_ndcg@20", "delta_vs_all_beam20"]
    assert [point["upgrade_fraction"] for point in curve["curve"]] == list(DEFAULT_UPGRADE_FRACTIONS)


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
    assert _status(payload, "entropy") == "skipped"
    assert "delta_ndcg@20" not in payload["features"]["feature_names"]

    write_jsonl(tmp_path / "features.jsonl", feature_rows)
    joined = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(tmp_path / "per_request.jsonl"),
        features=str(tmp_path / "features.jsonl"),
    )
    assert joined["features"]["full_p2_ready"] is True
    assert joined["features"]["dropped_label_features"] == ["delta_ndcg@20"]
    assert _status(joined, "entropy") == "ran"
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
    assert _status(payload, "entropy") == "ran"
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
    assert _status(payload, "entropy") == "skipped"
    assert _status(payload, "random") == "ran"


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
    assert smoke["quota_derived_from_positive_count"] is False
    assert smoke["curve"][-1]["policies"]["benefit_aware"]["mean_beam"] == pytest.approx(
        smoke["curve"][-1]["policies"]["random"]["mean_beam"]
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
    assert body["primary_metrics"] == ["mean_ndcg@20", "delta_vs_all_beam20"]
    assert body["explanatory_metrics"] == ["n_positive_captured", "negative_loss"]
    assert body["explanatory_metrics_are_not_primary"] is True
    assert body["quota_derived_from_positive_count"] is False
    assert body["seeds_retrain_recommender"] is False
    assert "positive_delta_mass_vs_oracle" not in body["primary_metrics"]


def test_upgrade_fraction_quota_math():
    assert DEFAULT_UPGRADE_FRACTIONS == (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
    for fraction in DEFAULT_UPGRADE_FRACTIONS:
        assert declared_mean_beam(fraction) == pytest.approx(20 + 40 * fraction)
    # Recorded Beauty validation size. The old narrow quota was min(207, ceiling).
    n = 22363
    n_positive = 207
    assert n_quota_for_fraction(n, 0.5) == n_large_for_budget(n)
    assert n_quota_from_positive_count(n, n_positive) == n_positive
    assert n_quota_for_fraction(n, 0.01) != n_positive
    for fraction in DEFAULT_UPGRADE_FRACTIONS:
        n_quota = n_quota_for_fraction(n, fraction)
        assert n_quota == int(np.floor(fraction * n + 1e-12))
        realized = realized_mean_beam(n, n_quota)
        assert realized == pytest.approx(20 + 40 * (n_quota / n))
        assert realized <= declared_mean_beam(fraction) + 1e-9
    assert declared_mean_beam(0.0) == pytest.approx(20.0)
    assert declared_mean_beam(0.01) == pytest.approx(20.4)
    assert declared_mean_beam(0.25) == pytest.approx(30.0)
    assert declared_mean_beam(0.5) == pytest.approx(40.0)
    assert parse_upgrade_fractions(None) == DEFAULT_UPGRADE_FRACTIONS
    assert parse_upgrade_fractions("0%,1%,5%,10%,25%,50%") == DEFAULT_UPGRADE_FRACTIONS
    assert parse_upgrade_fractions("0,1,5,10,25,50") == DEFAULT_UPGRADE_FRACTIONS
    assert parse_upgrade_fractions("0,0.01,0.05,0.10,0.25,0.50") == DEFAULT_UPGRADE_FRACTIONS


def test_default_replay_does_not_use_positive_count_quota():
    payload = replay_records(concentrated_tail_population(), seed=42, summary=SUMMARY)
    assert payload["quota_mode"] == "fixed_fraction"
    assert payload["quota_derived_from_positive_count"] is False
    base = payload["split_b"]
    legacy_quota = n_quota_from_positive_count(base["n"], base["n_positive"])
    assert payload["upgrade_fractions"] == list(DEFAULT_UPGRADE_FRACTIONS)
    for point in payload["curve"]:
        assert point["quota_source"] == "fixed_fraction"
        assert point["quota_derived_from_positive_count"] is False
        assert point["n_quota"] == n_quota_for_fraction(base["n"], point["upgrade_fraction"])
        assert point["declared_mean_beam"] == pytest.approx(20 + 40 * point["upgrade_fraction"])
        assert point["realized_mean_beam"] == pytest.approx(20 + 40 * (point["n_quota"] / base["n"]))
        ran = [
            point["policies"][name]
            for name in ("random", "entropy", "benefit_aware")
            if point["policies"][name]["status"] == "ran"
        ]
        assert {row["n_large"] for row in ran} == {point["n_quota"]}
        oracle = point["oracle_upper_bound"]
        assert oracle["n_large"] == min(base["n_positive"], point["n_quota"])
        assert oracle["forced_negative_delta_upgrades"] == 0
        assert oracle["filled_quota_with_nonpositive"] is False
        assert oracle["explanatory"]["negative_loss"] == pytest.approx(0.0)
        assert oracle["explanatory"]["n_negative_upgraded"] == 0
    assert any(point["n_quota"] != legacy_quota for point in payload["curve"])
    zero = payload["curve"][0]
    assert zero["upgrade_fraction"] == pytest.approx(0.0)
    assert zero["n_quota"] == 0
    assert zero["policies"]["random"]["primary"]["delta_vs_all_beam20"] == pytest.approx(0.0)
    assert zero["policies"]["random"]["primary"]["mean_ndcg@20"] == pytest.approx(base["all_20_ndcg@20"])


def test_positive_count_quota_is_opt_in():
    rows = concentrated_tail_population()
    legacy = replay_records(rows, seed=42, summary=SUMMARY, quota_mode="positive_count")
    assert legacy["protocol"] == "p2_narrow_positive_count"
    assert legacy["quota_derived_from_positive_count"] is True
    assert legacy["upgrade_fractions"] is None
    assert len(legacy["curve"]) == 1
    point = legacy["curve"][0]
    base_n = legacy["split_b"]["n"]
    n_positive = legacy["split_b"]["n_positive"]
    assert point["n_quota"] == n_quota_from_positive_count(base_n, n_positive)
    assert point["quota_source"] == "positive_count"
    assert point["policies"]["random"]["n_large"] == point["n_quota"]
    assert point["policies"]["benefit_aware"]["n_large"] == point["n_quota"]
    assert point["oracle_upper_bound"]["n_large"] == point["n_quota"]
    assert point["oracle_upper_bound"]["forced_negative_delta_upgrades"] == 0
    with pytest.raises(ValueError, match="positive_count"):
        replay_records(rows, seed=42, summary=SUMMARY, quota_mode="positive_count", upgrade_fractions=(0.1,))
    refused = command_replay_budgets(
        {"train": {"seed": 42}},
        smoke=True,
        quota_mode="positive_count",
        fractions="0.1,0.5",
    )
    assert refused["exit_code"] == 1
    assert "positive_count" in refused["message"]


def test_fraction_sweep_writes_curve_summary(tmp_path):
    rows = concentrated_tail_population(n_test=0)
    path = tmp_path / "per_request.jsonl"
    write_jsonl(path, rows)
    payload = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(path),
        fractions="0%,1%,5%,10%,25%,50%",
        seeds="42,43",
        output_dir=str(tmp_path / "out"),
    )
    assert payload["exit_code"] == 0, payload
    assert payload["quota_derived_from_positive_count"] is False
    assert payload["seeds_retrain_recommender"] is False
    assert [point["upgrade_fraction"] for point in payload["by_seed"][0]["curve"]] == list(DEFAULT_UPGRADE_FRACTIONS)
    curve_path = tmp_path / "out" / "curve_summary.json"
    document = json.loads(curve_path.read_text(encoding="utf-8"))
    assert document["seeds"] == [42, 43]
    assert document["primary_metrics"] == ["mean_ndcg@20", "delta_vs_all_beam20"]
    assert document["explanatory_metrics"] == ["n_positive_captured", "negative_loss"]
    assert len(document["curve"]) == 6
    for point in document["curve"]:
        assert point["quota_derived_from_positive_count"] is False
        benefit = point["policies"]["benefit_aware"]
        assert set(benefit["primary"]) == {"mean_ndcg@20", "delta_vs_all_beam20"}
        assert benefit["explanatory"]["role"] == "explanatory_only"
        assert "mean" in benefit["primary"]["mean_ndcg@20"]
        assert "mean" in benefit["explanatory"]["negative_loss"]
    percent = command_replay_budgets(
        {"train": {"seed": 42}},
        per_request=str(path),
        fractions="0,1,5,10,25,50",
    )
    assert percent["exit_code"] == 0
    assert percent["upgrade_fractions"] == list(DEFAULT_UPGRADE_FRACTIONS)


def test_score_allocation_keeps_explanatory_metrics_separate():
    rows = [
        {"request_id": "p", "ndcg20": 0.0, "ndcg60": 0.8, "delta": 0.8},
        {"request_id": "n", "ndcg20": 0.6, "ndcg60": 0.1, "delta": -0.5},
        {"request_id": "z", "ndcg20": 0.3, "ndcg60": 0.3, "delta": 0.0},
    ]
    scored = score_allocation(rows, ["p", "n"])
    assert scored["primary"]["mean_ndcg@20"] == pytest.approx(0.4)
    assert scored["primary"]["delta_vs_all_beam20"] == pytest.approx(0.1)
    assert scored["mean_beam"] == pytest.approx((2 * 60 + 20) / 3)
    assert scored["explanatory"]["n_positive_captured"] == 1
    assert scored["explanatory"]["negative_loss"] == pytest.approx(0.5)
    assert scored["explanatory"]["n_negative_upgraded"] == 1
    assert scored["explanatory"]["role"] == "explanatory_only"
    assert "n_positive_captured" not in scored["primary"]
    assert "negative_loss" not in scored["primary"]


def test_hgb_shares_the_fraction_quota_when_available():
    payload = replay_records(concentrated_tail_population(), seed=42, summary=SUMMARY)
    info = payload["predictor"]["hist_gradient_boosting"]
    assert info["same_features_as_ridge"] is True
    assert info["boundary_features"] is False
    assert info["max_depth"] <= 3
    assert info["max_iter"] <= 50
    point = payload["curve"][-1]
    ridge = point["policies"]["benefit_aware"]
    hgb = point["policies"]["benefit_aware_hgb"]
    assert ridge["status"] == "ran"
    assert ridge["n_large"] == point["n_quota"]
    if info["status"] == "unavailable":
        assert hgb["status"] == "unavailable"
        assert hgb["n_large"] is None
        return
    assert info["status"] == "fit"
    assert hgb["status"] == "ran"
    assert hgb["uses_labels"] is False
    assert hgb["n_large"] == ridge["n_large"]
    assert hgb["mean_beam"] == pytest.approx(ridge["mean_beam"])
