"""Offline replay of beam budgets on leak-safe per-request scores.

P2 fits the existing ridge benefit model on validation users A and scores
policies on the disjoint validation users B. Every non-oracle policy spends
the same number of large beams. That number is the corrected oracle's spend
(positive Δ only, capped by the beam-40 ceiling), not a Bernoulli draw and
not a force-fill of the ceiling.

Selection on B does not read per-request ΔNDCG. The quota count does: it is
the number of positive-Δ requests on B, capped by the ceiling. Metrics are
mean nDCG@20, mean beam, and positive-Δ mass against the oracle. MSE is only
a fit diagnostic.

This module does not call GRID, does not time decoding, and does not claim
an online benefit-aware result. ``per_request.jsonl`` has no first-step
logits. Missing logit features are reported. They are not invented.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from budget_sid.analyze_benefit import analyze_per_request_rows
from budget_sid.benefit import fit_benefit_predictor, n_large_for_budget
from budget_sid.constants import LARGE_BEAM, REFERENCE_BEAM, SMALL_BEAM
from budget_sid.errors import TestLeakError
from budget_sid.eval_samples import canonical_split, read_jsonl, write_jsonl
from budget_sid.features import FEATURE_NAMES, extract_features, stack_named_features
from budget_sid.splits import USER_PROTOCOLS, assert_test_events_unused, assert_test_unused, split_val_users
from budget_sid.stop_gate import request_ndcg

LOGIT_FEATURES = ("first_step_entropy", "candidate_prob_gap", "top1_prob")
LABEL_FEATURE_KEYS = frozenset(
    {
        "delta",
        "delta_ndcg",
        "delta_ndcg@20",
        "target",
        "relevant",
        "ndcg",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "ndcg_by_beam",
        "beams",
        "rank",
        "recall@5",
        "recall@10",
        "recall@20",
        "repeated_target_item",
    }
)
QUOTA_NOTE = (
    "The shared quota on B is min(n_positive, n_budget_ceiling): the number of "
    "large beams the corrected oracle spends. It is not a Bernoulli probability "
    "and it does not force-fill the beam-40 ceiling with non-positive requests. "
    "The count uses the label marginal on B. Random, entropy, and benefit-aware "
    "choose which requests without reading per-request delta_ndcg@20."
)
FULL_P2_NOTE = (
    "Full P2 features are history_length, first_step_entropy, candidate_prob_gap, "
    "and top1_prob from budget_sid.features.extract_features after the shared first "
    "decoding step. score_beam_ndcg per_request.jsonl stores beam nDCG and "
    "delta_ndcg@20 only. It has no logits. Pass --features with those columns, or "
    "history_length plus logits. This replay does not invent logits. Entropy stays "
    "off until first_step_entropy is joined. history_length may come from --samples, "
    "--predict-index n_history, or target_position (P0 sets that equal to len(history))."
)


def select_exact_quota(scores, request_ids, n_quota: int) -> list[str]:
    """Pick exactly ``n_quota`` request ids. Highest score, then request id.

    The arguments are scores and ids. Per-request labels are not an input.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    ids = [str(request_id) for request_id in request_ids]
    if scores.shape[0] != len(ids):
        raise ValueError("scores and request ids must have the same length")
    n_quota = int(n_quota)
    if n_quota < 0:
        raise ValueError("n_quota must be non-negative")
    if n_quota == 0 or not ids:
        return []
    order = sorted(range(len(ids)), key=lambda index: (-float(scores[index]), ids[index]))
    chosen = order[: min(n_quota, len(ids))]
    return sorted(ids[index] for index in chosen)


def replay_records(
    rows,
    *,
    seed: int = 42,
    user_protocol: str = "auto",
    summary: dict | None = None,
    feature_rows=None,
    sample_rows=None,
    index_rows=None,
    alpha: float = 1e-2,
    include_assignments: bool = True,
) -> dict:
    """Fit on validation users A and score quota-matched policies on users B."""
    if user_protocol not in {"auto", *USER_PROTOCOLS}:
        raise ValueError(f"unknown user_protocol {user_protocol!r}")
    if summary is not None and summary.get("auditable_for_delta_ndcg") is False:
        raise ValueError(
            "summary auditable_for_delta_ndcg is false. "
            "Refusing to fit P2 on rankings that are not leak-safe history."
        )
    normalized = [_normalize_request(row) for row in rows]
    if not normalized:
        raise ValueError("per_request file has no rows")
    _reject_duplicate_ids(normalized)
    feature_report = _attach_features(
        normalized,
        feature_rows=feature_rows,
        sample_rows=sample_rows,
        index_rows=index_rows,
    )
    protocol = _resolve_protocol(normalized, user_protocol)
    grouped = _assign_groups(normalized, seed=seed, user_protocol=protocol)
    train_rows = grouped["A"]
    budget_rows = grouped["B"]
    if len(train_rows) < 2:
        raise ValueError("need at least two split-A requests to fit the predictor")
    if not budget_rows:
        raise ValueError("split B has no validation requests")
    names = _common_feature_names(train_rows + budget_rows)
    feature_report["feature_names"] = list(names)
    feature_report["logit_features_present"] = all(name in names for name in LOGIT_FEATURES)
    feature_report["full_p2_ready"] = list(names) == list(FEATURE_NAMES)
    feature_report["full_p2_blockers"] = _feature_blockers(names)
    feature_report["invented_logits"] = False
    n_quota, quota_detail = _quota(budget_rows)
    policies, predictor_info, assignments = _score_policies(
        train_rows,
        budget_rows,
        names=names,
        n_quota=n_quota,
        seed=seed,
        alpha=alpha,
        blockers=feature_report["full_p2_blockers"],
    )
    context = _split_context(budget_rows, n_quota=n_quota, quota_detail=quota_detail)
    premise = analyze_per_request_rows(rows, summary=summary)
    payload = {
        "seed": int(seed),
        "user_protocol": protocol,
        "user_protocol_requested": user_protocol,
        "n_split_a_users": len(grouped["split_a"]),
        "n_split_b_users": len(grouped["split_b"]),
        "n_split_a_rows": len(train_rows),
        "n_split_b_rows": len(budget_rows),
        "split_a_hash": _id_hash(grouped["split_a"]),
        "split_b_hash": _id_hash(grouped["split_b"]),
        "splits_disjoint": grouped["split_a"].isdisjoint(grouped["split_b"]),
        "test_rows_present": bool(grouped["test"]),
        "test_lock": grouped["test_lock"],
        "n_train_split_rows_ignored": grouped["n_ignored_train_split"],
        "n_split_assumed_validation": sum(1 for row in normalized if row["split_assumed"]),
        "quota_note": QUOTA_NOTE,
        "split_b": context,
        "policies": policies,
        "predictor": predictor_info,
        "features": feature_report,
        "validation_premise": _premise_view(premise),
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_recommender_started": False,
        "invented_logits": False,
        "allocation_metrics": ["mean_ndcg@20", "mean_beam", "positive_delta_mass_vs_oracle"],
        "mse_is_not_the_allocation_metric": True,
    }
    if include_assignments:
        payload["assignments"] = assignments
    return payload


def command_replay_budgets(
    config: dict,
    *,
    per_request: str | None = None,
    summary: str | None = None,
    samples: str | None = None,
    features: str | None = None,
    predict_index: str | None = None,
    output_dir: str | None = None,
    seed: int | None = None,
    seeds: str | None = None,
    user_protocol: str = "auto",
    smoke: bool = False,
    dry_run: bool = False,
) -> dict:
    command = "replay_budgets"
    try:
        seed_list = _parse_seeds(seed, seeds, int(config["train"]["seed"]))
        if per_request:
            rows = read_jsonl(per_request)
            summary_doc, summary_source = _load_summary(per_request, summary)
            feature_rows = read_jsonl(features) if features else None
            sample_rows = read_jsonl(samples) if samples else None
            index_rows = read_jsonl(predict_index) if predict_index else None
            population = "per_request"
            scripted = False
        elif smoke:
            from budget_sid.synthetic import concentrated_tail_population

            rows = concentrated_tail_population()
            summary_doc = {
                "ranking_source": "synthetic_concentrated_tail",
                "auditable_for_delta_ndcg": True,
                "benefit_aware_claimed": False,
            }
            summary_source = "smoke"
            feature_rows = None
            sample_rows = None
            index_rows = None
            population = "synthetic_concentrated_tail"
            scripted = True
        else:
            return _error(
                command,
                "Pass --per-request outputs/leak_safe_beams/per_request.jsonl, "
                "or --smoke for the synthetic concentrated tail. "
                "That jsonl is gitignored. This process does not invent it.",
            )
        payloads = [
            replay_records(
                rows,
                seed=one_seed,
                user_protocol=user_protocol,
                summary=summary_doc,
                feature_rows=feature_rows,
                sample_rows=sample_rows,
                index_rows=index_rows,
                include_assignments=bool(output_dir) and not dry_run,
            )
            for one_seed in seed_list
        ]
        written = {}
        if output_dir and not dry_run:
            written = _write_outputs(output_dir, payloads)
        payload = _public_payload(payloads)
        payload.update(
            {
                "command": command,
                "ok": True,
                "exit_code": 0,
                "status": "smoke" if scripted else "replayed",
                "scripted": scripted,
                "dry_run": bool(dry_run),
                "population": population,
                "seeds": seed_list,
                "summary_source": summary_source,
                "per_request": per_request,
                "output": written,
                "benefit_aware_online_claimed": False,
                "latency_claimed": False,
                "grid_recommender_started": False,
                "invented_logits": False,
            }
        )
        return payload
    except (OSError, ValueError, RuntimeError, TestLeakError, KeyError, TypeError) as exc:
        return _error(command, str(exc))


def _public_payload(payloads: list[dict]) -> dict:
    stripped = [_without_assignments(item) for item in payloads]
    if len(stripped) == 1:
        return dict(stripped[0])
    return {
        "seeds": [item["seed"] for item in stripped],
        "by_seed": stripped,
        "policies": _aggregate_policies(stripped),
        "features": stripped[0]["features"],
        "validation_premise": stripped[0]["validation_premise"],
        "user_protocol": stripped[0]["user_protocol"],
        "quota_note": stripped[0]["quota_note"],
        "allocation_metrics": stripped[0]["allocation_metrics"],
        "mse_is_not_the_allocation_metric": True,
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_recommender_started": False,
        "invented_logits": False,
    }


def _aggregate_policies(payloads: list[dict]) -> dict:
    aggregated = {}
    for name in ("random", "entropy", "benefit_aware", "oracle"):
        rows = [item["policies"][name] for item in payloads]
        if any(row.get("status") != "ran" for row in rows):
            aggregated[name] = dict(rows[0])
            continue
        aggregated[name] = {
            "status": "ran",
            "uses_labels": rows[0]["uses_labels"],
            "mean_ndcg@20": _mean_std(row["mean_ndcg@20"] for row in rows),
            "mean_beam": _mean_std(row["mean_beam"] for row in rows),
            "positive_delta_mass_vs_oracle": _mean_std(row["positive_delta_mass_vs_oracle"] for row in rows),
            "positive_delta_mass_fraction": _mean_std(row["positive_delta_mass_fraction"] for row in rows),
            "n_large": _mean_std(row["n_large"] for row in rows),
        }
    return aggregated


def _mean_std(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "std": std}


def _without_assignments(payload: dict) -> dict:
    copied = dict(payload)
    copied.pop("assignments", None)
    return copied


def _write_outputs(output_dir, payloads: list[dict]) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    public = _public_payload(payloads)
    summary_path = root / "replay_summary.json"
    summary_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"replay_summary": str(summary_path)}
    for payload in payloads:
        assignments = payload.get("assignments") or []
        path = root / f"assignments_seed{payload['seed']}.jsonl"
        write_jsonl(path, assignments)
        written[f"assignments_seed{payload['seed']}"] = str(path)
    return written


def _score_policies(train_rows, budget_rows, *, names, n_quota, seed, alpha, blockers):
    ids = [row["request_id"] for row in budget_rows]
    deltas = np.array([row["delta"] for row in budget_rows], dtype=np.float64)
    oracle_ids = select_exact_quota(deltas, ids, n_quota)
    oracle_mass = _positive_mass(budget_rows, oracle_ids)
    policies = {
        "oracle": _policy_row(
            budget_rows,
            oracle_ids,
            oracle_mass=oracle_mass,
            status="ran",
            uses_labels=True,
            reason="upper bound; ranks B by true delta_ndcg@20",
        )
    }
    rng = np.random.default_rng(int(seed))
    random_ids = select_exact_quota(rng.random(len(budget_rows)), ids, n_quota)
    policies["random"] = _policy_row(
        budget_rows,
        random_ids,
        oracle_mass=oracle_mass,
        status="ran",
        uses_labels=False,
        reason="exact sample of the shared quota; not an independent coin flip",
    )
    if "first_step_entropy" in names:
        entropy_scores = [float(row["features"]["first_step_entropy"]) for row in budget_rows]
        entropy_ids = select_exact_quota(entropy_scores, ids, n_quota)
        policies["entropy"] = _policy_row(
            budget_rows,
            entropy_ids,
            oracle_mass=oracle_mass,
            status="ran",
            uses_labels=False,
            reason="highest first_step_entropy under the shared quota",
        )
    else:
        entropy_ids = None
        policies["entropy"] = _skipped_policy(
            "first_step_entropy is not in the joined features. " + FULL_P2_NOTE
        )
    predictor_info = {
        "status": "blocked",
        "feature_names": list(names),
        "alpha": float(alpha),
        "n_train_rows": len(train_rows),
        "split_b_mse": None,
        "note": "MSE is a fit diagnostic. The comparison is mean nDCG@20, mean beam, and positive-Δ mass.",
    }
    predicted = None
    if not names:
        policies["benefit_aware"] = _skipped_policy(
            "No non-label feature is present on every validation row. " + FULL_P2_NOTE,
            status="blocked",
        )
    else:
        predictor = fit_benefit_predictor(
            stack_named_features([row["features"] for row in train_rows], names),
            np.array([row["delta"] for row in train_rows], dtype=np.float64),
            alpha=float(alpha),
        )
        predictor.feature_names = tuple(names)
        predicted = predictor.predict(stack_named_features([row["features"] for row in budget_rows], names))
        benefit_ids = select_exact_quota(predicted, ids, n_quota)
        policies["benefit_aware"] = _policy_row(
            budget_rows,
            benefit_ids,
            oracle_mass=oracle_mass,
            status="ran",
            uses_labels=False,
            reason="highest predicted delta on the features that were actually joined",
            extra_blockers=list(blockers),
        )
        mse = float(np.mean((np.asarray(predicted, dtype=np.float64) - deltas) ** 2))
        predictor_info = {
            "status": "fit",
            "feature_names": list(names),
            "alpha": float(alpha),
            "n_train_rows": len(train_rows),
            "split_b_mse": mse,
            "note": predictor_info["note"],
        }
    benefit_ids = policies["benefit_aware"].get("selected_request_ids")
    assignments = _assignment_rows(
        train_rows,
        budget_rows,
        random_ids=set(random_ids),
        entropy_ids=None if entropy_ids is None else set(entropy_ids),
        benefit_ids=None if benefit_ids is None else set(benefit_ids),
        oracle_ids=set(oracle_ids),
        predicted=predicted,
        ids=ids,
    )
    for policy in policies.values():
        policy.pop("selected_request_ids", None)
    return policies, predictor_info, assignments


def _assignment_rows(train_rows, budget_rows, *, random_ids, entropy_ids, benefit_ids, oracle_ids, predicted, ids):
    predicted_by_id = {}
    if predicted is not None:
        for request_id, score in zip(ids, predicted):
            predicted_by_id[request_id] = float(score)
    rows = []
    for group, group_rows in (("A", train_rows), ("B", budget_rows)):
        for row in group_rows:
            request_id = row["request_id"]
            on_b = group == "B"
            rows.append(
                {
                    "request_id": request_id,
                    "user_id": row["user_id"],
                    "group": group,
                    "delta_ndcg@20": row["delta"],
                    "predicted_delta": predicted_by_id.get(request_id),
                    "selected_random": request_id in random_ids if on_b else None,
                    "selected_entropy": request_id in entropy_ids if on_b and entropy_ids is not None else None,
                    "selected_benefit_aware": request_id in benefit_ids if on_b and benefit_ids is not None else None,
                    "selected_oracle": request_id in oracle_ids if on_b else None,
                }
            )
    return rows


def _policy_row(rows, selected_ids, *, oracle_mass, status, uses_labels, reason, extra_blockers=None):
    selected = set(selected_ids)
    n = len(rows)
    n_large = len(selected)
    ndcg_sum = 0.0
    mass = 0.0
    total_mass = 0.0
    n_positive_captured = 0
    for row in rows:
        delta = float(row["delta"])
        if delta > 0.0:
            total_mass += delta
        chosen = row["request_id"] in selected
        ndcg_sum += float(row["ndcg60"] if chosen else row["ndcg20"])
        if chosen and delta > 0.0:
            mass += delta
            n_positive_captured += 1
    mean_beam = (n_large * LARGE_BEAM + (n - n_large) * SMALL_BEAM) / n
    return {
        "status": status,
        "uses_labels": bool(uses_labels),
        "reason": reason,
        "n": n,
        "n_large": n_large,
        "mean_beam": float(mean_beam),
        "mean_ndcg@20": float(ndcg_sum / n),
        "positive_delta_mass": float(mass),
        "positive_delta_mass_total": float(total_mass),
        "positive_delta_mass_fraction": _ratio(mass, total_mass),
        "positive_delta_mass_vs_oracle": _ratio(mass, oracle_mass),
        "n_positive_captured": n_positive_captured,
        "full_p2_blockers": list(extra_blockers or []),
        "selected_request_ids": list(selected_ids),
    }


def _skipped_policy(reason: str, status: str = "skipped") -> dict:
    return {
        "status": status,
        "uses_labels": False,
        "reason": reason,
        "n": None,
        "n_large": None,
        "mean_beam": None,
        "mean_ndcg@20": None,
        "positive_delta_mass": None,
        "positive_delta_mass_total": None,
        "positive_delta_mass_fraction": None,
        "positive_delta_mass_vs_oracle": None,
        "n_positive_captured": None,
        "full_p2_blockers": [reason],
        "selected_request_ids": None,
    }


def _positive_mass(rows, selected_ids) -> float:
    selected = set(selected_ids)
    return float(sum(float(row["delta"]) for row in rows if row["request_id"] in selected and float(row["delta"]) > 0.0))


def _ratio(numerator: float, denominator: float):
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _quota(rows) -> tuple[int, dict]:
    deltas = np.array([row["delta"] for row in rows], dtype=np.float64)
    n_positive = int(np.sum(deltas > 0.0))
    n_budget = n_large_for_budget(len(rows), SMALL_BEAM, LARGE_BEAM, REFERENCE_BEAM)
    n_quota = int(min(n_positive, n_budget))
    return n_quota, {"n_positive": n_positive, "n_budget_ceiling": int(n_budget)}


def _split_context(rows, *, n_quota: int, quota_detail: dict) -> dict:
    n = len(rows)
    deltas = [float(row["delta"]) for row in rows]
    n_positive = sum(delta > 0.0 for delta in deltas)
    n_negative = sum(delta < 0.0 for delta in deltas)
    n_zero = n - n_positive - n_negative
    all_20 = float(np.mean([row["ndcg20"] for row in rows]))
    all_40 = float(np.mean([row["ndcg40"] for row in rows]))
    all_60 = float(np.mean([row["ndcg60"] for row in rows]))
    return {
        "n": n,
        "n_positive": n_positive,
        "n_zero": n_zero,
        "n_negative": n_negative,
        "fraction_positive": n_positive / n,
        "fraction_zero": n_zero / n,
        "fraction_negative": n_negative / n,
        "all_20_ndcg@20": all_20,
        "all_40_ndcg@20": all_40,
        "all_60_ndcg@20": all_60,
        "uniform_beam60_beats_beam40": bool(all_60 > all_40),
        "n_quota": int(n_quota),
        "n_positive_for_quota": quota_detail["n_positive"],
        "n_budget_ceiling": quota_detail["n_budget_ceiling"],
        "rejected_force_fill_ndcg@20": _force_fill_ndcg(rows),
        "quota_matches_corrected_oracle_spend": True,
    }


def _force_fill_ndcg(rows) -> float:
    """Spend the whole ceiling, including non-positive Δ. P1 rejects this."""
    n_budget = n_large_for_budget(len(rows), SMALL_BEAM, LARGE_BEAM, REFERENCE_BEAM)
    order = sorted(rows, key=lambda row: (-float(row["delta"]), str(row["user_id"]), row["request_id"]))
    chosen = {row["request_id"] for row in order[:n_budget]}
    total = 0.0
    for row in rows:
        total += float(row["ndcg60"] if row["request_id"] in chosen else row["ndcg20"])
    return float(total / len(rows))


def _premise_view(premise: dict) -> dict:
    return {
        "decision_split": premise.get("decision_split"),
        "p2_worth_doing": premise.get("p2_worth_doing"),
        "p2_reasons": list(premise.get("p2_reasons") or []),
        "n": premise.get("n"),
        "fraction_positive": premise.get("fraction_positive"),
        "fraction_zero": premise.get("fraction_zero"),
        "fraction_negative": premise.get("fraction_negative"),
        "oracle_ndcg@20": premise.get("decision_oracle_ndcg@20"),
        "all_40_ndcg@20": premise.get("decision_all_40_ndcg@20"),
        "oracle_beats_all_40": premise.get("decision_oracle_beats_all_40"),
        "uniform_beam60_beats_beam40": _uniform_from_premise(premise),
        "auditable_for_delta_ndcg": premise.get("auditable_for_delta_ndcg"),
        "summary_cross_check": premise.get("summary_cross_check"),
    }


def _uniform_from_premise(premise: dict):
    by_split = premise.get("by_split") or {}
    decision = premise.get("decision_split")
    stats = by_split.get(decision) if decision else None
    if not isinstance(stats, dict):
        stats = premise
    all_40 = stats.get("all_40_ndcg@20")
    all_60 = stats.get("all_60_ndcg@20")
    if all_40 is None or all_60 is None:
        return None
    return bool(float(all_60) > float(all_40))


def _assign_groups(rows, *, seed: int, user_protocol: str) -> dict:
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    ignored = sum(1 for row in rows if row["split"] == "train")
    split_a, split_b = split_val_users((row["user_id"] for row in val_rows), seed=seed)
    if user_protocol == "disjoint_users":
        assert_test_unused(split_a | split_b, [row["user_id"] for row in test_rows])
        test_lock = "disjoint_users" if test_rows else "no_test_rows_in_file"
    else:
        if any(not row["request_id"] for row in val_rows + test_rows):
            raise ValueError("temporal_loo requires request_id on every val and test row")
        assert_test_events_unused(
            [row["request_id"] for row in val_rows],
            [row["request_id"] for row in test_rows],
        )
        test_lock = "temporal_loo"
    return {
        "A": [row for row in val_rows if row["user_id"] in split_a],
        "B": [row for row in val_rows if row["user_id"] in split_b],
        "test": test_rows,
        "split_a": split_a,
        "split_b": split_b,
        "test_lock": test_lock,
        "n_ignored_train_split": ignored,
    }


def _resolve_protocol(rows, requested: str) -> str:
    if requested != "auto":
        return requested
    val_users = {row["user_id"] for row in rows if row["split"] == "val"}
    test_users = {row["user_id"] for row in rows if row["split"] == "test"}
    if val_users & test_users:
        return "temporal_loo"
    return "disjoint_users"


def _common_feature_names(rows) -> tuple[str, ...]:
    present = []
    for name in FEATURE_NAMES:
        if rows and all(name in row["features"] for row in rows):
            present.append(name)
    return tuple(present)


def _feature_blockers(names) -> list[str]:
    missing = [name for name in FEATURE_NAMES if name not in names]
    if not missing:
        return []
    return [
        "missing_" + name + ": " + FULL_P2_NOTE
        for name in missing
    ]


def _attach_features(rows, *, feature_rows, sample_rows, index_rows) -> dict:
    feature_map, n_unmatched = _index_feature_rows(feature_rows, rows)
    sample_map = _index_by_request(sample_rows)
    index_map = _index_by_request(index_rows)
    disagreements = 0
    sources = []
    dropped_labels = []
    for row in rows:
        payload = {}
        inline = row.get("inline_features") or {}
        if isinstance(inline, dict):
            payload.update(inline)
        external = feature_map.get(row["request_id"])
        if external:
            payload.update(external)
        logits = payload.get("logits")
        cleaned, dropped = _clean_feature_payload(payload)
        dropped_labels.extend(dropped)
        if "history_length" not in cleaned:
            hint, source, disagreement = _history_length_hint(row, sample_map, index_map)
            disagreements += disagreement
            if hint is not None:
                cleaned["history_length"] = float(hint)
                sources.append(source)
            elif source:
                sources.append(source)
        else:
            sources.append("features")
        dropped_labels.extend(_apply_logits(cleaned, logits))
        row["features"] = cleaned
    source_name = _single_source(sources)
    return {
        "history_length_source": source_name,
        "history_length_disagreement_n": disagreements,
        "dropped_label_features": sorted(set(dropped_labels)),
        "n_feature_rows": 0 if feature_rows is None else len(list(feature_rows)),
        "n_unmatched_feature_rows": n_unmatched,
        "n_sample_rows": 0 if sample_rows is None else len(list(sample_rows)),
        "n_predict_index_rows": 0 if index_rows is None else len(list(index_rows)),
        "invented_logits": False,
        "note": FULL_P2_NOTE,
    }


def _history_length_hint(row, sample_map, index_map):
    samples = sample_map.get(row["request_id"])
    indexed = index_map.get(row["request_id"])
    sample_length = None
    index_length = None
    history = None if samples is None else samples.get("history")
    if history is not None and not isinstance(history, (str, bytes)):
        sample_length = len(history)
    if indexed is not None and indexed.get("n_history") is not None:
        index_length = int(indexed["n_history"])
    position = row.get("target_position")
    disagreement = 0
    if sample_length is not None and position is not None and int(position) != int(sample_length):
        disagreement = 1
    if sample_length is not None:
        return sample_length, "eval_samples", disagreement
    own = row.get("inline_history_length")
    if own is not None:
        if position is not None and int(position) != int(own):
            disagreement = 1
        return int(own), "per_request_history", disagreement
    if index_length is not None:
        return index_length, "predict_index", disagreement
    if position is not None:
        return int(position), "target_position", disagreement
    return None, None, disagreement


def _single_source(sources: list[str]):
    unique = sorted({source for source in sources if source})
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + ",".join(unique)


def _index_feature_rows(feature_rows, rows):
    if not feature_rows:
        return {}, 0
    known = {row["request_id"] for row in rows}
    mapped = {}
    unmatched = 0
    for raw in feature_rows:
        request_id = raw.get("request_id")
        if request_id is None or str(request_id) not in known:
            unmatched += 1
            continue
        mapped[str(request_id)] = _feature_payload(raw)
    return mapped, unmatched


def _feature_payload(row: dict) -> dict:
    payload = {}
    nested = row.get("features")
    if isinstance(nested, dict):
        payload.update(nested)
    for name in list(FEATURE_NAMES) + ["logits", *sorted(LABEL_FEATURE_KEYS)]:
        if name in row and name not in payload:
            payload[name] = row[name]
    return payload


def _clean_feature_payload(payload: dict) -> tuple[dict, list[str]]:
    """Keep numeric ridge columns. Label columns and raw logits stay out of the matrix."""
    dropped = []
    cleaned = {}
    for key, value in payload.items():
        if key == "logits":
            continue
        if _is_label_key(key):
            dropped.append(str(key))
            continue
        if key in FEATURE_NAMES and _is_number(value):
            cleaned[key] = float(value)
    return cleaned, dropped


def _apply_logits(cleaned: dict, logits) -> list[str]:
    """Fill entropy, gap, and top1 from supplied logits after history length is known.

    Logits are not synthesized. A vector that arrived without any history-length
    source is reported and left unused.
    """
    if logits is None or any(name in cleaned for name in LOGIT_FEATURES):
        return []
    history_length = cleaned.get("history_length")
    if history_length is None:
        return ["logits_without_history_length"]
    extracted = extract_features(int(history_length), logits)
    for name in LOGIT_FEATURES:
        cleaned[name] = float(extracted[name])
    cleaned.setdefault("history_length", float(extracted["history_length"]))
    return []


def _is_label_key(key: str) -> bool:
    text = str(key)
    if text in LABEL_FEATURE_KEYS:
        return True
    if text.startswith("ndcg") or text.startswith("recall") or text.startswith("rank"):
        return True
    return False


def _is_number(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    return False


def _index_by_request(rows):
    if not rows:
        return {}
    mapped = {}
    for row in rows:
        if row.get("request_id") is None:
            continue
        mapped[str(row["request_id"])] = row
    return mapped


def _normalize_request(row: dict) -> dict:
    if "user_id" not in row:
        raise ValueError("per_request row is missing user_id")
    user_id = str(row["user_id"])
    split_assumed = not row.get("split")
    split = "val" if split_assumed else canonical_split(str(row["split"]))
    position = row.get("target_position")
    if position is not None:
        position = int(position)
    request_id = row.get("request_id")
    if not request_id:
        request_id = f"{split}:{user_id}:{0 if position is None else position}"
    ndcg20 = request_ndcg(row, SMALL_BEAM)
    ndcg40 = request_ndcg(row, REFERENCE_BEAM)
    ndcg60 = request_ndcg(row, LARGE_BEAM)
    computed = float(ndcg60 - ndcg20)
    explicit = row.get("delta_ndcg@20", row.get("delta_ndcg"))
    if explicit is not None and abs(float(explicit) - computed) > 1e-6:
        raise ValueError(f"request {request_id} delta_ndcg@20 does not match beam 60 minus beam 20")
    inline = row.get("features") if isinstance(row.get("features"), dict) else {}
    history = row.get("history")
    inline_history_length = None
    if history is not None and not isinstance(history, (str, bytes)):
        inline_history_length = len(history)
    return {
        "request_id": str(request_id),
        "user_id": user_id,
        "split": split,
        "split_assumed": split_assumed,
        "target_position": position,
        "ndcg20": float(ndcg20),
        "ndcg40": float(ndcg40),
        "ndcg60": float(ndcg60),
        "delta": computed,
        "inline_features": dict(inline),
        "inline_history_length": inline_history_length,
        "features": {},
    }


def _reject_duplicate_ids(rows) -> None:
    seen = set()
    for row in rows:
        if row["request_id"] in seen:
            raise ValueError(f"duplicate request_id {row['request_id']}")
        seen.add(row["request_id"])


def _id_hash(user_ids) -> str:
    text = "\n".join(sorted(str(user_id) for user_id in user_ids))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_seeds(seed, seeds_text, config_seed: int) -> list[int]:
    if seeds_text and seed is not None:
        raise ValueError("pass --seed or --seeds, not both")
    if seeds_text:
        parsed = []
        for part in str(seeds_text).split(","):
            token = part.strip()
            if token:
                parsed.append(int(token))
        if not parsed:
            raise ValueError("--seeds did not contain an integer")
        return parsed
    if seed is not None:
        return [int(seed)]
    return [int(config_seed)]


def _load_summary(per_request: str, summary: str | None):
    if summary:
        return _read_summary(summary), "argument"
    sibling = Path(per_request).with_name("summary.json")
    if sibling.is_file():
        return _read_summary(str(sibling)), "sibling"
    return None, None


def _read_summary(path: str) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("summary JSON must be an object")
    return document


def _error(command: str, message: str) -> dict:
    return {
        "command": command,
        "ok": False,
        "exit_code": 1,
        "status": "error",
        "scripted": False,
        "message": message,
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_recommender_started": False,
        "invented_logits": False,
        "p2_worth_doing": None,
    }
