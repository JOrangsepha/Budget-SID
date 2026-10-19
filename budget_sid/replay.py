"""Offline replay of beam budgets on leak-safe per-request scores.

P2.1 pre-declares an upgrade fraction of validation split B and gives every
non-oracle policy that same number of beam-60 slots. The declared average
beam is ``20 + 40 * fraction``. The quota is the floor of that fraction. It
is not ``min(n_positive, ceiling)``.

The oracle upper bound is reported beside the curve. It upgrades positive-Δ
requests only and keeps its own mean beam (the compute it actually used). It
does not fill leftover quota with negative-Δ requests.

Primary metrics are final nDCG@20 and the net change versus all-beam 20.
``n_positive_captured`` and ``negative_loss`` are explanatory. Seeds 42, 43,
and 44 reshuffle the A/B split and the random draw. They do not retrain the
recommender.

This module does not call GRID, does not time decoding, and does not claim
an online benefit-aware result. ``per_request.jsonl`` has no first-step
logits. Missing logit features are reported. They are not invented. P3 is
not started.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from budget_sid.analyze_benefit import analyze_per_request_rows
from budget_sid.benefit import (
    HGB_MAX_DEPTH,
    HGB_MAX_ITER,
    fit_benefit_predictor,
    n_large_for_budget,
    try_fit_hist_gradient_boosting,
)
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
# 0%, 1%, 5%, 10%, 25%, 50% of requests upgraded from beam 20 to beam 60.
# Average beam = 20 + 40 * fraction. 50% is the beam-40 ceiling.
DEFAULT_UPGRADE_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
QUOTA_MODE_FIXED_FRACTION = "fixed_fraction"
QUOTA_MODE_POSITIVE_COUNT = "positive_count"
PRIMARY_METRICS = ("mean_ndcg@20", "delta_vs_all_beam20")
EXPLANATORY_METRICS = ("n_positive_captured", "negative_loss")
POLICY_NAMES = ("random", "entropy", "benefit_aware", "benefit_aware_hgb")
QUOTA_IDENTITY = (
    "declared_mean_beam = 20 + 40 * upgrade_fraction; "
    "n_quota = floor(upgrade_fraction * n_B); "
    "realized_mean_beam = 20 + 40 * (n_quota / n_B). "
    "Floor keeps the realized average beam at or under the declaration. "
    "At fraction 0.5, n_quota matches n_large_for_budget."
)
QUOTA_NOTE = (
    "P2.1 pre-declares the upgrade fraction on validation split B. "
    "Random, entropy, and benefit-aware (ridge, and HistGradientBoosting when "
    "scikit-learn is installed) each upgrade n_quota = floor(fraction * n_B) "
    "requests from beam 20 to beam 60. That count is the same for every policy "
    "that runs. It is not derived from B's positive-Δ count. "
    "The oracle upper bound upgrades positive-Δ requests only, highest delta "
    "first, and stops when the positives or the fraction quota run out. "
    "Leftover slots stay at beam 20. The oracle mean beam is the compute it "
    "actually used."
)
LEGACY_QUOTA_NOTE = (
    "Legacy narrow diagnostic. n_quota = min(n_positive on B, n_budget_ceiling). "
    "Opt in with --quota-mode positive_count. The default protocol is the "
    "pre-declared upgrade fractions."
)
SEED_NOTE = (
    "Seeds 42, 43, and 44 reshuffle the validation A/B user split and the "
    "random-policy scores. They are not independent recommender retrainings. "
    "The Beauty checkpoint stays the 2026-10-17 early stop."
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
PREDICTOR_NOTE = (
    "MSE is a fit diagnostic. Primary metrics are final mean nDCG@20 and net "
    "Δ versus all-beam 20. n_positive_captured and negative_loss are explanatory."
)


def declared_mean_beam(fraction: float) -> float:
    """Average beam of a pre-declared upgrade fraction: ``20 + 40 * fraction``."""
    fraction = _check_fraction(fraction)
    return float(SMALL_BEAM + (LARGE_BEAM - SMALL_BEAM) * fraction)


def n_quota_for_fraction(n_requests: int, fraction: float) -> int:
    """How many requests move 20 → 60 at a pre-declared fraction.

    ``floor`` matches ``n_large_for_budget`` at fraction 0.5 and keeps the
    realized average beam from exceeding ``20 + 40 * fraction``.
    """
    fraction = _check_fraction(fraction)
    n_requests = int(n_requests)
    if n_requests < 0:
        raise ValueError("n_requests must be non-negative")
    return int(np.floor(fraction * n_requests + 1e-12))


def realized_mean_beam(n_requests: int, n_large: int) -> float:
    """Average beam after upgrading exactly ``n_large`` requests to beam 60."""
    n_requests = int(n_requests)
    n_large = int(n_large)
    if n_requests <= 0:
        raise ValueError("n_requests must be positive")
    if n_large < 0 or n_large > n_requests:
        raise ValueError("n_large must lie between 0 and n_requests")
    return float((n_large * LARGE_BEAM + (n_requests - n_large) * SMALL_BEAM) / n_requests)


def n_quota_from_positive_count(n_requests: int, n_positive: int) -> int:
    """Legacy narrow diagnostic: ``min(n_positive, beam-40 ceiling)``.

    ``replay_records`` calls this only when ``quota_mode="positive_count"``.
    The default protocol uses ``n_quota_for_fraction``.
    """
    if int(n_positive) < 0:
        raise ValueError("n_positive must be non-negative")
    n_budget = n_large_for_budget(int(n_requests), SMALL_BEAM, LARGE_BEAM, REFERENCE_BEAM)
    return int(min(int(n_positive), int(n_budget)))


def parse_upgrade_fractions(text: str | None) -> tuple[float, ...]:
    """Parse a CLI fraction list.

    Tokens in ``[0, 1]`` are fractions. A trailing ``%``, or any token above 1,
    reads the whole list as percents (``1,5,10`` → 1%, 5%, 10%). ``1`` with no
    percent sign is the fraction 1 (upgrade every request). ``1%`` is 0.01.
    """
    if text is None or str(text).strip() == "":
        return DEFAULT_UPGRADE_FRACTIONS
    raw = []
    saw_percent = False
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        if token.endswith("%"):
            saw_percent = True
            token = token[:-1].strip()
        raw.append(float(token))
    if not raw:
        raise ValueError("upgrade fractions are empty")
    if saw_percent or any(value > 1.0 for value in raw):
        raw = [value / 100.0 for value in raw]
    return _dedupe_fractions(raw)


def coerce_upgrade_fractions(value) -> tuple[float, ...]:
    """Accept the default, a CLI string, or a sequence of fractions in ``[0, 1]``."""
    if value is None:
        return DEFAULT_UPGRADE_FRACTIONS
    if isinstance(value, str):
        return parse_upgrade_fractions(value)
    return _dedupe_fractions([float(item) for item in value])


def score_allocation(rows, selected_ids) -> dict:
    """Primary nDCG and explanatory capture for one assignment on split B.

    ``delta_vs_all_beam20`` is the per-request mean of (chosen beam nDCG minus
    beam-20 nDCG). ``negative_loss`` is the sum of ``-Δ`` over upgraded
    requests whose Δ is negative. It is mass, and it is explanatory. The
    primary delta already includes that loss in the mean.
    """
    selected = {str(request_id) for request_id in selected_ids}
    n = len(rows)
    if n == 0:
        raise ValueError("allocation needs at least one request")
    ndcg_sum = 0.0
    beam20_sum = 0.0
    n_positive_captured = 0
    negative_loss = 0.0
    n_negative_upgraded = 0
    for row in rows:
        delta = float(row["delta"])
        beam20_sum += float(row["ndcg20"])
        chosen = str(row["request_id"]) in selected
        ndcg_sum += float(row["ndcg60"] if chosen else row["ndcg20"])
        if not chosen:
            continue
        if delta > 0.0:
            n_positive_captured += 1
        elif delta < 0.0:
            negative_loss += -delta
            n_negative_upgraded += 1
    mean_ndcg = ndcg_sum / n
    all_beam20 = beam20_sum / n
    n_large = len(selected)
    return {
        "n": n,
        "n_large": n_large,
        "mean_beam": realized_mean_beam(n, n_large),
        "primary": {
            "mean_ndcg@20": float(mean_ndcg),
            "delta_vs_all_beam20": float(mean_ndcg - all_beam20),
        },
        "explanatory": {
            "n_positive_captured": int(n_positive_captured),
            "negative_loss": float(negative_loss),
            "n_negative_upgraded": int(n_negative_upgraded),
            "role": "explanatory_only",
        },
    }


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


def select_positive_oracle(rows, n_quota: int) -> list[str]:
    """Upgrade the highest positive-Δ requests, and nothing else.

    Spend is ``min(n_positive, n_quota)``. Non-positive rows are not passed
    to the ranker, so a larger quota cannot pull a negative-Δ request in.
    """
    positive = [row for row in rows if float(row["delta"]) > 0.0]
    if not positive or int(n_quota) <= 0:
        return []
    return select_exact_quota(
        [float(row["delta"]) for row in positive],
        [row["request_id"] for row in positive],
        min(len(positive), int(n_quota)),
    )


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
    upgrade_fractions=None,
    quota_mode: str = QUOTA_MODE_FIXED_FRACTION,
) -> dict:
    """Fit on validation users A and score the shared-quota curve on users B."""
    if user_protocol not in {"auto", *USER_PROTOCOLS}:
        raise ValueError(f"unknown user_protocol {user_protocol!r}")
    mode = _normalize_quota_mode(quota_mode)
    if mode == QUOTA_MODE_POSITIVE_COUNT and upgrade_fractions is not None:
        raise ValueError(
            "positive_count quota does not take upgrade fractions. "
            "Omit fractions, or use quota_mode='fixed_fraction'."
        )
    fractions = None if mode == QUOTA_MODE_POSITIVE_COUNT else coerce_upgrade_fractions(upgrade_fractions)
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
    fitted = _fit_predictors(
        train_rows,
        budget_rows,
        names=names,
        alpha=alpha,
        seed=seed,
        blockers=feature_report["full_p2_blockers"],
    )
    curve, assignments = _score_curve(
        train_rows,
        budget_rows,
        names=names,
        fractions=fractions,
        quota_mode=mode,
        seed=seed,
        fitted=fitted,
    )
    if mode == QUOTA_MODE_FIXED_FRACTION and any(point["quota_derived_from_positive_count"] for point in curve):
        raise RuntimeError("fixed-fraction replay derived a quota from the positive count")
    context = _split_base(budget_rows)
    premise = analyze_per_request_rows(rows, summary=summary)
    derived = mode == QUOTA_MODE_POSITIVE_COUNT
    payload = {
        "protocol": "p2.1" if mode == QUOTA_MODE_FIXED_FRACTION else "p2_narrow_positive_count",
        "seed": int(seed),
        "seed_note": SEED_NOTE,
        "seeds_retrain_recommender": False,
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
        "quota_mode": mode,
        "quota_derived_from_positive_count": derived,
        "quota_note": LEGACY_QUOTA_NOTE if derived else QUOTA_NOTE,
        "quota_identity": QUOTA_IDENTITY,
        "upgrade_fractions": None if fractions is None else [float(value) for value in fractions],
        "primary_metrics": list(PRIMARY_METRICS),
        "explanatory_metrics": list(EXPLANATORY_METRICS),
        "explanatory_metrics_are_not_primary": True,
        "split_b": context,
        "curve": curve,
        "predictor": fitted["predictor_info"],
        "features": feature_report,
        "validation_premise": _premise_view(premise),
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_recommender_started": False,
        "invented_logits": False,
        "p3_started": False,
        "boundary_features_dumped": False,
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
    fractions: str | None = None,
    quota_mode: str = QUOTA_MODE_FIXED_FRACTION,
) -> dict:
    command = "replay_budgets"
    try:
        mode = _normalize_quota_mode(quota_mode)
        if mode == QUOTA_MODE_POSITIVE_COUNT and fractions not in (None, ""):
            raise ValueError(
                "--quota-mode positive_count is the legacy narrow diagnostic and "
                "does not take --fractions. Drop --fractions, or keep the default "
                "quota mode."
            )
        parsed_fractions = None if mode == QUOTA_MODE_POSITIVE_COUNT else parse_upgrade_fractions(fractions)
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
                upgrade_fractions=parsed_fractions,
                quota_mode=mode,
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
                "p3_started": False,
                "seeds_retrain_recommender": False,
                "seed_note": SEED_NOTE,
            }
        )
        return payload
    except (OSError, ValueError, RuntimeError, TestLeakError, KeyError, TypeError) as exc:
        return _error(command, str(exc))


def _public_payload(payloads: list[dict]) -> dict:
    stripped = [_without_assignments(item) for item in payloads]
    if len(stripped) == 1:
        return dict(stripped[0])
    first = stripped[0]
    return {
        "protocol": first["protocol"],
        "seeds": [item["seed"] for item in stripped],
        "seed_note": first["seed_note"],
        "seeds_retrain_recommender": False,
        "by_seed": stripped,
        "curve": _aggregate_curve(stripped),
        "upgrade_fractions": first["upgrade_fractions"],
        "quota_mode": first["quota_mode"],
        "quota_derived_from_positive_count": first["quota_derived_from_positive_count"],
        "quota_note": first["quota_note"],
        "quota_identity": first["quota_identity"],
        "primary_metrics": list(first["primary_metrics"]),
        "explanatory_metrics": list(first["explanatory_metrics"]),
        "explanatory_metrics_are_not_primary": True,
        "features": first["features"],
        "predictor": first["predictor"],
        "validation_premise": first["validation_premise"],
        "user_protocol": first["user_protocol"],
        "mse_is_not_the_allocation_metric": True,
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_recommender_started": False,
        "invented_logits": False,
        "p3_started": False,
        "boundary_features_dumped": False,
    }


def _aggregate_curve(payloads: list[dict]) -> list[dict]:
    width = len(payloads[0]["curve"])
    if any(len(item["curve"]) != width for item in payloads):
        raise ValueError("seeds produced curves of different lengths")
    aggregated = []
    for index in range(width):
        points = [item["curve"][index] for item in payloads]
        fractions = [point["upgrade_fraction"] for point in points]
        if any(value != fractions[0] for value in fractions):
            raise ValueError("seeds produced different upgrade fractions")
        policies = {}
        for name in POLICY_NAMES:
            policies[name] = _aggregate_measured([point["policies"][name] for point in points])
        aggregated.append(
            {
                "upgrade_fraction": points[0]["upgrade_fraction"],
                "fraction_key": points[0]["fraction_key"],
                "declared_mean_beam": points[0]["declared_mean_beam"],
                "n_quota": _mean_std(point["n_quota"] for point in points),
                "realized_mean_beam": _mean_std(point["realized_mean_beam"] for point in points),
                "quota_source": points[0]["quota_source"],
                "quota_derived_from_positive_count": points[0]["quota_derived_from_positive_count"],
                "policies": policies,
                "oracle_upper_bound": _aggregate_oracle([point["oracle_upper_bound"] for point in points]),
            }
        )
    return aggregated


def _aggregate_oracle(rows: list[dict]) -> dict:
    base = _aggregate_measured(rows)
    if any(row.get("status") != "ran" for row in rows):
        return base
    base["forced_negative_delta_upgrades"] = _mean_std(row["forced_negative_delta_upgrades"] for row in rows)
    base["same_compute_as_policies"] = all(bool(row["same_compute_as_policies"]) for row in rows)
    base["filled_quota_with_nonpositive"] = any(bool(row["filled_quota_with_nonpositive"]) for row in rows)
    return base


def _aggregate_measured(rows: list[dict]) -> dict:
    statuses = {row.get("status") for row in rows}
    if statuses != {"ran"}:
        copied = {
            "status": rows[0]["status"] if len(statuses) == 1 else "mixed",
            "uses_labels": rows[0]["uses_labels"],
            "reason": rows[0]["reason"],
            "n_large": None,
            "mean_beam": None,
            "primary": {"mean_ndcg@20": None, "delta_vs_all_beam20": None},
            "explanatory": {
                "n_positive_captured": None,
                "negative_loss": None,
                "n_negative_upgraded": None,
                "role": "explanatory_only",
            },
        }
        if len(statuses) != 1:
            copied["reason"] = "seeds disagreed: " + ",".join(sorted(str(status) for status in statuses))
        return copied
    return {
        "status": "ran",
        "uses_labels": rows[0]["uses_labels"],
        "reason": rows[0]["reason"],
        "n_large": _mean_std(row["n_large"] for row in rows),
        "mean_beam": _mean_std(row["mean_beam"] for row in rows),
        "primary": {
            "mean_ndcg@20": _mean_std(row["primary"]["mean_ndcg@20"] for row in rows),
            "delta_vs_all_beam20": _mean_std(row["primary"]["delta_vs_all_beam20"] for row in rows),
        },
        "explanatory": {
            "n_positive_captured": _mean_std(row["explanatory"]["n_positive_captured"] for row in rows),
            "negative_loss": _mean_std(row["explanatory"]["negative_loss"] for row in rows),
            "n_negative_upgraded": _mean_std(row["explanatory"]["n_negative_upgraded"] for row in rows),
            "role": "explanatory_only",
        },
    }


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
    curve_path = root / "curve_summary.json"
    summary_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    curve_path.write_text(json.dumps(_curve_summary(public, payloads), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = {"replay_summary": str(summary_path), "curve_summary": str(curve_path)}
    for payload in payloads:
        assignments = payload.get("assignments") or []
        path = root / f"assignments_seed{payload['seed']}.jsonl"
        write_jsonl(path, assignments)
        written[f"assignments_seed{payload['seed']}"] = str(path)
    return written


def _curve_summary(public: dict, payloads: list[dict]) -> dict:
    points = []
    for point in public["curve"]:
        policies = {}
        for name, policy in point["policies"].items():
            policies[name] = {
                "status": policy["status"],
                "n_large": policy["n_large"],
                "mean_beam": policy["mean_beam"],
                "primary": policy["primary"],
                "explanatory": policy["explanatory"],
            }
        oracle = point["oracle_upper_bound"]
        points.append(
            {
                "upgrade_fraction": point["upgrade_fraction"],
                "fraction_key": point["fraction_key"],
                "declared_mean_beam": point["declared_mean_beam"],
                "n_quota": point["n_quota"],
                "realized_mean_beam": point["realized_mean_beam"],
                "quota_source": point["quota_source"],
                "quota_derived_from_positive_count": point["quota_derived_from_positive_count"],
                "policies": policies,
                "oracle_upper_bound": {
                    "status": oracle["status"],
                    "n_large": oracle["n_large"],
                    "mean_beam": oracle["mean_beam"],
                    "primary": oracle["primary"],
                    "explanatory": oracle["explanatory"],
                    "same_compute_as_policies": oracle.get("same_compute_as_policies"),
                    "forced_negative_delta_upgrades": oracle.get("forced_negative_delta_upgrades"),
                    "filled_quota_with_nonpositive": oracle.get("filled_quota_with_nonpositive"),
                    "reason": oracle.get("reason"),
                },
            }
        )
    document = {
        "protocol": public["protocol"],
        "quota_mode": public["quota_mode"],
        "quota_derived_from_positive_count": public["quota_derived_from_positive_count"],
        "quota_identity": public["quota_identity"],
        "quota_note": public["quota_note"],
        "primary_metrics": list(public["primary_metrics"]),
        "explanatory_metrics": list(public["explanatory_metrics"]),
        "explanatory_metrics_are_not_primary": True,
        "seed_note": public["seed_note"],
        "seeds_retrain_recommender": False,
        "seeds": [item["seed"] for item in payloads],
        "upgrade_fractions": public["upgrade_fractions"],
        "curve": points,
        "p3_started": False,
        "note": (
            "Primary metrics are final mean nDCG@20 and net Δ versus all-beam 20. "
            "n_positive_captured and negative_loss are explanatory only. "
            "Policy n_quota is the shared upgrade count. "
            "Oracle mean beam is the positive-Δ compute actually used."
        ),
    }
    if "split_b" in public:
        base = public["split_b"]
        document["split_b"] = {
            "n": base["n"],
            "n_positive": base["n_positive"],
            "n_negative": base["n_negative"],
            "all_20_ndcg@20": base["all_20_ndcg@20"],
            "all_40_ndcg@20": base["all_40_ndcg@20"],
            "all_60_ndcg@20": base["all_60_ndcg@20"],
        }
    return document


def _fit_predictors(train_rows, budget_rows, *, names, alpha, seed, blockers):
    ridge_info = {
        "status": "blocked",
        "model": "ridge",
        "feature_names": list(names),
        "alpha": float(alpha),
        "n_train_rows": len(train_rows),
        "split_b_mse": None,
        "note": PREDICTOR_NOTE,
    }
    hgb_info = {
        "status": "blocked",
        "model": "HistGradientBoostingRegressor",
        "max_depth": HGB_MAX_DEPTH,
        "max_iter": HGB_MAX_ITER,
        "same_features_as_ridge": True,
        "boundary_features": False,
        "feature_names": list(names),
        "split_b_mse": None,
        "reason": "No non-label feature is present on every validation row. " + FULL_P2_NOTE,
    }
    ridge_pred = None
    hgb_pred = None
    benefit_blockers = list(blockers)
    if names:
        train_matrix = stack_named_features([row["features"] for row in train_rows], names)
        budget_matrix = stack_named_features([row["features"] for row in budget_rows], names)
        train_target = np.array([row["delta"] for row in train_rows], dtype=np.float64)
        budget_target = np.array([row["delta"] for row in budget_rows], dtype=np.float64)
        predictor = fit_benefit_predictor(train_matrix, train_target, alpha=float(alpha))
        predictor.feature_names = tuple(names)
        ridge_pred = np.asarray(predictor.predict(budget_matrix), dtype=np.float64)
        ridge_info = {
            "status": "fit",
            "model": "ridge",
            "feature_names": list(names),
            "alpha": float(alpha),
            "n_train_rows": len(train_rows),
            "split_b_mse": float(np.mean((ridge_pred - budget_target) ** 2)),
            "note": PREDICTOR_NOTE,
        }
        model, hgb_info = try_fit_hist_gradient_boosting(
            train_matrix,
            train_target,
            max_depth=HGB_MAX_DEPTH,
            max_iter=HGB_MAX_ITER,
            random_state=int(seed),
        )
        hgb_info = dict(hgb_info)
        hgb_info["feature_names"] = list(names)
        hgb_info["same_features_as_ridge"] = True
        hgb_info["boundary_features"] = False
        if model is not None:
            hgb_pred = np.asarray(model.predict(budget_matrix), dtype=np.float64)
            hgb_info["split_b_mse"] = float(np.mean((hgb_pred - budget_target) ** 2))
            hgb_info["status"] = "fit"
    return {
        "ridge_pred": ridge_pred,
        "hgb_pred": hgb_pred,
        "benefit_blockers": benefit_blockers,
        "predictor_info": {"ridge": ridge_info, "hist_gradient_boosting": hgb_info},
    }


def _score_curve(train_rows, budget_rows, *, names, fractions, quota_mode, seed, fitted):
    ids = [row["request_id"] for row in budget_rows]
    n = len(budget_rows)
    rng = np.random.default_rng(int(seed))
    random_scores = rng.random(n)
    entropy_scores = None
    if "first_step_entropy" in names:
        entropy_scores = [float(row["features"]["first_step_entropy"]) for row in budget_rows]
    if quota_mode == QUOTA_MODE_POSITIVE_COUNT:
        n_positive = sum(1 for row in budget_rows if float(row["delta"]) > 0.0)
        specs = [
            {
                "upgrade_fraction": None,
                "fraction_key": "positive_count",
                "n_quota": n_quota_from_positive_count(n, n_positive),
                "declared_mean_beam": None,
                "quota_source": QUOTA_MODE_POSITIVE_COUNT,
                "quota_derived_from_positive_count": True,
            }
        ]
    else:
        specs = []
        for fraction in fractions:
            n_quota = n_quota_for_fraction(n, fraction)
            specs.append(
                {
                    "upgrade_fraction": float(fraction),
                    "fraction_key": _fraction_key(fraction),
                    "n_quota": n_quota,
                    "declared_mean_beam": declared_mean_beam(fraction),
                    "quota_source": QUOTA_MODE_FIXED_FRACTION,
                    "quota_derived_from_positive_count": False,
                }
            )
    curve = []
    selections = []
    for spec in specs:
        point, selected = _score_point(
            budget_rows,
            ids=ids,
            random_scores=random_scores,
            entropy_scores=entropy_scores,
            ridge_pred=fitted["ridge_pred"],
            hgb_pred=fitted["hgb_pred"],
            blockers=fitted["benefit_blockers"],
            spec=spec,
        )
        curve.append(point)
        selections.append(selected)
    assignments = _assignment_rows(
        train_rows,
        budget_rows,
        ids=ids,
        ridge_pred=fitted["ridge_pred"],
        hgb_pred=fitted["hgb_pred"],
        selections=selections,
    )
    return curve, assignments


def _score_point(rows, *, ids, random_scores, entropy_scores, ridge_pred, hgb_pred, blockers, spec):
    n_quota = int(spec["n_quota"])
    random_ids = select_exact_quota(random_scores, ids, n_quota)
    policies = {
        "random": _policy_from_ids(
            rows,
            random_ids,
            status="ran",
            uses_labels=False,
            reason="exact sample of the shared quota; one ranking per seed, prefix at each fraction",
        )
    }
    if entropy_scores is None:
        entropy_ids = None
        policies["entropy"] = _skipped_policy(
            "first_step_entropy is not in the joined features. " + FULL_P2_NOTE
        )
    else:
        entropy_ids = select_exact_quota(entropy_scores, ids, n_quota)
        policies["entropy"] = _policy_from_ids(
            rows,
            entropy_ids,
            status="ran",
            uses_labels=False,
            reason="highest first_step_entropy under the shared quota",
        )
    if ridge_pred is None:
        benefit_ids = None
        policies["benefit_aware"] = _skipped_policy(
            "No non-label feature is present on every validation row. " + FULL_P2_NOTE,
            status="blocked",
        )
    else:
        benefit_ids = select_exact_quota(ridge_pred, ids, n_quota)
        policies["benefit_aware"] = _policy_from_ids(
            rows,
            benefit_ids,
            status="ran",
            uses_labels=False,
            reason="highest ridge predicted delta on the features that were actually joined",
            extra_blockers=list(blockers),
        )
    if hgb_pred is None:
        hgb_ids = None
        reason = "HistGradientBoostingRegressor was not fit on the joined features."
        status = "blocked" if ridge_pred is None else "unavailable"
        policies["benefit_aware_hgb"] = _skipped_policy(reason, status=status)
    else:
        hgb_ids = select_exact_quota(hgb_pred, ids, n_quota)
        policies["benefit_aware_hgb"] = _policy_from_ids(
            rows,
            hgb_ids,
            status="ran",
            uses_labels=False,
            reason=(
                f"highest HistGradientBoosting predicted delta "
                f"(max_depth={HGB_MAX_DEPTH}, max_iter={HGB_MAX_ITER}) on the same features as ridge"
            ),
            extra_blockers=list(blockers),
        )
    oracle_ids = select_positive_oracle(rows, n_quota)
    oracle = _oracle_from_ids(rows, oracle_ids, n_quota=n_quota)
    for policy in policies.values():
        policy.pop("selected_request_ids", None)
    oracle.pop("selected_request_ids", None)
    n = len(rows)
    point = {
        "upgrade_fraction": spec["upgrade_fraction"],
        "fraction_key": spec["fraction_key"],
        "declared_mean_beam": spec["declared_mean_beam"],
        "n_quota": n_quota,
        "realized_upgrade_fraction": float(n_quota / n),
        "realized_mean_beam": realized_mean_beam(n, n_quota),
        "quota_source": spec["quota_source"],
        "quota_derived_from_positive_count": bool(spec["quota_derived_from_positive_count"]),
        "policies": policies,
        "oracle_upper_bound": oracle,
    }
    selected = {
        "fraction_key": spec["fraction_key"],
        "upgrade_fraction": spec["upgrade_fraction"],
        "random": set(random_ids),
        "entropy": None if entropy_ids is None else set(entropy_ids),
        "benefit": None if benefit_ids is None else set(benefit_ids),
        "hgb": None if hgb_ids is None else set(hgb_ids),
        "oracle": set(oracle_ids),
    }
    return point, selected


def _policy_from_ids(rows, selected_ids, *, status, uses_labels, reason, extra_blockers=None):
    measured = score_allocation(rows, selected_ids)
    return {
        "status": status,
        "uses_labels": bool(uses_labels),
        "reason": reason,
        "n": measured["n"],
        "n_large": measured["n_large"],
        "mean_beam": measured["mean_beam"],
        "primary": measured["primary"],
        "explanatory": measured["explanatory"],
        "full_p2_blockers": list(extra_blockers or []),
        "selected_request_ids": list(selected_ids),
    }


def _oracle_from_ids(rows, selected_ids, *, n_quota: int):
    selected = set(selected_ids)
    forced = 0
    for row in rows:
        if row["request_id"] in selected and float(row["delta"]) <= 0.0:
            forced += 1
    if forced:
        raise RuntimeError("oracle upgraded a non-positive delta to fill the quota")
    measured = score_allocation(rows, selected_ids)
    n_positive = sum(1 for row in rows if float(row["delta"]) > 0.0)
    expected = min(n_positive, int(n_quota))
    if measured["n_large"] != expected:
        raise RuntimeError("oracle spend does not match min(n_positive, n_quota)")
    return {
        "status": "ran",
        "uses_labels": True,
        "reason": (
            "Upper bound. Upgrades positive-Δ requests only, highest delta first. "
            "Does not fill leftover quota with negative-Δ or zero-Δ requests. "
            "mean_beam is the compute actually used."
        ),
        "n": measured["n"],
        "n_large": measured["n_large"],
        "mean_beam": measured["mean_beam"],
        "primary": measured["primary"],
        "explanatory": measured["explanatory"],
        "forced_negative_delta_upgrades": 0,
        "filled_quota_with_nonpositive": False,
        "same_compute_as_policies": measured["n_large"] == int(n_quota),
        "full_p2_blockers": [],
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
        "primary": {"mean_ndcg@20": None, "delta_vs_all_beam20": None},
        "explanatory": {
            "n_positive_captured": None,
            "negative_loss": None,
            "n_negative_upgraded": None,
            "role": "explanatory_only",
        },
        "full_p2_blockers": [reason],
        "selected_request_ids": None,
    }


def _assignment_rows(train_rows, budget_rows, *, ids, ridge_pred, hgb_pred, selections):
    ridge_by_id = {}
    hgb_by_id = {}
    if ridge_pred is not None:
        for request_id, score in zip(ids, ridge_pred):
            ridge_by_id[request_id] = float(score)
    if hgb_pred is not None:
        for request_id, score in zip(ids, hgb_pred):
            hgb_by_id[request_id] = float(score)
    rows = []
    for group, group_rows in (("A", train_rows), ("B", budget_rows)):
        for row in group_rows:
            request_id = row["request_id"]
            on_b = group == "B"
            by_fraction = {}
            for selected in selections:
                if not on_b:
                    by_fraction[selected["fraction_key"]] = None
                    continue
                by_fraction[selected["fraction_key"]] = {
                    "selected_random": request_id in selected["random"],
                    "selected_entropy": None if selected["entropy"] is None else request_id in selected["entropy"],
                    "selected_benefit_aware": None if selected["benefit"] is None else request_id in selected["benefit"],
                    "selected_benefit_aware_hgb": None if selected["hgb"] is None else request_id in selected["hgb"],
                    "selected_oracle": request_id in selected["oracle"],
                }
            rows.append(
                {
                    "request_id": request_id,
                    "user_id": row["user_id"],
                    "group": group,
                    "delta_ndcg@20": row["delta"],
                    "predicted_delta": ridge_by_id.get(request_id),
                    "predicted_delta_hgb": hgb_by_id.get(request_id),
                    "by_fraction": by_fraction,
                }
            )
    return rows


def _split_base(rows) -> dict:
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
        "n_budget_ceiling": int(n_large_for_budget(n, SMALL_BEAM, LARGE_BEAM, REFERENCE_BEAM)),
        "rejected_force_fill_ndcg@20": _force_fill_ndcg(rows),
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


def _check_fraction(fraction: float) -> float:
    value = float(fraction)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"upgrade fraction {fraction!r} is outside [0, 1]")
    return value


def _dedupe_fractions(values) -> tuple[float, ...]:
    unique = []
    seen = set()
    for value in values:
        checked = _check_fraction(value)
        key = round(checked, 10)
        if key in seen:
            continue
        seen.add(key)
        unique.append(checked)
    if not unique:
        raise ValueError("upgrade fractions are empty")
    return tuple(unique)


def _fraction_key(fraction: float) -> str:
    return f"{float(fraction):.4f}"


def _normalize_quota_mode(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {QUOTA_MODE_FIXED_FRACTION, "fixed", "p2.1", "p2_1"}:
        return QUOTA_MODE_FIXED_FRACTION
    if text in {QUOTA_MODE_POSITIVE_COUNT, "positive", "legacy", "narrow"}:
        return QUOTA_MODE_POSITIVE_COUNT
    raise ValueError(
        f"unknown quota_mode {value!r}. Use {QUOTA_MODE_FIXED_FRACTION} or {QUOTA_MODE_POSITIVE_COUNT}."
    )


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
