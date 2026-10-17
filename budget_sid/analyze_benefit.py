"""Offline reading of leak-safe per-request ΔNDCG@20.

The command reports whether a concentrated positive tail would let a
budgeted oracle beat uniform beam 40. It does not fit a benefit model and
it does not claim an online gain.
"""

from __future__ import annotations

import json
from pathlib import Path

from budget_sid.benefit import n_large_for_budget
from budget_sid.constants import LARGE_BEAM, REFERENCE_BEAM, SMALL_BEAM
from budget_sid.eval_samples import read_jsonl
from budget_sid.stop_gate import (
    budgeted_positive_assignment,
    has_concentrated_gain,
    request_ndcg,
)
from budget_sid.synthetic import selective_oracle_counterexample

BEAM_MEAN_TOLERANCE = 1.5e-5
DELTA_MEAN_TOLERANCE = 5e-7
RECALL_TOLERANCE = 5.1e-5


def command_analyze_benefit(
    *,
    per_request: str | None = None,
    summary: str | None = None,
    smoke: bool = False,
    output: str | None = None,
) -> dict:
    command = "analyze_benefit"
    try:
        if per_request:
            rows = read_jsonl(per_request)
            summary_doc = _read_summary(summary) if summary else None
            payload = analyze_per_request_rows(rows, summary=summary_doc)
            payload["population"] = "per_request"
            payload["scripted"] = False
            payload["status"] = "analyzed"
        elif smoke:
            rows = [to_per_request_record(row) for row in selective_oracle_counterexample()]
            payload = analyze_per_request_rows(rows, summary=_synthetic_summary(rows))
            payload["population"] = "synthetic_counterexample"
            payload["scripted"] = True
            payload["status"] = "smoke"
        elif summary:
            summary_doc = _read_summary(summary)
            payload = analyze_summary_only(summary_doc)
            payload["population"] = "summary_only"
            payload["scripted"] = False
            payload["status"] = "summary_only"
        else:
            return _error(
                command,
                "Pass --per-request, or --summary for the aggregate bound, or --smoke for the synthetic counterexample.",
            )
        payload["command"] = command
        payload["ok"] = True
        payload["exit_code"] = 0
        payload["benefit_aware_claimed"] = False
        payload["grid_recommender_started"] = False
        if output:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload["output"] = str(destination)
        return payload
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return _error(command, str(exc))


def analyze_summary_only(summary: dict) -> dict:
    """Say what the beam means can and cannot decide. They cannot clear P2."""
    bound = bound_from_summary(summary)
    bound["per_request_analyzed"] = False
    bound["fraction_positive"] = None
    bound["fraction_zero"] = None
    bound["fraction_negative"] = None
    bound["oracle_ndcg@20"] = None
    bound["p2_worth_doing"] = None
    bound["p2_reasons"] = ["per_request_labels_required"]
    bound["decision_split"] = None
    bound["note"] = (
        "Summary means do not identify the positive tail. "
        "Do not start P2 until analyze_benefit on validation rows of per_request.jsonl "
        "returns p2_worth_doing true. This is not a benefit-aware online gain."
    )
    return bound


def bound_from_summary(summary: dict) -> dict:
    beams = summary["beams"]
    delta = summary["delta_ndcg@20"]
    ndcg20 = float(beams["20"]["ndcg@20"])
    ndcg40 = float(beams["40"]["ndcg@20"])
    ndcg60 = float(beams["60"]["ndcg@20"])
    mean_delta = float(delta["mean"])
    n = int(summary.get("n_requests") or delta["n"])
    gap = ndcg40 - ndcg20
    implied = ndcg60 - ndcg20
    max_delta = None if delta.get("max") is None else float(delta["max"])
    min_delta = None if delta.get("min") is None else float(delta["min"])
    return {
        "n": n,
        "ranking_source": summary.get("ranking_source"),
        "auditable_for_delta_ndcg": summary.get("auditable_for_delta_ndcg"),
        "ndcg@20_beam20": ndcg20,
        "ndcg@20_beam40": ndcg40,
        "ndcg@20_beam60": ndcg60,
        "recall@20_beam20": _summary_recall(beams, "20"),
        "recall@20_beam40": _summary_recall(beams, "40"),
        "recall@20_beam60": _summary_recall(beams, "60"),
        "mean_delta_ndcg@20": mean_delta,
        "min_delta_ndcg@20": min_delta,
        "max_delta_ndcg@20": max_delta,
        "gap_ndcg@20_beam40_minus_beam20": gap,
        "beam60_minus_beam20": implied,
        "reported_mean_minus_beam_difference": mean_delta - implied,
        "uniform_beam60_beats_beam40": bool(ndcg60 > ndcg40),
        "mean_delta_exceeds_beam40_gap": bool(mean_delta > gap),
        "p2_identified_by_summary_means": False,
        "one_max_request_mean_shift": None if max_delta is None or n == 0 else max_delta / n,
        "one_min_request_mean_shift": None if min_delta is None or n == 0 else min_delta / n,
        "benefit_aware_claimed": False,
    }


def analyze_per_request_rows(rows, summary: dict | None = None) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("per_request file has no rows")
    all_stats = _population_stats(rows)
    splits = sorted({row.get("split") for row in rows if row.get("split")})
    by_split = {split: _population_stats([row for row in rows if row.get("split") == split]) for split in splits}
    decision_split, decision_rows, decision_note = _decision_rows(rows)
    if decision_split == "none":
        decision = {
            "split": None,
            "p2_worth_doing": False,
            "p2_reasons": ["no_validation_rows"],
            "note": decision_note,
        }
        concentrated = False
        beats = False
    else:
        decision_stats = _population_stats(decision_rows)
        auditable = None if summary is None else summary.get("auditable_for_delta_ndcg")
        if summary is not None and auditable is None and summary.get("ranking_source") == "synthetic_counterexample":
            auditable = True
        verdict = _p2_verdict(
            concentrated=decision_stats["concentrated_gain"],
            beats=decision_stats["oracle_beats_all_40"],
            auditable=auditable,
        )
        decision = {"split": decision_split, "note": decision_note, **decision_stats, **verdict}
        concentrated = decision_stats["concentrated_gain"]
        beats = decision_stats["oracle_beats_all_40"]
    cross = None if summary is None else cross_check_summary(all_stats, summary)
    if summary is not None and summary.get("auditable_for_delta_ndcg") is False:
        decision["p2_worth_doing"] = False
        reasons = list(decision.get("p2_reasons") or [])
        if "not_auditable_for_delta_ndcg" not in reasons:
            reasons.append("not_auditable_for_delta_ndcg")
        decision["p2_reasons"] = reasons
    payload = {
        "n": all_stats["n"],
        "fraction_positive": all_stats["fraction_positive"],
        "fraction_zero": all_stats["fraction_zero"],
        "fraction_negative": all_stats["fraction_negative"],
        "n_positive": all_stats["n_positive"],
        "n_zero": all_stats["n_zero"],
        "n_negative": all_stats["n_negative"],
        "mean_delta_ndcg@20": all_stats["mean_delta_ndcg@20"],
        "min_delta_ndcg@20": all_stats["min_delta_ndcg@20"],
        "max_delta_ndcg@20": all_stats["max_delta_ndcg@20"],
        "concentrated_gain": all_stats["concentrated_gain"],
        "positive_user_fraction": all_stats["positive_user_fraction"],
        "top_mass_share": all_stats["top_mass_share"],
        "all_20_ndcg@20": all_stats["all_20_ndcg@20"],
        "all_40_ndcg@20": all_stats["all_40_ndcg@20"],
        "all_60_ndcg@20": all_stats["all_60_ndcg@20"],
        "oracle_ndcg@20": all_stats["oracle_ndcg@20"],
        "oracle_mean_beam": all_stats["oracle_mean_beam"],
        "oracle_n_upgraded": all_stats["oracle_n_upgraded"],
        "oracle_n_budget": all_stats["oracle_n_budget"],
        "oracle_beats_all_40": all_stats["oracle_beats_all_40"],
        "rejected_force_fill_ndcg@20": all_stats["rejected_force_fill_ndcg@20"],
        "all_requests_role": "descriptive",
        "decision_split": decision.get("split"),
        "decision_concentrated_gain": decision.get("concentrated_gain", concentrated),
        "decision_oracle_ndcg@20": decision.get("oracle_ndcg@20"),
        "decision_all_40_ndcg@20": decision.get("all_40_ndcg@20"),
        "decision_oracle_beats_all_40": decision.get("oracle_beats_all_40", beats),
        "p2_worth_doing": decision.get("p2_worth_doing"),
        "p2_reasons": list(decision.get("p2_reasons") or []),
        "decision_note": decision.get("note"),
        "by_split": by_split,
        "summary_cross_check": cross,
        "auditable_for_delta_ndcg": None if summary is None else summary.get("auditable_for_delta_ndcg"),
        "per_request_analyzed": True,
        "benefit_aware_claimed": False,
    }
    if (
        summary is not None
        and payload["auditable_for_delta_ndcg"] is True
        and "beams" in summary
        and "delta_ndcg@20" in summary
    ):
        payload["summary_bound"] = bound_from_summary(summary)
    return payload


def _decision_rows(rows):
    labeled = [row for row in rows if row.get("split")]
    if not labeled:
        return "unspecified", list(rows), "No split field. These rows are the whole population, not a held-out test set."
    val_rows = [row for row in rows if row.get("split") == "val"]
    if not val_rows:
        return "none", [], "The file has split labels and no validation rows. Test rows cannot clear P2."
    return "val", val_rows, "P2 uses validation rows only. The all-rows table is descriptive."


def _p2_verdict(*, concentrated: bool, beats: bool, auditable) -> dict:
    if auditable is False:
        return {"p2_worth_doing": False, "p2_reasons": ["not_auditable_for_delta_ndcg"]}
    reasons = []
    if not concentrated:
        reasons.append("positive_tail_not_concentrated")
    if not beats:
        reasons.append("oracle_not_better_than_uniform_beam_40")
    if reasons:
        return {"p2_worth_doing": False, "p2_reasons": reasons}
    if auditable is None:
        return {
            "p2_worth_doing": None,
            "p2_reasons": ["summary_required_to_confirm_leak_safe_labels"],
        }
    return {
        "p2_worth_doing": True,
        "p2_reasons": ["validation_oracle_beats_beam_40_on_a_concentrated_tail"],
    }


def _population_stats(rows) -> dict:
    rows = list(rows)
    deltas = [_delta(row) for row in rows]
    n = len(deltas)
    n_positive = sum(delta > 0.0 for delta in deltas)
    n_negative = sum(delta < 0.0 for delta in deltas)
    n_zero = n - n_positive - n_negative
    concentrated, gain_stats = has_concentrated_gain(deltas)
    assigned = budgeted_positive_assignment(rows)
    recalls = {beam: _mean_optional(rows, beam, "recall@20") for beam in (SMALL_BEAM, REFERENCE_BEAM, LARGE_BEAM)}
    return {
        "n": n,
        "n_positive": n_positive,
        "n_zero": n_zero,
        "n_negative": n_negative,
        "fraction_positive": n_positive / n,
        "fraction_zero": n_zero / n,
        "fraction_negative": n_negative / n,
        "mean_delta_ndcg@20": float(sum(deltas) / n),
        "min_delta_ndcg@20": float(min(deltas)),
        "max_delta_ndcg@20": float(max(deltas)),
        "concentrated_gain": bool(concentrated),
        "positive_user_fraction": gain_stats["positive_user_fraction"],
        "top_mass_share": gain_stats["top_mass_share"],
        "all_20_ndcg@20": assigned["small_ndcg"],
        "all_40_ndcg@20": assigned["reference_ndcg"],
        "all_60_ndcg@20": assigned["large_ndcg"],
        "oracle_ndcg@20": assigned["oracle_ndcg"],
        "oracle_mean_beam": assigned["mean_beam"],
        "oracle_n_upgraded": assigned["n_upgraded"],
        "oracle_n_budget": assigned["n_budget"],
        "oracle_beats_all_40": assigned["beats_reference"],
        "rejected_force_fill_ndcg@20": _force_fill_ndcg(rows),
        "recall@20_beam20": recalls[SMALL_BEAM],
        "recall@20_beam40": recalls[REFERENCE_BEAM],
        "recall@20_beam60": recalls[LARGE_BEAM],
    }


def _delta(row) -> float:
    computed = _ndcg_for_analysis(row, LARGE_BEAM) - _ndcg_for_analysis(row, SMALL_BEAM)
    explicit = row.get("delta_ndcg@20", row.get("delta_ndcg"))
    if explicit is None:
        return computed
    explicit = float(explicit)
    if abs(explicit - computed) > 1e-6:
        request = row.get("request_id", row.get("user_id"))
        raise ValueError(f"request {request} delta_ndcg@20 does not match beam 60 minus beam 20")
    return explicit


def _ndcg_for_analysis(row, beam: int) -> float:
    return request_ndcg(row, beam)


def _mean_optional(rows, beam: int, metric: str):
    values = []
    for row in rows:
        cell = (row.get("beams") or {}).get(str(beam))
        if isinstance(cell, dict) and cell.get(metric) is not None:
            values.append(float(cell[metric]))
    if len(values) != len(rows):
        return None
    return float(sum(values) / len(values))


def _force_fill_ndcg(rows) -> float:
    """The rejected policy: spend the whole large-beam cap, including negative Δ."""
    rows = list(rows)
    deltas = [_delta(row) for row in rows]
    n_budget = n_large_for_budget(len(rows), SMALL_BEAM, LARGE_BEAM, REFERENCE_BEAM)
    order = sorted(range(len(rows)), key=lambda index: (-deltas[index], str(rows[index].get("user_id"))))
    upgraded = set(order[:n_budget])
    total = 0.0
    for index, row in enumerate(rows):
        beam = LARGE_BEAM if index in upgraded else SMALL_BEAM
        total += _ndcg_for_analysis(row, beam)
    return float(total / len(rows))


def cross_check_summary(stats: dict, summary: dict) -> dict:
    if "beams" not in summary or "delta_ndcg@20" not in summary:
        return {"ok": None, "mismatches": [], "skipped": "summary has no beam table to cross-check"}
    mismatches = []
    expected_n = int(summary.get("n_requests") or summary["delta_ndcg@20"]["n"])
    if int(stats["n"]) != expected_n:
        mismatches.append({"field": "n", "per_request": stats["n"], "summary": expected_n})
    _compare(mismatches, "mean_delta_ndcg@20", stats["mean_delta_ndcg@20"], summary["delta_ndcg@20"].get("mean"), DELTA_MEAN_TOLERANCE)
    _compare(mismatches, "min_delta_ndcg@20", stats["min_delta_ndcg@20"], summary["delta_ndcg@20"].get("min"), DELTA_MEAN_TOLERANCE)
    _compare(mismatches, "max_delta_ndcg@20", stats["max_delta_ndcg@20"], summary["delta_ndcg@20"].get("max"), DELTA_MEAN_TOLERANCE)
    for beam, key in (("20", "all_20_ndcg@20"), ("40", "all_40_ndcg@20"), ("60", "all_60_ndcg@20")):
        _compare(
            mismatches,
            f"ndcg@20_beam{beam}",
            stats[key],
            summary["beams"][beam].get("ndcg@20"),
            BEAM_MEAN_TOLERANCE,
        )
        recall_key = f"recall@20_beam{beam}"
        if stats.get(recall_key) is not None:
            _compare(
                mismatches,
                recall_key,
                stats[recall_key],
                summary["beams"][beam].get("recall@20"),
                RECALL_TOLERANCE,
            )
    return {"ok": not mismatches, "mismatches": mismatches}


def _compare(mismatches, field, actual, expected, tolerance):
    if expected is None or actual is None:
        return
    if abs(float(actual) - float(expected)) > tolerance:
        mismatches.append({"field": field, "per_request": float(actual), "summary": float(expected)})


def to_per_request_record(row: dict) -> dict:
    """Map a stop-gate row onto the SeetaCloud ``per_request.jsonl`` schema."""
    if "beams" in row and "delta_ndcg@20" in row:
        return row
    table = row["ndcg_by_beam"]
    beams = {}
    for beam in (SMALL_BEAM, REFERENCE_BEAM, LARGE_BEAM):
        value = float(table[beam] if beam in table else table[str(beam)])
        beams[str(beam)] = {"ndcg@20": value}
    split = row.get("split", "val")
    user_id = str(row["user_id"])
    return {
        "request_id": row.get("request_id") or f"{split}:{user_id}:0",
        "user_id": user_id,
        "split": split,
        "beams": beams,
        "delta_ndcg@20": float(beams["60"]["ndcg@20"] - beams["20"]["ndcg@20"]),
    }


def _synthetic_summary(rows) -> dict:
    stats = _population_stats(rows)
    return {
        "ranking_source": "synthetic_counterexample",
        "auditable_for_delta_ndcg": True,
        "benefit_aware_claimed": False,
        "n_requests": stats["n"],
        "beams": {
            "20": {"ndcg@20": stats["all_20_ndcg@20"]},
            "40": {"ndcg@20": stats["all_40_ndcg@20"]},
            "60": {"ndcg@20": stats["all_60_ndcg@20"]},
        },
        "delta_ndcg@20": {
            "n": stats["n"],
            "mean": stats["mean_delta_ndcg@20"],
            "min": stats["min_delta_ndcg@20"],
            "max": stats["max_delta_ndcg@20"],
        },
    }


def _summary_recall(beams, beam: str):
    value = beams[beam].get("recall@20")
    if value is None:
        return None
    return float(value)


def _read_summary(path: str) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if "beams" not in document or "delta_ndcg@20" not in document:
        raise ValueError("summary JSON needs beams and delta_ndcg@20")
    return document


def _error(command: str, message: str) -> dict:
    return {
        "command": command,
        "ok": False,
        "exit_code": 1,
        "status": "error",
        "scripted": False,
        "message": message,
        "benefit_aware_claimed": False,
        "grid_recommender_started": False,
        "p2_worth_doing": None,
    }
