"""Audit Beauty eval samples for target-event leakage and the split protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from budget_sid.constants import GRID_COMMIT
from budget_sid.eval_samples import (
    iter_leak_failures,
    read_jsonl,
    synthetic_samples,
)
from budget_sid.grid_collate import COLLATE_NOTE, describe_collate_inputs
from budget_sid.splits import describe_split_users

PROTOCOL_SUMMARY = (
    "Amazon Beauty in this repo follows GRID's P5 leave-one-out files. "
    "For each user the interactions are time-ordered. The testing sequence ends with the "
    "test event, the evaluation sequence is that sequence without the test event, and the "
    "training sequence is the evaluation sequence without the validation event. "
    "The same user id in more than one split is expected. "
    "Shared user ids are not label leakage. "
    "Label leakage is a target interaction event that is still inside history."
)


def _sequence(sample: dict) -> list:
    if sample.get("source_sequence") is not None:
        return list(sample["source_sequence"])
    return list(sample["history"]) + [sample["target"]]


def _prefix_report(samples) -> dict:
    by_user: dict[str, dict[str, list]] = {}
    for sample in samples:
        by_user.setdefault(str(sample["user_id"]), {}).setdefault(sample["split"], []).append(sample)
    n_all = 0
    n_ok = 0
    n_mismatch = 0
    mismatch_user_ids: list[str] = []
    for user_id, splits in sorted(by_user.items()):
        if not all(name in splits for name in ("train", "val", "test")):
            continue
        n_all += 1
        train = _single_row(splits["train"])
        val = _single_row(splits["val"])
        test = _single_row(splits["test"])
        ok = (
            train is not None
            and val is not None
            and test is not None
            and _sequence(val) == _sequence(test)[:-1]
            and _sequence(train) == _sequence(val)[:-1]
        )
        if ok:
            n_ok += 1
        else:
            n_mismatch += 1
            if len(mismatch_user_ids) < 8:
                mismatch_user_ids.append(user_id)
    return {
        "n_users_with_all_splits": n_all,
        "n_prefix_consistent": n_ok,
        "n_prefix_mismatch": n_mismatch,
        "mismatch_user_ids": mismatch_user_ids,
        "rule": "val sequence == test sequence without its last event, and train sequence == val sequence without its last event",
    }


def _single_row(rows: list[dict]) -> dict | None:
    sequences = {tuple(_sequence(row)) for row in rows}
    if len(sequences) != 1:
        return None
    return rows[0]


def _collate_report(samples) -> dict:
    n_inference_includes_target = 0
    for sample in samples:
        described = describe_collate_inputs(len(_sequence(sample)))
        if described["inference_input_includes_target_event"]:
            n_inference_includes_target += 1
    verified_files_end_with_target = None
    return {
        "grid_commit": GRID_COMMIT,
        "train_input_includes_target_event": False,
        "reported_at5_at10_uses_masked_target": True,
        "logged_ndcg_at5_at10_treated_as_leaked": False,
        "inference_collate": "collate_fn_inference_for_sequence",
        "n_rows": len(samples),
        "n_rows_inference_input_would_include_target": n_inference_includes_target,
        "pad_or_trim_keeps_tail": True,
        "note": COLLATE_NOTE,
        "stored_sequences_include_target_event": verified_files_end_with_target,
    }


def _spot_check_ids(samples, *, sample_size: int = 20, seed: int = 42) -> list[str]:
    ids = [str(sample["request_id"]) for sample in samples]
    if len(ids) <= sample_size:
        return ids
    rng = np.random.default_rng(seed)
    chosen = sorted(int(index) for index in rng.choice(len(ids), size=sample_size, replace=False))
    return [ids[index] for index in chosen]


def audit_samples(samples) -> dict:
    """Check every sample, then describe the Beauty split and the GRID collate."""
    if not samples:
        raise ValueError("no samples to audit")
    failures = iter_leak_failures(samples)
    prefixes = _prefix_report(samples)
    users = describe_split_users(samples)
    collate = _collate_report(samples)
    protocol_verified = prefixes["n_users_with_all_splits"] > 0 and prefixes["n_prefix_mismatch"] == 0
    if protocol_verified:
        collate["stored_sequences_include_target_event"] = True
        rankings_leak_safe = False
        rankings_reason = (
            "Cross-split prefixes match leave-one-out, so each stored sequence ends with its "
            "target event. collate_fn_inference_for_sequence keeps that event. "
            "Beam rankings from tiger_inference_flat are not leak-safe ΔNDCG inputs."
        )
    else:
        rankings_leak_safe = False
        rankings_reason = (
            "Existing tiger_inference_flat rankings cannot be certified leak-safe. "
            "The inference collate keeps the tail of the stored sequence. "
            "Load training, evaluation, and testing together to confirm those files end with the target event."
        )
    if failures or prefixes["n_prefix_mismatch"]:
        status = "fail"
        ok = False
    elif protocol_verified:
        status = "verified"
        ok = True
    else:
        status = "event_checks_passed_protocol_unverified"
        ok = True
    n_repeated = sum(1 for sample in samples if sample.get("target") in list(sample.get("history", [])))
    return {
        "protocol": "temporal_leave_one_out",
        "protocol_summary": PROTOCOL_SUMMARY,
        "protocol_verified": protocol_verified,
        "status": status,
        "ok": ok,
        "n_samples": len(samples),
        "n_checked": len(samples),
        "n_target_event_in_history": len(failures),
        "leak_failures": failures[:20],
        "n_repeated_target_item": n_repeated,
        "repeated_target_item_allowed": True,
        "users": users,
        "prefixes": prefixes,
        "collate": collate,
        "tiger_inference_flat_rankings_leak_safe": rankings_leak_safe,
        "tiger_inference_flat_rankings_reason": rankings_reason,
        "spot_check_request_ids": _spot_check_ids(samples),
        "benefit_aware_claimed": False,
        "p1_next_step": (
            "Re-run fixed beams 20/40/60 on the exported history (the target event removed), "
            "score them with score_beam_ndcg --ranking-source leak_safe_history, and use "
            "per_request.jsonl field delta_ndcg@20 as the oracle label. "
            "Do not fit the benefit model on the 2026-10-17 tiger_inference_flat rankings. "
            "Val events train the regressor and choose the threshold; test events stay out. "
            "Shared user ids are allowed under temporal leave-one-out."
        ),
    }


def write_audit_report(path, report: dict) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(destination)


def command_audit_eval_leakage(
    *,
    smoke: bool,
    samples: str | None = None,
    output: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "audit_eval_leakage"
    try:
        if smoke:
            loaded = synthetic_samples()
        else:
            if not samples:
                return _error(command, "Pass --samples, or --smoke for a synthetic dry-run.")
            loaded = read_jsonl(samples)
        report = audit_samples(loaded)
        written = None
        if output and not dry_run:
            written = write_audit_report(output, report)
        return {
            "command": command,
            "ok": bool(report["ok"]),
            "exit_code": 0 if report["ok"] else 1,
            "status": "smoke" if smoke else report["status"],
            "scripted": bool(smoke),
            "dry_run": bool(dry_run),
            "protocol_verified": report["protocol_verified"],
            "n_samples": report["n_samples"],
            "n_checked": report["n_checked"],
            "n_target_event_in_history": report["n_target_event_in_history"],
            "n_repeated_target_item": report["n_repeated_target_item"],
            "n_shared_val_test": report["users"]["n_shared_val_test"],
            "tiger_inference_flat_rankings_leak_safe": report["tiger_inference_flat_rankings_leak_safe"],
            "logged_ndcg_at5_at10_treated_as_leaked": report["collate"]["logged_ndcg_at5_at10_treated_as_leaked"],
            "output": written,
            "benefit_aware_claimed": False,
            "p1_next_step": report["p1_next_step"],
        }
    except (OSError, ValueError, RuntimeError) as exc:
        return _error(command, str(exc))


def _error(command: str, message: str) -> dict:
    return {
        "command": command,
        "ok": False,
        "exit_code": 1,
        "status": "error",
        "scripted": False,
        "message": message,
        "benefit_aware_claimed": False,
    }
