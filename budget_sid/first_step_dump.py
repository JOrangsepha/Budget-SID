"""Leak-safe first-step features joined on ``request_id``.

The file this module writes is the ``--features`` input for
``replay_budgets``. Rows follow ``predict_index.jsonl`` from
``build_leak_safe_predict``, which is the history-only order used by
``scripts/run_leak_safe_beams.sh`` and by ``outputs/leak_safe_beams/per_request.jsonl``
when both were built from the same ``eval_samples.jsonl``.

``history_length`` is ``n_history`` on that index: the exported history, with
the target event left out. The encoder still sees only the collate tail. The
length feature is the full leak-safe history, the same value replay reads from
``--predict-index``.

Pinned GRID ``SemanticIDEncoderDecoder.generate`` (commit ``2fe3475``) computes
hierarchy-0 ``candidate_logits`` and then discards them.
``predict_step`` returns semantic ids only. The GPU path calls those same
public methods and keeps the logit matrix. It does not run the rest of the
beam search, and it does not read the default ``testing`` folder. A missing
tensor is an error. This module does not fill one in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from budget_sid.errors import MissingFirstStepLogits
from budget_sid.eval_samples import read_jsonl, write_jsonl
from budget_sid.features import FEATURE_NAMES, extract_features
from budget_sid.leak_safe_predict import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    PREDICT_FILENAME,
    RANKING_SOURCE,
)

FEATURES_FILENAME = "first_step_features.jsonl"
LOGITS_SOURCE_GRID = "grid_hierarchy0_candidate_logits"
LOGITS_SOURCE_SMOKE = "scripted_cpu"
HIERARCHY = 0


def _to_numpy(logits) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach()
    if hasattr(logits, "cpu"):
        logits = logits.cpu()
    if hasattr(logits, "numpy") and not isinstance(logits, np.ndarray):
        logits = logits.numpy()
    try:
        return np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MissingFirstStepLogits(
            "first-step logits are not a numeric tensor. Refusing to invent zeros. " + str(exc)
        ) from exc


def require_logit_vector(logits) -> np.ndarray:
    """One finite-or--inf score vector. ``None`` and NaN are errors, not zeros."""
    if logits is None:
        raise MissingFirstStepLogits(
            "first-step logits are missing. Refusing to invent zeros."
        )
    array = _to_numpy(logits)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise MissingFirstStepLogits(
                f"expected one logit vector, got shape {tuple(int(part) for part in array.shape)}. "
                "Refusing to invent zeros."
            )
        array = array.reshape(-1)
    if array.ndim != 1 or array.size < 2:
        raise MissingFirstStepLogits(
            "first-step distribution needs a vector of at least two candidates. "
            "Refusing to invent zeros."
        )
    if np.isnan(array).any():
        raise MissingFirstStepLogits(
            "first-step logits contain NaN. Refusing to invent zeros."
        )
    if not np.isfinite(array).any():
        raise MissingFirstStepLogits(
            "first-step logits have no finite score. Refusing to invent zeros."
        )
    return array


def require_logit_matrix(logits) -> np.ndarray:
    """A ``[batch, vocab]`` matrix. Empty batches and NaN rows are errors."""
    if logits is None:
        raise MissingFirstStepLogits(
            "first-step logits are missing. Refusing to invent zeros."
        )
    array = _to_numpy(logits)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        shape = tuple(int(part) for part in array.shape)
        raise MissingFirstStepLogits(
            f"first-step logits must be a [batch, vocab] matrix with at least two candidates, got {shape}. "
            "Refusing to invent zeros."
        )
    if np.isnan(array).any():
        raise MissingFirstStepLogits(
            "first-step logits contain NaN. Refusing to invent zeros."
        )
    bad = [index for index, row in enumerate(array) if not np.isfinite(row).any()]
    if bad:
        shown = bad[:8]
        raise MissingFirstStepLogits(
            f"logit rows {shown} have no finite score. Refusing to invent zeros."
        )
    return array


def ordered_index(index_rows) -> list[dict]:
    """``predict_key`` must be the row offset, ``0 .. n-1``, as ``history_only_rows`` writes it."""
    if not index_rows:
        raise ValueError("predict index is empty")
    keys: set[int] = set()
    request_ids: set[str] = set()
    ordered = []
    for position, row in enumerate(index_rows):
        request_id = row.get("request_id")
        if request_id in (None, ""):
            raise ValueError(f"predict index row {position} is missing request_id")
        if "predict_key" not in row:
            raise ValueError(f"predict index row {request_id} is missing predict_key")
        key = int(row["predict_key"])
        if key != position:
            raise ValueError(
                f"predict_key {key} is not row offset {position}. "
                "leak-safe predict_index.jsonl is 0..n-1 in eval-sample order, "
                "the same order as per_request.jsonl."
            )
        if key in keys:
            raise ValueError(f"duplicate predict_key {key}")
        if str(request_id) in request_ids:
            raise ValueError(f"duplicate request_id {request_id}")
        if row.get("n_history") is None:
            raise ValueError(
                f"{request_id} predict index row is missing n_history. "
                "Refusing to invent history_length."
            )
        keys.add(key)
        request_ids.add(str(request_id))
        ordered.append(row)
    return ordered


def assert_leak_safe_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("predict manifest must be an object")
    if manifest.get("includes_target_event") is not False:
        raise ValueError(
            "predict manifest includes_target_event is not false. "
            "Refusing to dump first-step features from a sequence that still contains the target event. "
            "Build the folder with build_leak_safe_predict."
        )
    if manifest.get("ranking_source") != RANKING_SOURCE:
        raise ValueError(
            "predict manifest ranking_source must be leak_safe_history. "
            "Refusing to dump features for a different predict folder."
        )


def load_leak_safe_predict_dir(predict_dir) -> tuple[list[dict], dict]:
    root = Path(predict_dir)
    index_path = root / INDEX_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    tfrecord_path = root / PREDICT_FILENAME
    missing = [
        name
        for name, path in (
            (PREDICT_FILENAME, tfrecord_path),
            (INDEX_FILENAME, index_path),
            (MANIFEST_FILENAME, manifest_path),
        )
        if not path.is_file()
    ]
    if missing:
        raise ValueError(
            f"leak-safe predict set under {root} is missing {missing}. "
            "Expected the folder from build_leak_safe_predict "
            "(history_only.tfrecord.gz, predict_index.jsonl, manifest.json). "
            "Refusing to read GRID's default testing folder."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_leak_safe_manifest(manifest)
    return ordered_index(read_jsonl(index_path)), manifest


def feature_row_from_logits(
    *,
    request_id,
    predict_key: int,
    history_length: int,
    logits,
    logits_source: str,
) -> dict:
    vector = require_logit_vector(logits)
    if history_length is None or int(history_length) < 0:
        raise MissingFirstStepLogits(
            f"{request_id} is missing a non-negative history_length. Refusing to invent it."
        )
    features = extract_features(int(history_length), vector)
    return {
        "request_id": str(request_id),
        "predict_key": int(predict_key),
        "history_length": float(features["history_length"]),
        "first_step_entropy": float(features["first_step_entropy"]),
        "candidate_prob_gap": float(features["candidate_prob_gap"]),
        "top1_prob": float(features["top1_prob"]),
        "n_logits": int(vector.size),
        "hierarchy": HIERARCHY,
        "includes_target_event": False,
        "logits_source": str(logits_source),
        "raw_logits_stored": False,
    }


def feature_rows_from_keyed_logits(index_rows, logits_by_key, *, logits_source: str) -> list[dict]:
    """One feature row per index row, in ``predict_key`` order.

    ``logits_by_key`` is keyed by the GRID ``user_id`` field, which the
    leak-safe TFRecord sets to ``predict_key``. Dataloader order is not the
    join key. A key with no logits is an error, not a zero vector.
    """
    ordered = ordered_index(index_rows)
    if not isinstance(logits_by_key, dict):
        raise MissingFirstStepLogits(
            "logits must be a map from predict_key to a vector. Refusing to invent zeros."
        )
    expected = {int(row["predict_key"]) for row in ordered}
    extra = [int(key) for key in logits_by_key if int(key) not in expected]
    if extra:
        shown = sorted(extra)[:8]
        raise MissingFirstStepLogits(
            f"predict keys {shown} are not in the leak-safe index. Refusing to drop them."
        )
    rows = []
    for row in ordered:
        key = int(row["predict_key"])
        if key not in logits_by_key or logits_by_key[key] is None:
            raise MissingFirstStepLogits(
                f"predict_key {key} ({row['request_id']}) has no first-step logits. "
                "Refusing to invent zeros."
            )
        rows.append(
            feature_row_from_logits(
                request_id=row["request_id"],
                predict_key=key,
                history_length=int(row["n_history"]),
                logits=logits_by_key[key],
                logits_source=logits_source,
            )
        )
    return rows


def assert_same_request_ids(feature_rows, other_rows, *, other_name: str) -> None:
    """The dump and ``per_request.jsonl`` share eval-sample order."""
    left = [str(row["request_id"]) for row in feature_rows]
    right = []
    for row in other_rows:
        request_id = row.get("request_id")
        if request_id in (None, ""):
            raise ValueError(f"{other_name} row is missing request_id")
        right.append(str(request_id))
    if left != right:
        raise ValueError(
            f"request_id order does not match {other_name} "
            f"(features n={len(left)}, {other_name} n={len(right)}). "
            "first_step_features.jsonl follows predict_index.jsonl. "
            "per_request.jsonl follows the same eval_samples.jsonl order when "
            "score_beam_ndcg and build_leak_safe_predict read that file. "
            "Refusing to join on a different order."
        )


def dump_manifest(rows, *, logits_source: str, extra: dict | None = None) -> dict:
    document = {
        "filename": FEATURES_FILENAME,
        "n": len(rows),
        "feature_names": list(FEATURE_NAMES),
        "hierarchy": HIERARCHY,
        "logits_source": logits_source,
        "raw_logits_stored": False,
        "includes_target_event": False,
        "invented_logits": False,
        "alignment": (
            "Rows are predict_index.jsonl order (predict_key 0..n-1). "
            "GRID user_id on the history-only TFRecord is that predict_key. "
            "request_id is copied from the index, not from batch position. "
            "outputs/leak_safe_beams/per_request.jsonl uses the same order when "
            "it was scored from the same eval_samples.jsonl."
        ),
        "history_length": "n_history from predict_index.jsonl; the target event is not included",
        "features_from": "budget_sid.features.extract_features on hierarchy-0 candidate logits",
        "p3_started": False,
        "benefit_aware_online_claimed": False,
        "grid_pin_changed": False,
    }
    if extra:
        document.update(extra)
    return document


def write_feature_dump(output, rows, manifest: dict) -> dict[str, str]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    write_jsonl(partial, rows)
    os.replace(partial, destination)
    sidecar = destination.with_name(destination.stem + "_manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"features": str(destination), "manifest": str(sidecar)}


def smoke_feature_rows() -> list[dict]:
    """Scripted CPU stand-in. These logits are not a Beauty GRID dump."""
    from budget_sid.eval_samples import synthetic_samples
    from budget_sid.leak_safe_predict import history_only_rows
    from budget_sid.recommender import ScriptedRecommender

    samples = synthetic_samples()
    index_rows = history_only_rows(samples)
    model = ScriptedRecommender(catalog=(1, 2, 3, 4), seed=0)
    logits_by_key = {}
    for row in index_rows:
        state = model.first_step(row["sequence_data"])
        if state.logits is None:
            raise MissingFirstStepLogits(
                "scripted first step returned no logits. Refusing to invent zeros."
            )
        logits_by_key[int(row["predict_key"])] = state.logits
    return feature_rows_from_keyed_logits(
        index_rows,
        logits_by_key,
        logits_source=LOGITS_SOURCE_SMOKE,
    )


def command_dump_first_step_features(
    *,
    smoke: bool = False,
    predict_dir: str | None = None,
    output: str | None = None,
    ckpt: str | None = None,
    semantic_id_path: str | None = None,
    data_dir: str | None = None,
    grid_dir: str | None = None,
    per_request: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "dump_first_step_features"
    try:
        if smoke:
            rows = smoke_feature_rows()
            written = {}
            if output and not dry_run:
                written = write_feature_dump(
                    output,
                    rows,
                    dump_manifest(rows, logits_source=LOGITS_SOURCE_SMOKE),
                )
            return _ok(
                command,
                status="smoke",
                scripted=True,
                dry_run=bool(dry_run),
                rows=rows,
                logits_source=LOGITS_SOURCE_SMOKE,
                grid_first_step=False,
                output=written,
            )
        if not predict_dir:
            return _error(
                command,
                "Pass --predict-dir from build_leak_safe_predict and --ckpt. "
                "Use --smoke for the scripted CPU join. This process does not invent logits.",
            )
        # The GRID runner chdirs into the checkout. Relative outputs would land there.
        if output:
            output = str(Path(output).resolve())
        predict_dir = str(Path(predict_dir).resolve())
        index_rows, manifest = load_leak_safe_predict_dir(predict_dir)
        other_rows = read_jsonl(per_request) if per_request else None
        if dry_run:
            if not ckpt:
                return _error(command, "Pass --ckpt. Dry-run does not invent logits.")
            if not Path(ckpt).is_file():
                return _error(
                    command,
                    f"Checkpoint not found: {ckpt}. Refusing to invent first-step logits.",
                )
            from budget_sid.grid_first_step import hydra_overrides

            overrides = []
            if grid_dir and semantic_id_path and data_dir:
                overrides = hydra_overrides(
                    data_dir=str(Path(data_dir).resolve()),
                    sid_path=str(Path(semantic_id_path).resolve()),
                    ckpt=str(Path(ckpt).resolve()),
                    predict_dir=str(Path(predict_dir).resolve()),
                )
            return _ok(
                command,
                status="dry_run",
                scripted=False,
                dry_run=True,
                rows=None,
                n=len(index_rows),
                logits_source=LOGITS_SOURCE_GRID,
                grid_first_step=False,
                output={},
                extra={
                    "predict_dir": str(predict_dir),
                    "ckpt": str(ckpt),
                    "ckpt_loaded": False,
                    "includes_target_event": False,
                    "ranking_source": manifest.get("ranking_source"),
                    "hydra_overrides": overrides,
                    "per_request_checked": False,
                    "message": (
                        "Dry-run checked the leak-safe predict folder and the checkpoint file. "
                        "The model was not loaded and no logits were written."
                    ),
                },
            )
        if not ckpt or not semantic_id_path or not data_dir or not grid_dir:
            return _error(
                command,
                "Pass --predict-dir, --ckpt, --semantic-id-path, --data-dir, and --grid-dir. "
                "Refusing to invent first-step logits.",
            )
        from budget_sid.grid_first_step import run_grid_first_step_dump

        rows = run_grid_first_step_dump(
            grid_dir=str(Path(grid_dir).resolve()),
            data_dir=str(Path(data_dir).resolve()),
            sid_path=str(Path(semantic_id_path).resolve()),
            ckpt=str(Path(ckpt).resolve()),
            predict_dir=str(Path(predict_dir).resolve()),
            index_rows=index_rows,
        )
        if other_rows is not None:
            assert_same_request_ids(rows, other_rows, other_name="per_request.jsonl")
        written = {}
        if output:
            written = write_feature_dump(
                output,
                rows,
                dump_manifest(
                    rows,
                    logits_source=LOGITS_SOURCE_GRID,
                    extra={
                        "grid_commit": manifest.get("grid_commit"),
                        "predict_dir": str(Path(predict_dir).resolve()),
                        "ckpt": str(Path(ckpt).resolve()),
                        "per_request_order_checked": other_rows is not None,
                    },
                ),
            )
        return _ok(
            command,
            status="dumped",
            scripted=False,
            dry_run=False,
            rows=rows,
            logits_source=LOGITS_SOURCE_GRID,
            grid_first_step=True,
            output=written,
            extra={"per_request_order_checked": other_rows is not None},
        )
    except (OSError, ValueError, RuntimeError, MissingFirstStepLogits, KeyError, TypeError) as exc:
        return _error(command, str(exc))


def _ok(
    command,
    *,
    status,
    scripted,
    dry_run,
    rows,
    logits_source,
    grid_first_step,
    output,
    n=None,
    extra=None,
) -> dict:
    payload = {
        "command": command,
        "ok": True,
        "exit_code": 0,
        "status": status,
        "scripted": bool(scripted),
        "dry_run": bool(dry_run),
        "n": len(rows) if rows is not None else int(n or 0),
        "feature_names": list(FEATURE_NAMES),
        "logits_source": logits_source,
        "grid_first_step": bool(grid_first_step),
        "includes_target_event": False,
        "invented_logits": False,
        "raw_logits_stored": False,
        "alignment": "predict_index.jsonl order, joined on predict_key",
        "output": output,
        "p3_started": False,
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_pin_changed": False,
    }
    if extra:
        payload.update(extra)
    return payload


def _error(command: str, message: str) -> dict:
    return {
        "command": command,
        "ok": False,
        "exit_code": 1,
        "status": "error",
        "scripted": False,
        "message": message,
        "invented_logits": False,
        "p3_started": False,
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_pin_changed": False,
    }
