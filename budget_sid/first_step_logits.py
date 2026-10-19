"""Cached hierarchy-0 first-step logits and the beam-20/60 boundary features.

``dump_first_step_features`` keeps four scalars and drops the logit vector.
The boundary cut (mass on ranks 21–60, the gap between rank 20 and rank 21,
the top-20 cumulative probability) needs that vector. This module stores it
once, float16 when that cast keeps every finite score finite, so a later
scalar can be derived on CPU.

The cache joins the leak-safe predict index on ``predict_key`` (the GRID
``user_id``) and joins ``replay_budgets --features`` on ``request_id``.
Batch position is not a join key.

Default scope is validation. When ``predict_rows.jsonl`` is present the GPU
pass reads a val-only copy of the history-only folder and keeps the original
``predict_key`` as ``user_id``. Otherwise the full folder is scored and
non-validation rows are dropped after the forward. A missing tensor is an
error. Nothing is filled with zeros.

The target event stays out of the history and out of the routing features.
Whether the target's hierarchy-0 code sat in the top 20 is an attribution
question, not a feature.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from budget_sid.constants import GRID_COMMIT
from budget_sid.errors import MissingFirstStepLogits
from budget_sid.eval_samples import canonical_split, read_jsonl, write_jsonl
from budget_sid.features import (
    BOUNDARY_FEATURE_NAMES,
    BOUNDARY_TAIL_RANK,
    ROUTING_FEATURE_NAMES,
    extract_boundary_features,
)
from budget_sid.first_step_dump import (
    HIERARCHY,
    LOGITS_SOURCE_GRID,
    LOGITS_SOURCE_SMOKE,
    load_leak_safe_predict_dir,
    require_logit_vector,
)
from budget_sid.leak_safe_predict import (
    NUM_HIERARCHIES,
    ROWS_FILENAME,
    SEQUENCE_LENGTH,
    write_predict_dataset,
)

LOGITS_SOURCE_CACHE = "cached_hierarchy0_logits"
VAL_LOGITS_FILENAME = "first_step_logits_val.npz"
ALL_LOGITS_FILENAME = "first_step_logits.npz"
BOUNDARY_FEATURES_FILENAME = "boundary_features.jsonl"
SPLIT_VAL = "val"
SPLIT_ALL = "all"
JOIN_NOTE = (
    "Join replay_budgets --features on request_id. "
    "predict_key is the GRID user_id on the history-only TFRecord and the "
    "predict_key column of predict_index.jsonl. Batch position is not a join key. "
    "The target event is not in the history and is not a routing feature."
)
P3_GATE_NOTE = (
    "P3 stays gated until benefit-aware routing beats both random and entropy "
    "on mean nDCG@20 and delta_vs_all_beam20 under the pre-declared upgrade "
    "fractions, and that ordering holds across seeds 42, 43, and 44. "
    "This cache does not start P3 and does not claim that comparison."
)
# Offline attribution, not implemented and not a column: was the target
# item's hierarchy-0 semantic-id code among the top-20 first-step codes?
ATTRIBUTION_NOTE = (
    "An offline attribution check can ask whether the target item's hierarchy-0 "
    "code was inside the top-20 first-step codes. That bit reads the target, so "
    "it is a label. It is not written into the routing feature file."
)


def parse_dump_split(value: str | None) -> str:
    """``val`` (default) or ``all``. Train-only and test-only caches are refused."""
    if value is None or str(value).strip() == "":
        return SPLIT_VAL
    key = str(value).strip().lower()
    if key in {SPLIT_ALL, "*"}:
        return SPLIT_ALL
    if key in {SPLIT_VAL, "validation", "evaluation"}:
        return SPLIT_VAL
    raise ValueError(
        f"logits split must be 'val' or 'all', got {value!r}. "
        "Refusing to guess a different slice."
    )


def select_split_rows(index_rows, split: str) -> list[dict]:
    """Keep ``split`` rows in index order. ``predict_key`` is not renumbered.

    A row with no ``split`` is an error when the filter is validation. An
    empty selection is an error. The full leak-safe index is still the file
    whose ``predict_key`` values are ``0 .. n-1``.
    """
    chosen = parse_dump_split(split)
    if not index_rows:
        raise ValueError("predict index is empty")
    selected = []
    seen_keys: set[int] = set()
    seen_ids: set[str] = set()
    for position, row in enumerate(index_rows):
        request_id = row.get("request_id")
        if request_id in (None, ""):
            raise ValueError(f"predict index row {position} is missing request_id")
        if "predict_key" not in row:
            raise ValueError(f"{request_id} is missing predict_key")
        if row.get("n_history") is None:
            raise ValueError(f"{request_id} is missing n_history. Refusing to invent history_length.")
        key = int(row["predict_key"])
        if key in seen_keys:
            raise ValueError(f"duplicate predict_key {key}")
        if str(request_id) in seen_ids:
            raise ValueError(f"duplicate request_id {request_id}")
        seen_keys.add(key)
        seen_ids.add(str(request_id))
        raw_split = row.get("split")
        if chosen == SPLIT_ALL:
            if raw_split in (None, ""):
                raise ValueError(
                    f"{request_id} has no split. The cache stores split so a later "
                    "filter does not have to guess."
                )
            canonical = canonical_split(str(raw_split))
        else:
            if raw_split in (None, ""):
                raise ValueError(
                    f"{request_id} has no split. Refusing to guess which rows are validation."
                )
            canonical = canonical_split(str(raw_split))
            if canonical != chosen:
                continue
        copied = dict(row)
        copied["predict_key"] = key
        copied["request_id"] = str(request_id)
        copied["split"] = canonical
        copied["n_history"] = int(row["n_history"])
        selected.append(copied)
    if not selected:
        raise ValueError(
            f"predict index has no {chosen} rows. Refusing to invent an empty logits cache."
        )
    return selected


def sequences_for_rows(predict_dir, selected) -> list[dict] | None:
    """History rows for a val-only TFRecord, or ``None`` when the file is absent.

    ``None`` means the caller must score the full folder and filter afterwards.
    A present file that is missing a selected key is an error, not a fallback.
    ``predict_key`` stays the original index key so ``user_id`` matches the
    full-folder forward.
    """
    path = Path(predict_dir) / ROWS_FILENAME
    if not path.is_file():
        return None
    by_key: dict[int, dict] = {}
    for row in read_jsonl(path):
        if "predict_key" not in row:
            raise ValueError(f"{ROWS_FILENAME} row is missing predict_key")
        key = int(row["predict_key"])
        if key in by_key:
            raise ValueError(f"duplicate predict_key {key} in {ROWS_FILENAME}")
        by_key[key] = row
    filtered = []
    for index_row in selected:
        key = int(index_row["predict_key"])
        if key not in by_key:
            raise MissingFirstStepLogits(
                f"{ROWS_FILENAME} has no sequence for predict_key {key} "
                f"({index_row['request_id']}). Refusing to invent a history."
            )
        row = by_key[key]
        if str(row.get("request_id")) != str(index_row["request_id"]):
            raise ValueError(
                f"predict_key {key} request_id {row.get('request_id')!r} "
                f"does not match the predict index ({index_row['request_id']})."
            )
        sequence = row.get("sequence_data")
        if sequence is None:
            raise MissingFirstStepLogits(
                f"predict_key {key} has no sequence_data. Refusing to invent a history."
            )
        sequence = list(sequence)
        if len(sequence) != int(index_row["n_history"]):
            raise ValueError(
                f"predict_key {key} sequence length {len(sequence)} != "
                f"n_history {int(index_row['n_history'])}. Refusing to rewrite history."
            )
        user_id = index_row.get("user_id", row.get("user_id"))
        if user_id in (None, ""):
            raise ValueError(f"predict_key {key} is missing user_id")
        filtered.append(
            {
                "predict_key": key,
                "request_id": str(index_row["request_id"]),
                "user_id": str(user_id),
                "split": str(index_row["split"]),
                "sequence_data": sequence,
                "n_history": int(index_row["n_history"]),
            }
        )
    return filtered


def take_logits_for_index(index_rows, logits_by_key, *, allow_extra: bool) -> dict[int, np.ndarray]:
    """Keep one vector per index row.

    ``allow_extra`` is the full-then-filter case: keys outside the selection
    are dropped after every selected key has a vector. A selected key with no
    vector is an error in both modes. Extra keys are an error when the loader
    was pointed at a folder that should contain only the selection.
    """
    if not isinstance(logits_by_key, dict):
        raise MissingFirstStepLogits(
            "logits must be a map from predict_key to a vector. Refusing to invent zeros."
        )
    expected: dict[int, dict] = {}
    for row in index_rows:
        key = int(row["predict_key"])
        if key in expected:
            raise MissingFirstStepLogits(f"duplicate predict_key {key} in the logits selection")
        expected[key] = row
    if not expected:
        raise MissingFirstStepLogits("logits selection is empty. Refusing to invent zeros.")
    missing = [key for key in expected if key not in logits_by_key or logits_by_key[key] is None]
    if missing:
        shown = missing[:8]
        raise MissingFirstStepLogits(
            f"predict keys {shown} have no first-step logits. Refusing to invent zeros."
        )
    extra = sorted(int(key) for key in logits_by_key if int(key) not in expected)
    if extra and not allow_extra:
        shown = extra[:8]
        raise MissingFirstStepLogits(
            f"predict keys {shown} are outside the logits selection. Refusing to drop them."
        )
    kept = {}
    for key in expected:
        kept[key] = require_logit_vector(logits_by_key[key])
    return kept


def stack_logit_rows(index_rows, logits_by_key) -> np.ndarray:
    """``[n, vocab]`` float64 matrix in index order. Width must be constant."""
    rows = []
    width = None
    for row in index_rows:
        vector = require_logit_vector(logits_by_key[int(row["predict_key"])])
        if vector.size < BOUNDARY_TAIL_RANK:
            raise MissingFirstStepLogits(
                f"predict_key {int(row['predict_key'])} logit width {vector.size} "
                f"< {BOUNDARY_TAIL_RANK}. Boundary features need ranks 1..{BOUNDARY_TAIL_RANK}. "
                "Refusing to pad with zeros."
            )
        if width is None:
            width = int(vector.size)
        elif int(vector.size) != width:
            raise MissingFirstStepLogits(
                f"predict_key {int(row['predict_key'])} logit width {int(vector.size)} != {width}. "
                "Refusing to pad or truncate."
            )
        rows.append(vector)
    return np.stack(rows, axis=0)


def store_logits(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    """Float16 when every finite score stays finite. Otherwise float32.

    Underflow of a tiny finite value to zero is kept. Overflow to inf is not:
    softmax of an invented inf is not the model's score.
    """
    original = np.asarray(matrix, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] == 0 or original.shape[1] < BOUNDARY_TAIL_RANK:
        shape = tuple(int(part) for part in original.shape)
        raise MissingFirstStepLogits(
            f"logits cache must be [n, vocab>={BOUNDARY_TAIL_RANK}], got {shape}. "
            "Refusing to pad with zeros."
        )
    if np.isnan(original).any():
        raise MissingFirstStepLogits(
            "first-step logits contain NaN. Refusing to invent zeros."
        )
    if not np.isfinite(original).any(axis=1).all():
        raise MissingFirstStepLogits(
            "a logit row has no finite score. Refusing to invent zeros."
        )
    with np.errstate(over="ignore"):
        stored16 = original.astype(np.float16)
    if np.array_equal(np.isfinite(original), np.isfinite(stored16)):
        return np.ascontiguousarray(stored16), "float16"
    stored32 = original.astype(np.float32)
    if not np.array_equal(np.isfinite(original), np.isfinite(stored32)):
        raise MissingFirstStepLogits(
            "first-step logits overflow float32. Refusing to store inf as a score."
        )
    return np.ascontiguousarray(stored32), "float32"


def _as_text_array(values) -> np.ndarray:
    text = [str(value) for value in values]
    width = max((len(item) for item in text), default=1)
    return np.asarray(text, dtype=f"U{width}")


def logits_manifest(
    rows,
    *,
    logits_dtype: str,
    vocab: int,
    split: str,
    logits_pass: str,
    logits_source: str,
    extra: dict | None = None,
) -> dict:
    document = {
        "filename": VAL_LOGITS_FILENAME if split == SPLIT_VAL else ALL_LOGITS_FILENAME,
        "n": len(rows),
        "vocab": int(vocab),
        "logits_dtype": logits_dtype,
        "hierarchy": HIERARCHY,
        "split": split,
        "logits_pass": logits_pass,
        "logits_source": logits_source,
        "raw_logits_stored": True,
        "includes_target_event": False,
        "invented_logits": False,
        "grid_commit": GRID_COMMIT,
        "grid_pin_changed": False,
        "p3_started": False,
        "benefit_aware_online_claimed": False,
        "join": {
            "request_id": "replay_budgets --features and per_request.jsonl",
            "predict_key": "GRID user_id and predict_index.jsonl",
        },
        "join_note": JOIN_NOTE,
        "boundary_feature_names": list(BOUNDARY_FEATURE_NAMES),
        "routing_feature_names": list(ROUTING_FEATURE_NAMES),
        "derive_features_without_another_gpu_pass": (
            "python -m budget_sid extract_boundary_features "
            "--logits <this npz> --output boundary_features.jsonl --split "
            + split
        ),
        "p3_gate": P3_GATE_NOTE,
        "attribution_note": ATTRIBUTION_NOTE,
        "targets_are_routing_features": False,
    }
    if extra:
        document.update(extra)
    return document


def write_logits_cache(
    output,
    index_rows,
    logits_by_key,
    *,
    split: str,
    logits_pass: str,
    logits_source: str,
    extra: dict | None = None,
) -> dict[str, str]:
    """Write the npz and a JSON manifest. Validation runs before either file is replaced."""
    rows = select_split_rows(index_rows, split) if split else list(index_rows)
    # ``select_split_rows`` re-checks identity. Callers that already filtered
    # pass the filtered rows and the same split, which keeps them all.
    matrix = stack_logit_rows(rows, logits_by_key)
    stored, logits_dtype = store_logits(matrix)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logits": stored,
        "predict_key": np.asarray([int(row["predict_key"]) for row in rows], dtype=np.int64),
        "request_id": _as_text_array(row["request_id"] for row in rows),
        "history_length": np.asarray([int(row["n_history"]) for row in rows], dtype=np.int32),
        "split": _as_text_array(row["split"] for row in rows),
        "logits_dtype": np.asarray(logits_dtype),
    }
    partial = Path(str(destination) + ".partial")
    written_partial = Path(str(partial) + ".npz")
    try:
        np.savez_compressed(partial, **payload)
        if not written_partial.is_file():
            raise MissingFirstStepLogits(
                f"npz writer did not produce {written_partial}. Refusing to invent a cache."
            )
        os.replace(written_partial, destination)
    except Exception:
        written_partial.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        raise
    merged_extra = dict(extra or {})
    merged_extra["filename"] = destination.name
    manifest = logits_manifest(
        rows,
        logits_dtype=logits_dtype,
        vocab=int(stored.shape[1]),
        split=parse_dump_split(split),
        logits_pass=logits_pass,
        logits_source=logits_source,
        extra=merged_extra,
    )
    sidecar = destination.with_name(destination.stem + "_manifest.json")
    sidecar_partial = sidecar.with_suffix(sidecar.suffix + ".partial")
    sidecar_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(sidecar_partial, sidecar)
    return {"logits": str(destination), "manifest": str(sidecar)}


def _required_arrays(data) -> dict[str, np.ndarray]:
    required = ("logits", "predict_key", "request_id", "history_length", "split")
    names = set(getattr(data, "files", ()))
    missing = [name for name in required if name not in names]
    if missing:
        raise MissingFirstStepLogits(
            f"logits cache is missing {missing}. Refusing to invent zeros."
        )
    return {name: np.asarray(data[name]) for name in required}


def load_logits_cache(path) -> dict:
    """Load a cache written by ``write_logits_cache``.

    A sibling manifest is checked when it exists. NaN, a missing array, or a
    dtype disagreement is an error.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise MissingFirstStepLogits(
            f"logits cache not found: {file_path}. Refusing to invent zeros."
        )
    try:
        data = np.load(file_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise MissingFirstStepLogits(
            f"could not read logits cache {file_path}. Refusing to invent zeros. " + str(exc)
        ) from exc
    arrays = _required_arrays(data)
    logits = arrays["logits"]
    if logits.dtype.kind != "f":
        raise MissingFirstStepLogits(
            f"logits dtype {logits.dtype} is not floating. Refusing to invent scores."
        )
    if logits.ndim != 2 or logits.shape[0] == 0 or int(logits.shape[1]) < BOUNDARY_TAIL_RANK:
        shape = tuple(int(part) for part in logits.shape)
        raise MissingFirstStepLogits(
            f"logits cache must be [n, vocab>={BOUNDARY_TAIL_RANK}], got {shape}. "
            "Refusing to pad with zeros."
        )
    if np.isnan(logits.astype(np.float64)).any():
        raise MissingFirstStepLogits(
            "cached first-step logits contain NaN. Refusing to invent zeros."
        )
    finite = np.isfinite(logits.astype(np.float64))
    if not finite.any(axis=1).all():
        raise MissingFirstStepLogits(
            "a cached logit row has no finite score. Refusing to invent zeros."
        )
    n = int(logits.shape[0])
    keys = arrays["predict_key"].reshape(-1)
    request_ids = arrays["request_id"].reshape(-1)
    lengths = arrays["history_length"].reshape(-1)
    splits = arrays["split"].reshape(-1)
    if keys.shape[0] != n or request_ids.shape[0] != n or lengths.shape[0] != n or splits.shape[0] != n:
        raise MissingFirstStepLogits(
            "logits cache columns do not share one row count. Refusing to zip by position."
        )
    rows = []
    seen_keys: set[int] = set()
    seen_ids: set[str] = set()
    for position in range(n):
        request_id = str(request_ids[position])
        if request_id == "":
            raise MissingFirstStepLogits(
                f"cache row {position} is missing request_id. Refusing to align by position."
            )
        key = int(keys[position])
        if key in seen_keys:
            raise MissingFirstStepLogits(f"duplicate predict_key {key} in the logits cache")
        if request_id in seen_ids:
            raise MissingFirstStepLogits(f"duplicate request_id {request_id} in the logits cache")
        seen_keys.add(key)
        seen_ids.add(request_id)
        if splits[position] in (None, ""):
            raise MissingFirstStepLogits(
                f"{request_id} cache row has no split. Refusing to guess validation."
            )
        length = int(lengths[position])
        if length < 0:
            raise MissingFirstStepLogits(
                f"{request_id} history_length is negative. Refusing to invent it."
            )
        rows.append(
            {
                "predict_key": key,
                "request_id": request_id,
                "n_history": length,
                "split": canonical_split(str(splits[position])),
            }
        )
    dtype_name = str(logits.dtype)
    if "logits_dtype" in set(data.files):
        recorded = str(np.asarray(data["logits_dtype"]).reshape(-1)[0])
        if recorded != dtype_name:
            raise MissingFirstStepLogits(
                f"logits_dtype label {recorded} does not match array dtype {dtype_name}. "
                "Refusing to reinterpret the cache."
            )
        dtype_name = recorded
    manifest_path = file_path.with_name(file_path.stem + "_manifest.json")
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("invented_logits") is True:
            raise MissingFirstStepLogits(
                "logits manifest says invented_logits. Refusing to extract features from it."
            )
        if manifest.get("includes_target_event") is True:
            raise MissingFirstStepLogits(
                "logits manifest includes the target event. Refusing to use those scores as routing features."
            )
        recorded = manifest.get("logits_dtype")
        if recorded not in (None, dtype_name):
            raise MissingFirstStepLogits(
                f"manifest logits_dtype {recorded} does not match the array ({dtype_name})."
            )
        if manifest.get("n") not in (None, n):
            raise MissingFirstStepLogits(
                f"manifest n={manifest.get('n')} does not match the cache n={n}."
            )
    return {
        "rows": rows,
        "logits": np.asarray(logits, dtype=np.float64),
        "logits_dtype": dtype_name,
        "manifest": manifest,
        "path": str(file_path),
    }


def boundary_rows_from_cache(cache, *, split: str | None = None) -> list[dict]:
    """One routing-feature row per cached request, optionally filtered by split."""
    rows = list(cache["rows"])
    logits = cache["logits"]
    if split not in (None, ""):
        chosen = parse_dump_split(split)
        if chosen != SPLIT_ALL:
            rows = [row for row in rows if row["split"] == chosen]
            if not rows:
                raise ValueError(
                    f"logits cache has no {chosen} rows. Refusing to invent boundary features."
                )
    by_key = {int(row["predict_key"]): position for position, row in enumerate(cache["rows"])}
    extracted = []
    for row in rows:
        position = by_key[int(row["predict_key"])]
        vector = logits[position]
        features = extract_boundary_features(int(row["n_history"]), vector)
        record = {
            "request_id": str(row["request_id"]),
            "predict_key": int(row["predict_key"]),
            "split": str(row["split"]),
            "n_logits": int(vector.shape[0]),
            "hierarchy": HIERARCHY,
            "includes_target_event": False,
            "logits_source": LOGITS_SOURCE_CACHE,
            "raw_logits_stored": False,
        }
        for name in ROUTING_FEATURE_NAMES:
            record[name] = float(features[name])
        extracted.append(record)
    return extracted


def boundary_manifest(rows, *, split: str | None, logits_path: str | None) -> dict:
    return {
        "filename": BOUNDARY_FEATURES_FILENAME,
        "n": len(rows),
        "feature_names": list(ROUTING_FEATURE_NAMES),
        "boundary_feature_names": list(BOUNDARY_FEATURE_NAMES),
        "hierarchy": HIERARCHY,
        "split": None if split in (None, "") else parse_dump_split(split),
        "logits_cache": logits_path,
        "logits_source": LOGITS_SOURCE_CACHE,
        "raw_logits_stored": False,
        "includes_target_event": False,
        "invented_logits": False,
        "targets_are_routing_features": False,
        "join_note": JOIN_NOTE,
        "p3_gate": P3_GATE_NOTE,
        "attribution_note": ATTRIBUTION_NOTE,
        "grid_commit": GRID_COMMIT,
        "grid_pin_changed": False,
        "p3_started": False,
        "benefit_aware_online_claimed": False,
    }


def write_boundary_features(output, rows, manifest: dict) -> dict[str, str]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    write_jsonl(partial, rows)
    os.replace(partial, destination)
    sidecar = destination.with_name(destination.stem + "_manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"features": str(destination), "manifest": str(sidecar)}


def assert_cached_request_coverage(cache_rows, other_rows, *, split: str, other_name: str) -> None:
    """Every cached id is in ``other_rows``, and every other row of ``split`` is cached."""
    other_by_id = {}
    for row in other_rows:
        request_id = row.get("request_id")
        if request_id in (None, ""):
            raise ValueError(f"{other_name} row is missing request_id")
        other_by_id[str(request_id)] = row
    cache_ids = []
    for row in cache_rows:
        request_id = str(row["request_id"])
        if request_id not in other_by_id:
            raise ValueError(
                f"{request_id} is in the logits cache but not in {other_name}. "
                "Refusing to drop a cached request."
            )
        cache_ids.append(request_id)
    cache_set = set(cache_ids)
    chosen = parse_dump_split(split)
    if chosen == SPLIT_ALL:
        missing = [request_id for request_id in other_by_id if request_id not in cache_set]
    else:
        missing = []
        saw_split = False
        for request_id, row in other_by_id.items():
            raw_split = row.get("split")
            if raw_split in (None, ""):
                continue
            saw_split = True
            if canonical_split(str(raw_split)) == chosen and request_id not in cache_set:
                missing.append(request_id)
        if not saw_split:
            return
    if missing:
        shown = missing[:8]
        raise ValueError(
            f"{other_name} is missing {len(missing)} cached-split request ids, including {shown}. "
            "The logits cache does not cover that split."
        )


def _scripted_wide_logits(request_id: str, width: int = 64) -> np.ndarray:
    """Deterministic CPU stand-in. Not a Beauty GRID vector and not a zero fill."""
    import hashlib

    if width < BOUNDARY_TAIL_RANK:
        raise ValueError(f"scripted width {width} cannot name ranks 1..{BOUNDARY_TAIL_RANK}")
    digest = hashlib.sha256(str(request_id).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=width).astype(np.float64)
    logits[seed % width] += 4.0
    return logits


def smoke_logits_selection(split: str = SPLIT_VAL) -> tuple[list[dict], dict[int, np.ndarray]]:
    from budget_sid.eval_samples import synthetic_samples
    from budget_sid.leak_safe_predict import history_only_rows

    index_rows = history_only_rows(synthetic_samples())
    selected = select_split_rows(index_rows, split)
    logits = {int(row["predict_key"]): _scripted_wide_logits(row["request_id"]) for row in selected}
    return selected, logits


def _flags(**extra) -> dict:
    payload = {
        "invented_logits": False,
        "includes_target_event": False,
        "p3_started": False,
        "benefit_aware_online_claimed": False,
        "latency_claimed": False,
        "grid_pin_changed": False,
        "grid_commit": GRID_COMMIT,
        "targets_are_routing_features": False,
    }
    payload.update(extra)
    return payload


def _ok(command: str, **extra) -> dict:
    payload = {
        "command": command,
        "ok": True,
        "exit_code": 0,
        "status": extra.pop("status"),
        "scripted": bool(extra.pop("scripted")),
        "dry_run": bool(extra.pop("dry_run")),
    }
    payload.update(_flags())
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
        **_flags(),
    }


def _plan_logits_pass(predict_dir: str, selected, split: str) -> str:
    if split == SPLIT_ALL:
        return "all"
    if (Path(predict_dir) / ROWS_FILENAME).is_file():
        # Touch the sequences so a mismatch fails before anyone calls this a dry success
        # only when we are about to claim val_only. Dry-run still validates the file.
        sequences_for_rows(predict_dir, selected)
        return "val_only"
    return "full_then_filter"


def command_dump_first_step_logits(
    *,
    smoke: bool = False,
    predict_dir: str | None = None,
    output: str | None = None,
    ckpt: str | None = None,
    semantic_id_path: str | None = None,
    data_dir: str | None = None,
    grid_dir: str | None = None,
    per_request: str | None = None,
    split: str | None = None,
    filtered_predict_dir: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "dump_first_step_logits"
    try:
        chosen = parse_dump_split(split)
        if smoke:
            selected, logits_by_key = smoke_logits_selection(chosen)
            written = {}
            if output and not dry_run:
                written = write_logits_cache(
                    output,
                    selected,
                    logits_by_key,
                    split=chosen,
                    logits_pass="scripted",
                    logits_source=LOGITS_SOURCE_SMOKE,
                )
            return _ok(
                command,
                status="smoke",
                scripted=True,
                dry_run=bool(dry_run),
                n=len(selected),
                split=chosen,
                logits_pass="scripted",
                logits_source=LOGITS_SOURCE_SMOKE,
                grid_first_step=False,
                raw_logits_stored=bool(written),
                vocab=int(next(iter(logits_by_key.values())).shape[0]),
                boundary_feature_names=list(BOUNDARY_FEATURE_NAMES),
                routing_feature_names=list(ROUTING_FEATURE_NAMES),
                join_note=JOIN_NOTE,
                output=written,
            )
        if not predict_dir:
            return _error(
                command,
                "Pass --predict-dir from build_leak_safe_predict and --ckpt. "
                "Use --smoke for the scripted CPU cache. This process does not invent logits.",
            )
        predict_dir = str(Path(predict_dir).resolve())
        if output:
            output = str(Path(output).resolve())
        index_rows, manifest = load_leak_safe_predict_dir(predict_dir)
        selected = select_split_rows(index_rows, chosen)
        logits_pass = _plan_logits_pass(predict_dir, selected, chosen)
        if per_request:
            assert_cached_request_coverage(
                selected,
                read_jsonl(per_request),
                split=chosen,
                other_name="per_request.jsonl",
            )
        if dry_run:
            if not ckpt:
                return _error(command, "Pass --ckpt. Dry-run does not invent logits.")
            if not Path(ckpt).is_file():
                return _error(
                    command,
                    f"Checkpoint not found: {ckpt}. Refusing to invent first-step logits.",
                )
            from budget_sid.grid_first_step import hydra_overrides

            run_dir = predict_dir
            if logits_pass == "val_only":
                run_dir = str(
                    Path(filtered_predict_dir).resolve()
                    if filtered_predict_dir
                    else Path(predict_dir).with_name(Path(predict_dir).name + "_val")
                )
            overrides = []
            if grid_dir and semantic_id_path and data_dir:
                overrides = hydra_overrides(
                    data_dir=str(Path(data_dir).resolve()),
                    sid_path=str(Path(semantic_id_path).resolve()),
                    ckpt=str(Path(ckpt).resolve()),
                    predict_dir=run_dir,
                )
            return _ok(
                command,
                status="dry_run",
                scripted=False,
                dry_run=True,
                n=len(selected),
                n_index=len(index_rows),
                split=chosen,
                logits_pass=logits_pass,
                logits_source=LOGITS_SOURCE_GRID,
                grid_first_step=False,
                raw_logits_stored=False,
                ckpt_loaded=False,
                grid_pin_checked=False,
                ranking_source=manifest.get("ranking_source"),
                predict_dir=predict_dir,
                filtered_predict_dir=run_dir if logits_pass == "val_only" else None,
                hydra_overrides=overrides,
                join_note=JOIN_NOTE,
                boundary_feature_names=list(BOUNDARY_FEATURE_NAMES),
                output={},
                message=(
                    "Dry-run checked the leak-safe predict folder and the checkpoint file. "
                    "The model was not loaded and no logits were written."
                ),
            )
        if not ckpt or not semantic_id_path or not data_dir or not grid_dir:
            return _error(
                command,
                "Pass --predict-dir, --ckpt, --semantic-id-path, --data-dir, and --grid-dir. "
                "Refusing to invent first-step logits.",
            )
        if not output:
            return _error(command, "Pass --output for the logits npz. Refusing to drop the vectors.")
        from budget_sid.grid_first_step import run_grid_first_step_logits_dump

        run_dir = predict_dir
        run_expected = index_rows
        allow_extra = False
        if logits_pass == "val_only":
            sequences = sequences_for_rows(predict_dir, selected)
            if sequences is None:
                raise MissingFirstStepLogits(
                    f"{ROWS_FILENAME} disappeared after the dry plan. Refusing to invent histories."
                )
            run_dir = str(
                Path(filtered_predict_dir).resolve()
                if filtered_predict_dir
                else Path(predict_dir).with_name(Path(predict_dir).name + "_val")
            )
            write_predict_dataset(
                run_dir,
                sequences,
                sequence_length=SEQUENCE_LENGTH,
                num_hierarchies=NUM_HIERARCHIES,
                sid_checked=False,
            )
            run_expected = selected
            allow_extra = False
        elif logits_pass == "full_then_filter":
            run_expected = index_rows
            allow_extra = False
        collected = run_grid_first_step_logits_dump(
            grid_dir=str(Path(grid_dir).resolve()),
            data_dir=str(Path(data_dir).resolve()),
            sid_path=str(Path(semantic_id_path).resolve()),
            ckpt=str(Path(ckpt).resolve()),
            predict_dir=run_dir,
        )
        if logits_pass == "full_then_filter":
            take_logits_for_index(index_rows, collected, allow_extra=False)
            kept = take_logits_for_index(selected, collected, allow_extra=True)
        else:
            kept = take_logits_for_index(run_expected, collected, allow_extra=allow_extra)
        written = write_logits_cache(
            output,
            selected,
            kept,
            split=chosen,
            logits_pass=logits_pass,
            logits_source=LOGITS_SOURCE_GRID,
            extra={
                "predict_dir": predict_dir,
                "filtered_predict_dir": run_dir if logits_pass == "val_only" else None,
                "ckpt": str(Path(ckpt).resolve()),
                "per_request_coverage_checked": per_request is not None,
                "grid_pin_checked": True,
            },
        )
        return _ok(
            command,
            status="dumped",
            scripted=False,
            dry_run=False,
            n=len(selected),
            split=chosen,
            logits_pass=logits_pass,
            logits_source=LOGITS_SOURCE_GRID,
            grid_first_step=True,
            raw_logits_stored=True,
            grid_pin_checked=True,
            join_note=JOIN_NOTE,
            boundary_feature_names=list(BOUNDARY_FEATURE_NAMES),
            routing_feature_names=list(ROUTING_FEATURE_NAMES),
            output=written,
        )
    except (OSError, ValueError, RuntimeError, MissingFirstStepLogits, KeyError, TypeError) as exc:
        return _error(command, str(exc))


def command_extract_boundary_features(
    *,
    smoke: bool = False,
    logits: str | None = None,
    output: str | None = None,
    split: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "extract_boundary_features"
    try:
        if smoke:
            chosen_smoke = parse_dump_split(split or SPLIT_VAL)
            selected, logits_by_key = smoke_logits_selection(chosen_smoke)
            matrix = stack_logit_rows(selected, logits_by_key)
            cache = {
                "rows": [
                    {
                        "predict_key": int(row["predict_key"]),
                        "request_id": str(row["request_id"]),
                        "n_history": int(row["n_history"]),
                        "split": str(row["split"]),
                    }
                    for row in selected
                ],
                "logits": matrix,
                "logits_dtype": "float64",
                "manifest": None,
                "path": None,
            }
            rows = boundary_rows_from_cache(cache, split=chosen_smoke)
            written = {}
            if output and not dry_run:
                written = write_boundary_features(
                    output,
                    rows,
                    boundary_manifest(rows, split=chosen_smoke, logits_path=None),
                )
            return _ok(
                command,
                status="smoke",
                scripted=True,
                dry_run=bool(dry_run),
                n=len(rows),
                split=chosen_smoke,
                feature_names=list(ROUTING_FEATURE_NAMES),
                boundary_feature_names=list(BOUNDARY_FEATURE_NAMES),
                logits_source=LOGITS_SOURCE_SMOKE,
                raw_logits_stored=False,
                grid_first_step=False,
                join_note=JOIN_NOTE,
                output=written,
            )
        if not logits:
            return _error(
                command,
                "Pass --logits (the hierarchy-0 npz cache). "
                "This process does not invent logits and does not run GRID.",
            )
        cache = load_logits_cache(logits)
        chosen = None if split in (None, "") else parse_dump_split(split)
        rows = boundary_rows_from_cache(cache, split=chosen)
        if dry_run:
            return _ok(
                command,
                status="dry_run",
                scripted=False,
                dry_run=True,
                n=len(rows),
                split=chosen if chosen is not None else "cache",
                feature_names=list(ROUTING_FEATURE_NAMES),
                boundary_feature_names=list(BOUNDARY_FEATURE_NAMES),
                logits_source=LOGITS_SOURCE_CACHE,
                raw_logits_stored=False,
                grid_first_step=False,
                logits_dtype=cache["logits_dtype"],
                join_note=JOIN_NOTE,
                output={},
                message="Dry-run read the logits cache and did not write a feature file.",
            )
        if not output:
            return _error(command, "Pass --output for the boundary feature jsonl.")
        written = write_boundary_features(
            output,
            rows,
            boundary_manifest(rows, split=chosen, logits_path=str(Path(logits).resolve())),
        )
        return _ok(
            command,
            status="extracted",
            scripted=False,
            dry_run=False,
            n=len(rows),
            split=chosen if chosen is not None else "cache",
            feature_names=list(ROUTING_FEATURE_NAMES),
            boundary_feature_names=list(BOUNDARY_FEATURE_NAMES),
            logits_source=LOGITS_SOURCE_CACHE,
            raw_logits_stored=False,
            grid_first_step=False,
            logits_dtype=cache["logits_dtype"],
            join_note=JOIN_NOTE,
            output=written,
        )
    except (OSError, ValueError, RuntimeError, MissingFirstStepLogits, KeyError, TypeError) as exc:
        return _error(command, str(exc))
