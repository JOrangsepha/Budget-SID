"""History-only predict inputs and ranking export for leak-safe fixed beams.

``tiger_inference_flat`` (GRID ``2fe3475``) reads gzip TFRecords from
``data_loading.predict_dataloader_config.dataloader.data_folder``. The default
folder is ``${data_dir}/testing``, whose stored sequence ends with the target
event. ``collate_fn_inference_for_sequence`` pads or trims and keeps that
tail, so those beams are not leak-safe ΔNDCG inputs.

This module writes a folder the same loader can read:

- ``sequence_data`` is the exported ``history`` and nothing else. Item ids are
  looked up in the semantic-id tensor (``[num_hierarchies, n_items]``) and
  flattened to ``num_hierarchies`` tokens per item.
- ``user_id`` is a dense ``predict_key`` in ``0 .. n-1``, not the raw user id.
  Raw ids collide across the temporal leave-one-out splits, and GRID's merged
  prediction tensor indexes by this field. ``predict_index.jsonl`` maps the
  key back to ``request_id``.

The collate still trims the token sequence to ``sequence_length`` (120) from
the right. With 4 hierarchies that window is the last 30 history items. The
target event is not in the window because it was never written.
"""

from __future__ import annotations

import json
from pathlib import Path

from budget_sid.beam_score import dedupe_keep_order, load_sid_map
from budget_sid.constants import BEAM_WIDTHS
from budget_sid.eval_samples import (
    normalize_item,
    read_jsonl,
    synthetic_samples,
    target_event_in_history,
    write_jsonl,
)
from budget_sid.tfrecord_io import read_tfrecord_gz, write_tfrecord_gz

SEQUENCE_LENGTH = 120
NUM_HIERARCHIES = 4
PREDICT_FILENAME = "history_only.tfrecord.gz"
INDEX_FILENAME = "predict_index.jsonl"
ROWS_FILENAME = "predict_rows.jsonl"
MANIFEST_FILENAME = "manifest.json"

HYDRA_PREDICT_FOLDER_KEY = "data_loading.predict_dataloader_config.dataloader.data_folder"
HYDRA_PICKLE_DIR_KEY = "callbacks.pickle_writer.output_dir"
HYDRA_CKPT = "ckpt_path='$CKPT'"
HYDRA_WORKERS = "data_loading.predict_dataloader_config.dataloader.num_workers=0"
HYDRA_TIMEOUT = "data_loading.predict_dataloader_config.dataloader.timeout=0"
RANKING_SOURCE = "leak_safe_history"


def encoder_item_window(
    sequence_length: int = SEQUENCE_LENGTH,
    num_hierarchies: int = NUM_HIERARCHIES,
) -> int:
    """How many history items survive ``pad_or_trim_sequence`` on the token ids.

    Each item becomes ``num_hierarchies`` tokens before the trim. The trim
    keeps the tail and the Beauty config uses a length that divides evenly,
    so the window is a whole number of items.
    """
    if int(sequence_length) < 1 or int(num_hierarchies) < 1:
        raise ValueError("sequence_length and num_hierarchies must be >= 1")
    if int(sequence_length) % int(num_hierarchies) != 0:
        raise ValueError(
            f"sequence_length {sequence_length} is not a multiple of num_hierarchies {num_hierarchies}; "
            "the tail trim would split a semantic-id group"
        )
    return int(sequence_length) // int(num_hierarchies)


def history_only_rows(samples) -> list[dict]:
    """One GRID record per eval sample. ``sequence_data`` is history only."""
    rows = []
    seen: set[str] = set()
    for sample in samples:
        target_event_in_history(sample)
        request_id = str(sample["request_id"])
        if request_id in seen:
            raise ValueError(f"duplicate request_id {request_id}")
        seen.add(request_id)
        history = [normalize_item(item) for item in sample["history"]]
        target_position = int(sample["target_position"])
        if len(history) != target_position:
            raise ValueError(
                f"{request_id} history length {len(history)} != target_position {target_position}"
            )
        for item in history:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{request_id} history items must be non-negative ints, got {item!r}")
        sequence = list(history)
        if sequence != history or len(sequence) != target_position:
            raise ValueError(f"{request_id} predict sequence must be the exported history")
        rows.append(
            {
                "predict_key": len(rows),
                "request_id": request_id,
                "user_id": str(sample["user_id"]),
                "split": str(sample["split"]),
                "sequence_data": sequence,
                "n_history": len(sequence),
            }
        )
    if not rows:
        raise ValueError("no eval samples to encode")
    return rows


def _known_item_ids(sid_to_item: dict) -> set:
    return set(sid_to_item.values())


def assert_items_in_sid_map(rows, sid_to_item: dict) -> None:
    known = _known_item_ids(sid_to_item)
    missing = []
    for row in rows:
        for item in row["sequence_data"]:
            if item not in known and len(missing) < 8:
                missing.append({"request_id": row["request_id"], "item_id": item})
            elif item not in known:
                break
        if len(missing) >= 8:
            break
    if missing:
        raise ValueError(f"history item ids missing from the semantic-id map: {missing}")


def predict_manifest(rows, *, sequence_length: int, num_hierarchies: int, sid_checked: bool) -> dict:
    window = encoder_item_window(sequence_length, num_hierarchies)
    n_long = sum(1 for row in rows if row["n_history"] > window)
    return {
        "format": "grid_tfrecord_gz",
        "grid_experiment": "tiger_inference_flat",
        "features": ["sequence_data", "user_id"],
        "sequence_data": "history item ids only; the target event is not appended",
        "user_id": "predict_key in 0..n-1, not the raw user id",
        "includes_target_event": False,
        "n_requests": len(rows),
        "n_empty_history": sum(1 for row in rows if row["n_history"] == 0),
        "sequence_length_tokens": int(sequence_length),
        "num_hierarchies": int(num_hierarchies),
        "max_items_kept_by_collate": window,
        "n_history_longer_than_encoder_window": n_long,
        "collate": "collate_fn_inference_for_sequence keeps the token tail; the target is absent",
        "hydra_data_folder_key": HYDRA_PREDICT_FOLDER_KEY,
        "hydra_pickle_dir_key": HYDRA_PICKLE_DIR_KEY,
        "sid_map_checked": bool(sid_checked),
        "ranking_source": RANKING_SOURCE,
        "do_not_score_2026_10_17_tiger_inference_flat_pickles": True,
    }


def write_predict_dataset(
    output_dir,
    rows,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    num_hierarchies: int = NUM_HIERARCHIES,
    sid_checked: bool = False,
) -> dict[str, str]:
    """Write the TFRecord folder, the request index, and a manifest."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = [
        {"sequence_data": list(row["sequence_data"]), "user_id": [int(row["predict_key"])]}
        for row in rows
    ]
    tfrecord_path = root / PREDICT_FILENAME
    write_tfrecord_gz(tfrecord_path, records)
    written_back = read_tfrecord_gz(tfrecord_path)
    if len(written_back) != len(records):
        raise ValueError(f"read back {len(written_back)} TFRecords, wrote {len(records)}")
    for original, loaded in zip(records, written_back):
        if loaded.get("sequence_data", []) != list(original["sequence_data"]):
            raise ValueError("TFRecord sequence_data does not match history")
        if loaded.get("user_id") != list(original["user_id"]):
            raise ValueError("TFRecord user_id does not match predict_key")
        if len(loaded.get("sequence_data", [])) != len(original["sequence_data"]):
            raise ValueError("TFRecord sequence length changed")
    index_rows = [
        {
            "predict_key": row["predict_key"],
            "request_id": row["request_id"],
            "user_id": row["user_id"],
            "split": row["split"],
            "n_history": row["n_history"],
        }
        for row in rows
    ]
    index_path = root / INDEX_FILENAME
    rows_path = root / ROWS_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    write_jsonl(index_path, index_rows)
    write_jsonl(rows_path, rows)
    manifest = predict_manifest(
        rows,
        sequence_length=sequence_length,
        num_hierarchies=num_hierarchies,
        sid_checked=sid_checked,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "tfrecord": str(tfrecord_path),
        "predict_index": str(index_path),
        "predict_rows": str(rows_path),
        "manifest": str(manifest_path),
    }


def _as_codes(value) -> list[tuple]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        raise ValueError("prediction is missing semantic_ids")
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("semantic_ids must be a list of codes")
    if value and not isinstance(value[0], (list, tuple)):
        raise ValueError("semantic_ids must be a list of codes, one per ranked item")
    codes = []
    for code in value:
        if hasattr(code, "tolist"):
            code = code.tolist()
        codes.append(tuple(int(part) for part in code))
    return codes


def _prediction_rows(payload) -> list[dict]:
    if hasattr(payload, "detach"):
        payload = payload.detach().cpu()
    if hasattr(payload, "tolist") and not isinstance(payload, (list, tuple, dict)):
        payload = payload.tolist()
    if isinstance(payload, dict):
        rows = []
        for key, value in payload.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("user_id", key)
            else:
                row = {"user_id": key, "semantic_ids": value}
            rows.append(row)
        return rows
    if isinstance(payload, list):
        if payload and not isinstance(payload[0], dict) and isinstance(payload[0], (list, tuple)):
            return [{"user_id": index, "semantic_ids": codes} for index, codes in enumerate(payload)]
        return list(payload)
    raise ValueError("predictions must be a pickle list, a JSONL list, or a tensor indexed by predict_key")


def load_prediction_rows(path) -> list[dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pkl":
        import pickle

        with file_path.open("rb") as handle:
            payload = pickle.load(handle)
        return _prediction_rows(payload)
    if suffix == ".jsonl":
        return _prediction_rows(read_jsonl(file_path))
    if suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Reading merged_predictions_tensor.pt needs torch. Pass the merged_predictions.pkl list instead."
            ) from exc
        payload = torch.load(file_path, map_location="cpu")
        return _prediction_rows(payload)
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return _prediction_rows(payload)
    raise ValueError(f"unsupported prediction file {file_path}")


def index_by_predict_key(rows) -> dict[int, dict]:
    found = {}
    for row in rows:
        key = int(row["predict_key"])
        if key in found:
            raise ValueError(f"duplicate predict_key {key}")
        found[key] = row
    if not found:
        raise ValueError("predict index is empty")
    return found


def rankings_from_predictions(predictions, index: dict[int, dict], sid_to_item: dict, *, beam: int) -> tuple[list[dict], int]:
    """Map GRID ``user_id`` + ``semantic_ids`` rows onto ``request_id`` + ``ranked_items``."""
    beam = int(beam)
    if beam not in BEAM_WIDTHS:
        raise ValueError(f"beam must be one of {BEAM_WIDTHS}, got {beam}")
    if not sid_to_item:
        raise ValueError("a semantic-id map is required to decode beam codes")
    ranked_by_key: dict[int, dict] = {}
    n_unknown = 0
    for raw in predictions:
        if not isinstance(raw, dict):
            raise ValueError("each prediction row must be an object")
        if raw.get("user_id") in (None, ""):
            raise ValueError("prediction row is missing user_id (the predict_key)")
        key = int(raw["user_id"])
        if key not in index:
            raise ValueError(f"prediction user_id {key} is not in the predict index")
        if key in ranked_by_key:
            raise ValueError(f"duplicate prediction for predict_key {key}")
        codes = _as_codes(raw.get("semantic_ids"))
        items = []
        for code in codes:
            item = sid_to_item.get(code)
            if item is None:
                n_unknown += 1
                continue
            items.append(item)
        meta = index[key]
        ranked_by_key[key] = {
            "request_id": meta["request_id"],
            "user_id": meta["user_id"],
            "split": meta["split"],
            "ranked_items": dedupe_keep_order(items),
            "beam": beam,
            "n_unknown_codes": sum(1 for code in codes if code not in sid_to_item),
        }
    missing = [key for key in index if key not in ranked_by_key]
    if missing:
        shown = missing[:8]
        raise ValueError(f"{len(missing)} predict keys have no ranking, including {shown}")
    ordered = [ranked_by_key[key] for key in sorted(ranked_by_key)]
    return ordered, n_unknown


def synthetic_sid_map(samples) -> dict[tuple, object]:
    items = sorted({item for sample in samples for item in [*sample["history"], sample["target"]]}, key=lambda value: (isinstance(value, str), value))
    mapping = {}
    for index, item in enumerate(items):
        code = (int(index), 0, 0, 0)
        mapping[code] = item
    return mapping


def synthetic_grid_predictions(rows, sid_to_item: dict, *, beam: int) -> list[dict]:
    """Smoke predictions keyed by ``predict_key``.

    Beam 20 is filler codes. Wider beams put the row's ``target`` code first.
    ``target`` is not written into ``sequence_data``; callers attach it only in memory.
    """
    item_to_code = {item: code for code, item in sid_to_item.items()}
    filler = next(iter(item_to_code.values()))
    predictions = []
    for row in rows:
        target = row.get("target")
        if int(beam) == 20 or target not in item_to_code:
            codes = [list(filler) for _ in range(int(beam))]
        else:
            codes = [list(item_to_code[target]), *[list(filler) for _ in range(int(beam) - 1)]]
        predictions.append({"user_id": int(row["predict_key"]), "semantic_ids": codes})
    return predictions


def _rows_with_targets_for_smoke(samples, rows) -> list[dict]:
    """Attach targets only for the in-memory smoke. They are not written to TFRecords."""
    by_request = {str(sample["request_id"]): sample for sample in samples}
    attached = []
    for row in rows:
        copied = dict(row)
        copied["target"] = by_request[row["request_id"]]["target"]
        attached.append(copied)
    return attached


def command_build_leak_safe_predict(
    *,
    smoke: bool,
    samples: str | None = None,
    output_dir: str | None = None,
    semantic_id_path: str | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    num_hierarchies: int = NUM_HIERARCHIES,
    dry_run: bool = False,
) -> dict:
    command = "build_leak_safe_predict"
    try:
        if smoke:
            loaded = synthetic_samples()
            source = "smoke"
        else:
            if not samples:
                return _error(command, "Pass --samples, or --smoke for a synthetic dry-run.")
            loaded = read_jsonl(samples)
            source = "files"
        rows = history_only_rows(loaded)
        sid_checked = False
        if semantic_id_path:
            assert_items_in_sid_map(rows, load_sid_map(semantic_id_path))
            sid_checked = True
        manifest = predict_manifest(
            rows,
            sequence_length=sequence_length,
            num_hierarchies=num_hierarchies,
            sid_checked=sid_checked,
        )
        written = {}
        if output_dir and not dry_run:
            written = write_predict_dataset(
                output_dir,
                rows,
                sequence_length=sequence_length,
                num_hierarchies=num_hierarchies,
                sid_checked=sid_checked,
            )
        return {
            "command": command,
            "ok": True,
            "exit_code": 0,
            "status": "smoke" if smoke else "built",
            "scripted": bool(smoke),
            "dry_run": bool(dry_run),
            "source": source,
            "n": len(rows),
            "n_target_event_in_history": 0,
            "includes_target_event": False,
            "n_empty_history": manifest["n_empty_history"],
            "n_history_longer_than_encoder_window": manifest["n_history_longer_than_encoder_window"],
            "max_items_kept_by_collate": manifest["max_items_kept_by_collate"],
            "max_predict_key": rows[-1]["predict_key"],
            "hydra_data_folder_key": HYDRA_PREDICT_FOLDER_KEY,
            "ranking_source": RANKING_SOURCE,
            "output": written,
            "benefit_aware_claimed": False,
        }
    except (OSError, ValueError, RuntimeError) as exc:
        return _error(command, str(exc))


def command_export_leak_safe_rankings(
    *,
    smoke: bool,
    predictions: str | None = None,
    index: str | None = None,
    sid_map: str | None = None,
    beam: int | None = None,
    output: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "export_leak_safe_rankings"
    try:
        if smoke:
            loaded = synthetic_samples()
            rows = _rows_with_targets_for_smoke(loaded, history_only_rows(loaded))
            sid_to_item = synthetic_sid_map(loaded)
            beam_width = 60 if beam is None else int(beam)
            prediction_rows = synthetic_grid_predictions(rows, sid_to_item, beam=beam_width)
            index_map = index_by_predict_key(rows)
            source = "smoke"
        else:
            if not predictions or not index or not sid_map or beam is None:
                return _error(
                    command,
                    "Pass --predictions, --index, --sid-map, and --beam. Use --smoke for a synthetic dry-run.",
                )
            prediction_rows = load_prediction_rows(predictions)
            index_map = index_by_predict_key(read_jsonl(index))
            sid_to_item = load_sid_map(sid_map)
            beam_width = int(beam)
            source = "files"
        ranked, n_unknown = rankings_from_predictions(
            prediction_rows,
            index_map,
            sid_to_item,
            beam=beam_width,
        )
        written = None
        if output and not dry_run:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(destination, ranked)
            written = str(destination)
        return {
            "command": command,
            "ok": True,
            "exit_code": 0,
            "status": "smoke" if smoke else "exported",
            "scripted": bool(smoke),
            "dry_run": bool(dry_run),
            "source": source,
            "beam": beam_width,
            "n": len(ranked),
            "n_unknown_codes": n_unknown,
            "ranking_source": RANKING_SOURCE,
            "includes_target_event": False,
            "output": written,
            "benefit_aware_claimed": False,
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
