"""Leak-safe eval samples for offline NDCG on Beauty beam rankings.

Each stored GRID sequence is treated the way ``NextKTokenMasking`` treats it:
the last interaction event is the target, and history is the strict prefix.
The same item id earlier in that prefix is a repeated purchase and is kept.
The target event's own index is not.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from budget_sid.data_audit import assert_target_event_not_in_history

SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "evaluation": "val",
    "validation": "val",
    "val": "val",
    "testing": "test",
    "test": "test",
}
SEQUENCE_FIELDS = ("sequence_data", "sequence", "item_ids", "items")
USER_FIELDS = ("user_id", "userId", "user")
SAMPLE_FIELDS = (
    "request_id",
    "user_id",
    "split",
    "history",
    "target",
    "target_position",
    "checkpoint_hash",
    "semantic_id_hash",
    "repeated_target_item",
    "history_positions",
    "source_sequence",
)
DATA_SUFFIXES = (".jsonl", ".json", ".csv", ".tfrecord", ".tfrecord.gz")


def canonical_split(split: str) -> str:
    key = str(split).strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"unknown split {split!r}")
    return SPLIT_ALIASES[key]


def make_request_id(split: str, user_id, target_position: int) -> str:
    return f"{canonical_split(split)}:{user_id}:{int(target_position)}"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_hash(*, file_path, explicit_hash: str | None, label: str) -> str:
    if file_path:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
        hashed = sha256_file(path)
        if explicit_hash and str(explicit_hash) != hashed:
            raise ValueError(f"{label} hash does not match {path}")
        return hashed
    if explicit_hash:
        return str(explicit_hash)
    raise ValueError(f"pass a {label} file or an explicit {label} hash")


def normalize_item(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def normalize_user_id(value) -> str:
    value = _unwrap_singleton(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return str(value).strip()


def build_eval_sample(
    *,
    user_id,
    split,
    checkpoint_hash: str,
    semantic_id_hash: str,
    sequence=None,
    history=None,
    target=None,
    target_position: int | None = None,
    request_id: str | None = None,
) -> dict:
    """Build one eval record and reject a history that still holds the target event."""
    if not checkpoint_hash or not semantic_id_hash:
        raise ValueError("checkpoint_hash and semantic_id_hash are required")
    split_name = canonical_split(split)
    user = normalize_user_id(user_id)
    if sequence is not None:
        source = [normalize_item(item) for item in sequence]
        if not source:
            raise ValueError(f"user {user} has an empty sequence")
        canonical_position = len(source) - 1
        canonical_history = list(source[:-1])
        canonical_target = source[-1]
        if history is not None or target is not None or target_position is not None:
            supplied_history = canonical_history if history is None else [normalize_item(item) for item in history]
            supplied_target = canonical_target if target is None else normalize_item(target)
            supplied_position = canonical_position if target_position is None else int(target_position)
            if (
                supplied_history != canonical_history
                or supplied_target != canonical_target
                or supplied_position != canonical_position
            ):
                if supplied_position == canonical_position and supplied_history == source:
                    assert_target_event_not_in_history(range(len(supplied_history)), supplied_position)
                raise ValueError("history, target, and target_position disagree with the sequence")
        history = canonical_history
        target = canonical_target
        target_position = canonical_position
        source_sequence = source
    else:
        if history is None or target is None:
            raise ValueError("pass a sequence, or history and target")
        history = [normalize_item(item) for item in history]
        target = normalize_item(target)
        if target_position is None:
            target_position = len(history)
        target_position = int(target_position)
        source_sequence = list(history) + [target]

    if target_position < 0:
        raise ValueError("target_position must be >= 0")
    if len(history) > target_position:
        assert_target_event_not_in_history(range(len(history)), target_position)
    if len(history) != target_position:
        raise ValueError(
            f"history length {len(history)} != target_position {target_position}; "
            "history must be the strict prefix before the target event"
        )
    history_positions = list(range(target_position))
    assert_target_event_not_in_history(history_positions, target_position)
    if list(source_sequence[:target_position]) != list(history) or source_sequence[target_position] != target:
        raise ValueError("source sequence does not match history and target")
    return {
        "request_id": str(request_id) if request_id else make_request_id(split_name, user, target_position),
        "user_id": user,
        "split": split_name,
        "history": list(history),
        "target": target,
        "target_position": int(target_position),
        "checkpoint_hash": str(checkpoint_hash),
        "semantic_id_hash": str(semantic_id_hash),
        "repeated_target_item": target in history,
        "history_positions": history_positions,
        "source_sequence": list(source_sequence),
    }


def samples_from_rows(rows, *, checkpoint_hash: str, semantic_id_hash: str) -> list[dict]:
    samples = []
    seen: set[str] = set()
    for row in rows:
        sample = build_eval_sample(
            user_id=row["user_id"],
            split=row["split"],
            checkpoint_hash=checkpoint_hash,
            semantic_id_hash=semantic_id_hash,
            sequence=row.get("sequence"),
            history=row.get("history"),
            target=row.get("target"),
            target_position=row.get("target_position"),
            request_id=row.get("request_id"),
        )
        if sample["request_id"] in seen:
            raise ValueError(f"duplicate request_id {sample['request_id']}")
        seen.add(sample["request_id"])
        samples.append(sample)
    if not samples:
        raise ValueError("no eval rows to export")
    return samples


def synthetic_loo_rows() -> list[dict]:
    """One user in all three splits, plus a test-only user.

    User ``a`` repeats item 7. The test target is that second purchase.
    Prefixes match temporal leave-one-out: val is test without the last
    event, train is val without the last event.
    """
    full = [1, 7, 2, 3, 4, 7]
    return [
        {"user_id": "a", "split": "train", "sequence": full[:-2]},
        {"user_id": "a", "split": "val", "sequence": full[:-1]},
        {"user_id": "a", "split": "test", "sequence": list(full)},
        {"user_id": "b", "split": "test", "sequence": [5, 6, 8]},
    ]


def synthetic_hashes() -> dict[str, str]:
    return {
        "checkpoint_hash": sha256_text("smoke-checkpoint"),
        "semantic_id_hash": sha256_text("smoke-semantic-ids"),
    }


def synthetic_samples() -> list[dict]:
    hashes = synthetic_hashes()
    return samples_from_rows(synthetic_loo_rows(), **hashes)


def target_event_in_history(sample: dict) -> None:
    """Raise when a record's history still contains the target event.

    A correct export has ``history_positions == range(target_position)``.
    A hand edit that puts the target index back into those positions, or that
    extends ``history`` past ``target_position``, fails this check.
    """
    target_position = int(sample["target_position"])
    history = list(sample["history"])
    positions = sample.get("history_positions")
    if positions is None:
        positions = list(range(len(history)))
    else:
        positions = [int(index) for index in positions]
    if len(history) > target_position:
        positions = list(dict.fromkeys([*positions, *range(len(history))]))
    assert_target_event_not_in_history(positions, target_position)


def iter_leak_failures(samples) -> list[dict]:
    from budget_sid.errors import HistoryLeakError

    failures = []
    for sample in samples:
        try:
            target_event_in_history(sample)
        except HistoryLeakError as exc:
            failures.append({"request_id": sample.get("request_id"), "error": str(exc)})
    return failures


def write_jsonl(path, rows) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not JSON") from exc
    return rows


def write_samples_csv(path, samples) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SAMPLE_FIELDS)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {key: sample[key] for key in fieldnames}
            for key in ("history", "history_positions", "source_sequence"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def write_samples(output_dir, samples) -> dict[str, str]:
    root = Path(output_dir)
    jsonl_path = root / "eval_samples.jsonl"
    csv_path = root / "eval_samples.csv"
    write_jsonl(jsonl_path, samples)
    write_samples_csv(csv_path, samples)
    return {"eval_samples_jsonl": str(jsonl_path), "eval_samples_csv": str(csv_path)}


def load_sequence_rows(path) -> list[dict]:
    """Load item sequences from JSONL, CSV, JSON, or gzip TFRecord splits.

    A directory that contains ``training`` / ``evaluation`` / ``testing``
    (or ``train`` / ``validation`` / ``test``) is read per split. TFRecord
    files need TensorFlow, which the GRID environment on SeetaCloud has.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return _load_file(root, default_split=None)
    split_dirs = [child for child in root.iterdir() if child.is_dir() and child.name.lower() in SPLIT_ALIASES]
    split_dirs.sort(key=lambda child: ("train", "val", "test").index(canonical_split(child.name)))
    rows: list[dict] = []
    if split_dirs:
        for child in split_dirs:
            split = canonical_split(child.name)
            for file_path in _data_files(child):
                rows.extend(_load_file(file_path, default_split=split))
        return rows
    for file_path in _data_files(root):
        rows.extend(_load_file(file_path, default_split=None))
    return rows


def _data_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in DATA_SUFFIXES):
            files.append(path)
    return files


def _load_file(path: Path, *, default_split: str | None) -> list[dict]:
    name = path.name.lower()
    if name.endswith(".jsonl"):
        return [_row_from_mapping(row, split=default_split) for row in read_jsonl(path)]
    if name.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("samples", [payload]))
        if not isinstance(payload, list):
            raise ValueError(f"{path} JSON must be a list of rows")
        return [_row_from_mapping(row, split=default_split) for row in payload]
    if name.endswith(".csv"):
        return _load_csv(path, default_split)
    if name.endswith(".tfrecord") or name.endswith(".tfrecord.gz"):
        return [_row_from_mapping(row, split=default_split) for row in _iter_tfrecord(path)]
    raise ValueError(f"unsupported eval file {path}")


def _load_csv(path: Path, default_split: str | None) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    if not raw_rows:
        return []
    fields = set(raw_rows[0])
    if "item_id" in fields and not fields.intersection(SEQUENCE_FIELDS) and "history" not in fields:
        grouped: dict[str, list] = {}
        order: list[str] = []
        meta: dict[str, dict] = {}
        for raw in raw_rows:
            user = raw["user_id"]
            if user not in grouped:
                grouped[user] = []
                order.append(user)
                meta[user] = raw
            grouped[user].append(raw["item_id"])
        loaded = []
        for user in order:
            mapping = dict(meta[user])
            mapping["sequence"] = grouped[user]
            loaded.append(_row_from_mapping(mapping, split=default_split))
        return loaded
    return [_row_from_mapping(raw, split=default_split) for raw in raw_rows]


def _iter_tfrecord(path: Path):
    tensorflow = _require_tensorflow()
    compression = "GZIP" if path.name.lower().endswith(".gz") else ""
    dataset = tensorflow.data.TFRecordDataset([str(path)], compression_type=compression)
    iterator = iter(dataset)
    try:
        first = next(iterator)
    except StopIteration:
        return
    example = tensorflow.train.Example()
    example.ParseFromString(bytes(first.numpy()))
    feature_description = {}
    for key, value in example.features.feature.items():
        if value.HasField("bytes_list"):
            feature_description[key] = tensorflow.io.VarLenFeature(tensorflow.string)
        elif value.HasField("float_list"):
            feature_description[key] = tensorflow.io.VarLenFeature(tensorflow.float32)
        elif value.HasField("int64_list"):
            feature_description[key] = tensorflow.io.VarLenFeature(tensorflow.int64)
        else:
            raise ValueError(f"unsupported TFRecord feature {key} in {path}")

    def parse(raw) -> dict:
        parsed = tensorflow.io.parse_single_example(raw, feature_description)
        decoded = {}
        for key, tensor in parsed.items():
            decoded[key] = tensorflow.sparse.to_dense(tensor).numpy().tolist()
        return decoded

    yield parse(first)
    for raw in iterator:
        yield parse(raw)


def _require_tensorflow():
    try:
        import tensorflow as tensorflow
    except ImportError as exc:
        raise RuntimeError(
            "Reading .tfrecord.gz needs tensorflow, which is installed in the GRID "
            "environment on SeetaCloud. Convert the split to JSONL, or run this "
            "command on the GPU machine (/root/autodl-tmp/Budget-SID)."
        ) from exc
    return tensorflow


def _row_from_mapping(mapping, *, split: str | None) -> dict:
    if not isinstance(mapping, dict):
        raise ValueError("each eval row must be an object")
    data = dict(mapping)
    user = None
    for key in USER_FIELDS:
        if data.get(key) not in (None, ""):
            user = data[key]
            break
    if user is None:
        raise ValueError("row is missing user_id")
    user_id = normalize_user_id(user)
    row_split = data.get("split", split)
    if row_split in (None, ""):
        raise ValueError(f"user {user_id} has no split; put the file under training/evaluation/testing or set split")
    parsed = {"user_id": user_id, "split": canonical_split(str(row_split))}
    if data.get("request_id"):
        parsed["request_id"] = str(data["request_id"])
    if data.get("history") not in (None, "") and data.get("target") not in (None, ""):
        parsed["history"] = [_normalize_cell(item) for item in _as_list(data["history"])]
        parsed["target"] = normalize_item(_unwrap_singleton(data["target"]))
        if data.get("target_position") not in (None, ""):
            parsed["target_position"] = int(data["target_position"])
        return parsed
    sequence = None
    for key in SEQUENCE_FIELDS:
        if data.get(key) not in (None, ""):
            sequence = data[key]
            break
    if sequence is None:
        raise ValueError(f"user {user_id} has no sequence")
    parsed["sequence"] = [_normalize_cell(item) for item in _as_list(sequence)]
    return parsed


def _normalize_cell(value):
    return normalize_item(_unwrap_singleton(value))


def _unwrap_singleton(value):
    if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def _as_list(value) -> list:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return list(json.loads(text))
        separator = "|" if "|" in text else "," if "," in text else None
        if separator:
            return [part.strip() for part in text.split(separator) if part.strip()]
        return [part for part in text.split() if part]
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def command_export_eval_samples(
    *,
    smoke: bool,
    data_dir: str | None = None,
    sequences: str | None = None,
    output_dir: str | None = None,
    checkpoint: str | None = None,
    semantic_id_path: str | None = None,
    checkpoint_hash: str | None = None,
    semantic_id_hash: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "export_eval_samples"
    try:
        if smoke:
            samples = synthetic_samples()
            hashes = synthetic_hashes()
            source = "smoke"
        else:
            path = sequences or data_dir
            if not path:
                return _error(command, "Pass --data-dir or --sequences. Use --smoke for a synthetic dry-run.")
            checkpoint_hash = resolve_hash(
                file_path=checkpoint,
                explicit_hash=checkpoint_hash,
                label="checkpoint",
            )
            semantic_id_hash = resolve_hash(
                file_path=semantic_id_path,
                explicit_hash=semantic_id_hash,
                label="semantic_id",
            )
            samples = samples_from_rows(
                load_sequence_rows(path),
                checkpoint_hash=checkpoint_hash,
                semantic_id_hash=semantic_id_hash,
            )
            hashes = {"checkpoint_hash": checkpoint_hash, "semantic_id_hash": semantic_id_hash}
            source = "files"
        written = {}
        if output_dir and not dry_run:
            written = write_samples(output_dir, samples)
        failures = iter_leak_failures(samples)
        if failures:
            return _error(command, failures[0]["error"])
        return {
            "command": command,
            "ok": True,
            "exit_code": 0,
            "status": "smoke" if smoke else "exported",
            "scripted": bool(smoke),
            "dry_run": bool(dry_run),
            "source": source,
            "n": len(samples),
            "n_repeated_target_item": sum(1 for sample in samples if sample["repeated_target_item"]),
            "n_target_event_in_history": 0,
            "checkpoint_hash": hashes["checkpoint_hash"],
            "semantic_id_hash": hashes["semantic_id_hash"],
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
