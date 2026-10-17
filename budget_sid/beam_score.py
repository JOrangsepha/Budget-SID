"""Per-request NDCG for fixed beams 20, 40, and 60.

The cutoff stays 20 for the budget label. @5 and @10 are reported on the same
ranked list so they can be compared with the GRID training log. Beam width is
the search width, not the metric cutoff: a beam-60 list is still scored at 20.

``delta_ndcg@20`` is NDCG@20(beam 60) minus NDCG@20(beam 20). That is the
oracle label P1 can regress. This module does not fit the benefit model and
does not claim a benefit-aware gain.
"""

from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path

from budget_sid.constants import BEAM_WIDTHS
from budget_sid.eval_samples import (
    canonical_split,
    normalize_item,
    normalize_user_id,
    read_jsonl,
    synthetic_samples,
    write_jsonl,
)
from budget_sid.metrics import retrieval_at_ks

KS = (5, 10, 20)
RANKING_SOURCES = ("unspecified", "leak_safe_history", "grid_tiger_inference_flat")
_BEAM_IN_NAME = re.compile(r"(?:beam|top_k|topk|top)[_-]?(\d+)", re.IGNORECASE)
_RANKING_FIELDS = ("ranked_items", "predicted", "ranking", "recommendations")


def dedupe_keep_order(items) -> list:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def score_ranking(ranked, target, ks=KS) -> dict:
    predicted = dedupe_keep_order(list(ranked))
    scores = retrieval_at_ks(predicted, [target], ks=ks)
    try:
        scores["rank"] = predicted.index(target) + 1
    except ValueError:
        scores["rank"] = None
    return scores


def synthetic_rankings(samples, beam: int) -> list[dict]:
    """Rankings built from the exported history's target, not from a leaked input.

    Beam 20 misses. Beam 40 places the target at rank 6. Beam 60 places it first.
    """
    if int(beam) not in BEAM_WIDTHS:
        raise ValueError(f"synthetic rankings support beams {BEAM_WIDTHS}")
    rows = []
    for sample in samples:
        target = sample["target"]
        filler = [item for item in range(100, 160) if item != target]
        if int(beam) == 20:
            ranked = filler[:20]
        elif int(beam) == 40:
            ranked = filler[:5] + [target] + filler[5:19]
        else:
            ranked = [target, *filler[:20]]
        rows.append({"request_id": sample["request_id"], "ranked_items": ranked, "beam": int(beam)})
    return rows


def load_sid_map(path) -> dict[tuple, object]:
    """Map a semantic-id tuple to one item id.

    JSONL rows use ``item_id`` and ``semantic_id``. A JSON object may map the
    item id to its code list. A ``.pt`` tensor is read when torch is installed;
    shape ``[D, N]`` or ``[N, D]`` with D <= 8.
    """
    file_path = Path(path)
    if file_path.suffix == ".pt":
        return _sid_map_from_tensor(file_path)
    mapping: dict[tuple, object] = {}
    if file_path.suffix == ".jsonl":
        entries = read_jsonl(file_path)
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "item_id" not in payload and "semantic_id" not in payload:
            entries = [
                {"item_id": item_id, "semantic_id": code}
                for item_id, code in payload.items()
            ]
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = [payload]
    for entry in entries:
        code = tuple(int(part) for part in entry["semantic_id"])
        item_id = normalize_item(entry["item_id"])
        if code in mapping and mapping[code] != item_id:
            raise ValueError(f"semantic id {code} maps to both {mapping[code]} and {item_id}")
        mapping[code] = item_id
    if not mapping:
        raise ValueError(f"no semantic ids in {file_path}")
    return mapping


def codes_from_sid_matrix(array) -> list[tuple]:
    """Read item codes from a ``[D, N]`` or ``[N, D]`` matrix. D is at most 8."""
    if not array or not isinstance(array[0], (list, tuple)):
        raise ValueError("semantic-id matrix must be 2D")
    rows = len(array)
    cols = len(array[0])
    if rows <= 8 and cols > rows:
        codes = list(zip(*array))
    elif cols <= 8:
        codes = array
    else:
        raise ValueError(f"shape {[rows, cols]} does not look like semantic ids")
    return [tuple(int(part) for part in code) for code in codes]


def _sid_map_from_tensor(path: Path) -> dict[tuple, object]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Reading a .pt semantic-id map needs torch. On SeetaCloud, run inside the GRID env, "
            "or pass a JSONL map of item_id and semantic_id."
        ) from exc
    tensor = torch.load(path, map_location="cpu")
    codes = codes_from_sid_matrix(tensor.detach().cpu().tolist())
    mapping = {}
    for item_id, code in enumerate(codes):
        if code in mapping:
            raise ValueError(f"semantic id {code} collides for items {mapping[code]} and {item_id}")
        mapping[code] = item_id
    return mapping


def load_rankings(path, *, sid_to_item=None, beam: int | None = None) -> list[dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pkl":
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)
        raw_rows = list(payload)
    elif suffix == ".jsonl":
        raw_rows = read_jsonl(file_path)
    elif suffix == ".csv":
        raw_rows = _rankings_from_csv(file_path)
    elif suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        raw_rows = payload if isinstance(payload, list) else payload.get("rows", [])
    else:
        raise ValueError(f"unsupported ranking file {file_path}")
    rows = []
    for raw in raw_rows:
        rows.append(_normalize_ranking_row(raw, sid_to_item=sid_to_item, beam=beam))
    return rows


def rankings_from_directory(path, *, sid_to_item=None) -> dict[int, list[dict]]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    found: dict[int, list[dict]] = {}
    for file_path in sorted(root.iterdir()):
        if not file_path.is_file():
            continue
        match = _BEAM_IN_NAME.search(file_path.stem)
        if not match:
            continue
        beam = int(match.group(1))
        if beam not in BEAM_WIDTHS:
            continue
        found[beam] = load_rankings(file_path, sid_to_item=sid_to_item, beam=beam)
    if not found:
        raise ValueError(
            f"no beam_20/beam_40/beam_60 ranking files in {root}. "
            "Name them beam_20.jsonl, beam_40.jsonl, and beam_60.jsonl."
        )
    return found


def _rankings_from_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    if not raw_rows:
        return []
    fields = set(raw_rows[0])
    if fields.intersection(_RANKING_FIELDS) or "semantic_ids" in fields:
        parsed = []
        for raw in raw_rows:
            row = dict(raw)
            for key in (*_RANKING_FIELDS, "semantic_ids"):
                if row.get(key):
                    row[key] = _parse_cell_list(row[key])
            parsed.append(row)
        return parsed
    if "item_id" not in fields:
        raise ValueError(f"{path} needs ranked_items or item_id")
    grouped: dict[tuple, dict] = {}
    order: list[tuple] = []
    for raw in raw_rows:
        identity = (raw.get("request_id") or "", raw.get("user_id") or "", raw.get("split") or "", raw.get("beam") or "")
        if identity not in grouped:
            grouped[identity] = {
                "request_id": raw.get("request_id") or None,
                "user_id": raw.get("user_id") or None,
                "split": raw.get("split") or None,
                "beam": raw.get("beam") or None,
                "ranked_items": [],
            }
            order.append(identity)
        grouped[identity]["ranked_items"].append(raw["item_id"])
    return [grouped[key] for key in order]


def _parse_cell_list(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("["):
        return json.loads(text)
    separator = "|" if "|" in text else ","
    return [part.strip() for part in text.split(separator) if part.strip()]


def _normalize_ranking_row(raw, *, sid_to_item, beam: int | None) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("each ranking row must be an object")
    row_beam = raw.get("beam", raw.get("top_k", beam))
    if row_beam not in (None, ""):
        row_beam = int(row_beam)
        if beam is not None and row_beam != int(beam):
            raise ValueError(f"ranking beam {row_beam} does not match file beam {beam}")
    ranked, n_unknown = _ranked_items(raw, sid_to_item)
    parsed = {"ranked_items": ranked, "n_unknown_codes": n_unknown}
    if raw.get("request_id") not in (None, ""):
        parsed["request_id"] = str(_coerce_scalar(raw["request_id"]))
    if raw.get("user_id") not in (None, ""):
        parsed["user_id"] = normalize_user_id(_coerce_scalar(raw["user_id"]))
    if raw.get("split") not in (None, ""):
        parsed["split"] = canonical_split(str(_coerce_scalar(raw["split"])))
    if row_beam not in (None, ""):
        parsed["beam"] = int(row_beam)
    if "request_id" not in parsed and "user_id" not in parsed:
        raise ValueError("ranking row needs request_id or user_id")
    return parsed


def _ranked_items(raw, sid_to_item) -> tuple[list, int]:
    for key in _RANKING_FIELDS:
        if raw.get(key) not in (None, ""):
            values = _coerce_scalar(raw[key])
            return [normalize_item(_coerce_scalar(item)) for item in values], 0
    if raw.get("semantic_ids") in (None, ""):
        raise ValueError("ranking row needs ranked_items or semantic_ids")
    if sid_to_item is None:
        raise ValueError("semantic_ids rankings need a sid map (--sid-map)")
    codes = _coerce_scalar(raw["semantic_ids"])
    if codes and not isinstance(codes[0], (list, tuple)):
        raise ValueError("semantic_ids must be a list of codes, one per ranked item")
    items = []
    unknown = 0
    for code in codes:
        key = tuple(int(part) for part in _coerce_scalar(code))
        if key not in sid_to_item:
            unknown += 1
            items.append(f"__unknown__:{key}")
        else:
            items.append(sid_to_item[key])
    return items, unknown


def _coerce_scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _index_samples(samples) -> dict:
    by_request = {}
    by_user_split = {}
    by_user: dict[str, list] = {}
    for sample in samples:
        request_id = str(sample["request_id"])
        if request_id in by_request:
            raise ValueError(f"duplicate sample request_id {request_id}")
        by_request[request_id] = sample
        key = (str(sample["user_id"]), str(sample["split"]))
        if key in by_user_split:
            raise ValueError(f"duplicate sample for user {key[0]} split {key[1]}")
        by_user_split[key] = sample
        by_user.setdefault(str(sample["user_id"]), []).append(sample)
    return {"request": by_request, "user_split": by_user_split, "user": by_user}


def _match_sample(row: dict, index: dict) -> dict:
    if row.get("request_id") and row["request_id"] in index["request"]:
        return index["request"][row["request_id"]]
    if row.get("user_id") and row.get("split"):
        key = (row["user_id"], row["split"])
        if key not in index["user_split"]:
            raise ValueError(f"no sample for user {key[0]} split {key[1]}")
        return index["user_split"][key]
    if row.get("user_id"):
        matches = index["user"].get(row["user_id"], [])
        if len(matches) != 1:
            raise ValueError(
                f"user {row['user_id']} matches {len(matches)} samples; pass request_id or split"
            )
        return matches[0]
    raise ValueError("ranking row needs request_id or user_id")


def score_beams(samples, rankings_by_beam, *, ranking_source: str = "unspecified") -> tuple[list[dict], dict]:
    if ranking_source not in RANKING_SOURCES:
        raise ValueError(f"unknown ranking_source {ranking_source!r}")
    index = _index_samples(samples)
    scored_by_request: dict[str, dict[int, dict]] = {}
    n_unknown = 0
    for beam, rows in rankings_by_beam.items():
        beam = int(beam)
        if beam not in BEAM_WIDTHS:
            raise ValueError(f"beam must be one of {BEAM_WIDTHS}, got {beam}")
        for row in rows:
            sample = _match_sample(row, index)
            request_id = sample["request_id"]
            bucket = scored_by_request.setdefault(request_id, {})
            if beam in bucket:
                raise ValueError(f"duplicate ranking for {request_id} beam {beam}")
            bucket[beam] = score_ranking(row["ranked_items"], sample["target"])
            n_unknown += int(row.get("n_unknown_codes") or 0)
    per_request = []
    for sample in samples:
        beams = {}
        for beam in BEAM_WIDTHS:
            beams[str(beam)] = scored_by_request.get(sample["request_id"], {}).get(beam)
        beam_20 = beams["20"]
        beam_60 = beams["60"]
        delta = None
        if beam_20 is not None and beam_60 is not None:
            delta = float(beam_60["ndcg@20"] - beam_20["ndcg@20"])
        per_request.append(
            {
                "request_id": sample["request_id"],
                "user_id": sample["user_id"],
                "split": sample["split"],
                "target": sample["target"],
                "target_position": sample["target_position"],
                "checkpoint_hash": sample.get("checkpoint_hash"),
                "semantic_id_hash": sample.get("semantic_id_hash"),
                "beams": beams,
                "delta_ndcg@20": delta,
            }
        )
    summary = summarize_beam_scores(per_request, ranking_source=ranking_source, n_unknown_codes=n_unknown)
    return per_request, summary


def summarize_beam_scores(per_request, *, ranking_source: str, n_unknown_codes: int = 0) -> dict:
    beams = {}
    for beam in BEAM_WIDTHS:
        present = [row["beams"][str(beam)] for row in per_request if row["beams"][str(beam)] is not None]
        entry = {"n": len(present)}
        for metric in ("ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20"):
            entry[metric] = _mean(row[metric] for row in present) if present else None
        beams[str(beam)] = entry
    deltas = [row["delta_ndcg@20"] for row in per_request if row["delta_ndcg@20"] is not None]
    auditable = ranking_source == "leak_safe_history"
    reasons = {
        "leak_safe_history": "Rankings were produced from history that excludes the target event.",
        "grid_tiger_inference_flat": (
            "tiger_inference_flat keeps the target event in the encoder input. "
            "These numbers are not a leak-safe ΔNDCG."
        ),
        "unspecified": "Set ranking_source to leak_safe_history before using delta_ndcg@20 as a P1 label.",
    }
    return {
        "ranking_source": ranking_source,
        "auditable_for_delta_ndcg": auditable,
        "auditable_reason": reasons[ranking_source],
        "benefit_aware_claimed": False,
        "n_requests": len(per_request),
        "n_unknown_codes": int(n_unknown_codes),
        "beams": beams,
        "delta_ndcg@20": {
            "n": len(deltas),
            "mean": _mean(deltas) if deltas else None,
            "min": float(min(deltas)) if deltas else None,
            "max": float(max(deltas)) if deltas else None,
        },
        "p1_next_step": (
            "Join per_request.jsonl on request_id. Fit the benefit model on val rows using "
            "delta_ndcg@20 only when auditable_for_delta_ndcg is true. Keep test events out."
        ),
    }


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("no values to average")
    return float(sum(values) / len(values))


def write_score_outputs(output_dir, per_request, summary) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "per_request.jsonl"
    csv_path = root / "per_request.csv"
    summary_json = root / "summary.json"
    summary_csv = root / "summary.csv"
    write_jsonl(jsonl_path, per_request)
    _write_per_request_csv(csv_path, per_request)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(summary_csv, summary)
    return {
        "per_request_jsonl": str(jsonl_path),
        "per_request_csv": str(csv_path),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
    }


def _write_per_request_csv(path: Path, per_request) -> None:
    fieldnames = [
        "request_id",
        "user_id",
        "split",
        "target",
        "target_position",
        "checkpoint_hash",
        "semantic_id_hash",
        "delta_ndcg@20",
    ]
    for beam in BEAM_WIDTHS:
        for metric in ("ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20", "rank"):
            fieldnames.append(f"{metric}_beam{beam}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_request:
            flat = {
                "request_id": row["request_id"],
                "user_id": row["user_id"],
                "split": row["split"],
                "target": row["target"],
                "target_position": row["target_position"],
                "checkpoint_hash": row.get("checkpoint_hash"),
                "semantic_id_hash": row.get("semantic_id_hash"),
                "delta_ndcg@20": row["delta_ndcg@20"],
            }
            for beam in BEAM_WIDTHS:
                scores = row["beams"][str(beam)] or {}
                for metric in ("ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20", "rank"):
                    flat[f"{metric}_beam{beam}"] = scores.get(metric)
            writer.writerow(flat)


def _write_summary_csv(path: Path, summary: dict) -> None:
    fieldnames = ["metric", "beam_20", "beam_40", "beam_60", "delta_60_minus_20"]
    metrics = ["n", "ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            row = {"metric": metric, "delta_60_minus_20": ""}
            for beam in BEAM_WIDTHS:
                row[f"beam_{beam}"] = summary["beams"][str(beam)].get(metric if metric != "n" else "n")
            if metric == "ndcg@20":
                row["delta_60_minus_20"] = summary["delta_ndcg@20"]["mean"]
            if metric == "n":
                row["delta_60_minus_20"] = summary["delta_ndcg@20"]["n"]
            writer.writerow(row)


def command_score_beam_ndcg(
    *,
    smoke: bool,
    samples: str | None = None,
    beam20: str | None = None,
    beam40: str | None = None,
    beam60: str | None = None,
    rankings_dir: str | None = None,
    output_dir: str | None = None,
    ranking_source: str = "unspecified",
    sid_map: str | None = None,
    dry_run: bool = False,
) -> dict:
    command = "score_beam_ndcg"
    try:
        if smoke:
            loaded = synthetic_samples()
            rankings = {beam: synthetic_rankings(loaded, beam) for beam in BEAM_WIDTHS}
            source = "leak_safe_history" if ranking_source == "unspecified" else ranking_source
        else:
            if not samples:
                return _error(command, "Pass --samples, or --smoke for a synthetic dry-run.")
            loaded = read_jsonl(samples)
            sid_to_item = load_sid_map(sid_map) if sid_map else None
            if rankings_dir:
                rankings = rankings_from_directory(rankings_dir, sid_to_item=sid_to_item)
            else:
                provided = {20: beam20, 40: beam40, 60: beam60}
                if not any(provided.values()):
                    return _error(
                        command,
                        "Pass --rankings-dir or --beam20/--beam40/--beam60. Use --smoke for a synthetic dry-run.",
                    )
                rankings = {}
                for beam, file_path in provided.items():
                    if file_path:
                        rankings[beam] = load_rankings(file_path, sid_to_item=sid_to_item, beam=beam)
            source = ranking_source
        per_request, summary = score_beams(loaded, rankings, ranking_source=source)
        written = {}
        if output_dir and not dry_run:
            written = write_score_outputs(output_dir, per_request, summary)
        return {
            "command": command,
            "ok": True,
            "exit_code": 0,
            "status": "smoke" if smoke else "scored",
            "scripted": bool(smoke),
            "dry_run": bool(dry_run),
            "ranking_source": summary["ranking_source"],
            "auditable_for_delta_ndcg": summary["auditable_for_delta_ndcg"],
            "n_requests": summary["n_requests"],
            "mean_ndcg@20_beam20": summary["beams"]["20"]["ndcg@20"],
            "mean_ndcg@20_beam40": summary["beams"]["40"]["ndcg@20"],
            "mean_ndcg@20_beam60": summary["beams"]["60"]["ndcg@20"],
            "mean_delta_ndcg@20": summary["delta_ndcg@20"]["mean"],
            "output": written,
            "benefit_aware_claimed": False,
            "p1_next_step": summary["p1_next_step"],
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
