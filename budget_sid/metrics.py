"""Full-catalog retrieval metrics, latency, and the two requested slices."""

from __future__ import annotations

import math

import numpy as np

from budget_sid.constants import TOP_K


def recall_at_k(predicted, relevant, k: int = TOP_K) -> float:
    truth = list(dict.fromkeys(relevant))
    if not truth:
        return 0.0
    hits = len(set(list(predicted)[:k]) & set(truth))
    return hits / len(truth)


def ndcg_at_k(predicted, relevant, k: int = TOP_K) -> float:
    truth = set(relevant)
    if not truth:
        return 0.0
    gains = [1.0 if item in truth else 0.0 for item in list(predicted)[:k]]
    ideal_hits = min(k, len(truth))
    ideal = [1.0] * ideal_hits
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(gains) / idcg


def _dcg(gains) -> float:
    total = 0.0
    for index, gain in enumerate(gains):
        if gain:
            total += float(gain) / math.log2(index + 2)
    return total


def latency_mean(latencies) -> float:
    values = np.asarray(list(latencies), dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency_mean requires at least one request")
    return float(values.mean())


def latency_p95(latencies) -> float:
    values = np.asarray(list(latencies), dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency_p95 requires at least one request")
    return float(np.percentile(values, 95, method="linear"))


def throughput(n_requests: int, elapsed_s: float) -> float:
    if n_requests < 0:
        raise ValueError("n_requests must be non-negative")
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return float(n_requests) / float(elapsed_s)


def peak_vram_mb(samples=None) -> float:
    """Peak reserved memory in MiB.

    CPU smoke has no CUDA allocator, so the value is the max of any probe
    samples, or 0 when the process never touched a GPU.
    """
    if samples is not None:
        values = list(samples)
        if values:
            return float(max(values))
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def score_retrieval(predicted, relevant, k: int = TOP_K, sampled_negatives: bool = False) -> dict:
    if sampled_negatives:
        raise ValueError(
            "sampled negatives are disabled; score every request against the full catalog"
        )
    return {
        "recall@20": recall_at_k(predicted, relevant, k=k),
        "ndcg@20": ndcg_at_k(predicted, relevant, k=k),
    }


def long_tail_items(frequencies: dict, quantile: float = 0.2) -> set:
    if not frequencies:
        return set()
    threshold = float(np.quantile(np.array(list(frequencies.values()), dtype=np.float64), quantile))
    return {item for item, freq in frequencies.items() if freq <= threshold}


def summarize_requests(records, k: int = TOP_K, short_history_max: int = 5) -> dict:
    """Mean metrics plus long-tail and short-history slices.

    Each record needs ``predicted``, ``relevant``, ``latency_s``,
    ``history_length`` and ``long_tail`` (bool).
    """
    if any(record.get("sampled_negatives") for record in records):
        raise ValueError("sampled negatives are disabled")
    return {
        "recall@20": _mean(records, lambda record: recall_at_k(record["predicted"], record["relevant"], k)),
        "ndcg@20": _mean(records, lambda record: ndcg_at_k(record["predicted"], record["relevant"], k)),
        "latency_mean": latency_mean(record["latency_s"] for record in records),
        "latency_p95": latency_p95(record["latency_s"] for record in records),
        "throughput": throughput(len(records), sum(record["latency_s"] for record in records)),
        "peak_vram_mb": peak_vram_mb(record.get("vram_mb", 0.0) for record in records),
        "n": len(records),
        "slices": {
            "long_tail": _slice(records, lambda record: bool(record["long_tail"]), k),
            "short_history": _slice(records, lambda record: record["history_length"] <= short_history_max, k),
        },
    }


def aggregate_seed_metrics(per_seed: list[dict]) -> dict:
    if len(per_seed) < 2:
        raise ValueError("report mean and spread over at least two seeds")
    keys = [key for key, value in per_seed[0].items() if isinstance(value, (int, float))]
    aggregated = {}
    for key in keys:
        values = np.array([seed[key] for seed in per_seed], dtype=np.float64)
        aggregated[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return aggregated


def _mean(records, fn) -> float:
    if not records:
        raise ValueError("no requests to score")
    return float(np.mean([fn(record) for record in records]))


def _slice(records, predicate, k: int) -> dict:
    chosen = [record for record in records if predicate(record)]
    if not chosen:
        return {"n": 0, "recall@20": None, "ndcg@20": None}
    return {
        "n": len(chosen),
        "recall@20": _mean(chosen, lambda record: recall_at_k(record["predicted"], record["relevant"], k)),
        "ndcg@20": _mean(chosen, lambda record: ndcg_at_k(record["predicted"], record["relevant"], k)),
    }
