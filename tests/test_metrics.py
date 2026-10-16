import math

import numpy as np
import pytest

from budget_sid.metrics import (
    aggregate_seed_metrics,
    latency_mean,
    latency_p95,
    long_tail_items,
    ndcg_at_k,
    peak_vram_mb,
    recall_at_k,
    score_retrieval,
    summarize_requests,
    throughput,
)


def test_recall_and_ndcg_at_20():
    predicted = [1, 4, 2, *range(10, 30)]
    relevant = [1, 2, 3]
    assert recall_at_k(predicted, relevant, k=20) == pytest.approx(2 / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(predicted, relevant, k=20) == pytest.approx(dcg / ideal)
    assert recall_at_k([9], [], k=20) == 0.0


def test_latency_throughput_vram():
    assert latency_mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)
    assert latency_p95([1, 2, 3, 4, 5]) == pytest.approx(4.8)
    assert throughput(8, 2.0) == pytest.approx(4.0)
    assert peak_vram_mb([10.0, 40.0, 25.0]) == pytest.approx(40.0)
    assert peak_vram_mb() == pytest.approx(0.0)


def test_long_tail_and_short_history_slices():
    frequencies = {1: 1, 2: 1, 3: 100, 4: 100}
    tail = long_tail_items(frequencies, quantile=0.2)
    assert 1 in tail
    assert 3 not in tail
    records = [
        {"predicted": [1], "relevant": [1], "latency_s": 0.1, "history_length": 2, "long_tail": True},
        {"predicted": [9], "relevant": [2], "latency_s": 0.2, "history_length": 2, "long_tail": True},
        {"predicted": [3], "relevant": [3], "latency_s": 0.3, "history_length": 12, "long_tail": False},
        {"predicted": [8], "relevant": [4], "latency_s": 0.4, "history_length": 12, "long_tail": False},
    ]
    summary = summarize_requests(records, short_history_max=5)
    assert summary["slices"]["long_tail"]["n"] == 2
    assert summary["slices"]["short_history"]["n"] == 2
    assert summary["slices"]["long_tail"]["recall@20"] == pytest.approx(0.5)
    assert summary["n"] == 4
    assert summary["peak_vram_mb"] == pytest.approx(0.0)


def test_full_catalog_rejects_sampled_negatives():
    with pytest.raises(ValueError, match="sampled negatives"):
        score_retrieval([1], [1], sampled_negatives=True)
    with pytest.raises(ValueError, match="sampled negatives"):
        summarize_requests(
            [
                {
                    "predicted": [1],
                    "relevant": [1],
                    "latency_s": 0.1,
                    "history_length": 1,
                    "long_tail": False,
                    "sampled_negatives": True,
                }
            ]
        )


def test_three_seed_aggregate():
    aggregated = aggregate_seed_metrics(
        [{"ndcg@20": 0.1, "recall@20": 0.2}, {"ndcg@20": 0.2, "recall@20": 0.4}, {"ndcg@20": 0.3, "recall@20": 0.6}]
    )
    assert aggregated["ndcg@20"]["mean"] == pytest.approx(0.2)
    assert aggregated["ndcg@20"]["std"] == pytest.approx(float(np.std([0.1, 0.2, 0.3], ddof=1)))
    assert aggregated["recall@20"]["mean"] == pytest.approx(0.4)
