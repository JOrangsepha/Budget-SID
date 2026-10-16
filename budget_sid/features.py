"""Cheap first-step features for the benefit predictor."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "history_length",
    "first_step_entropy",
    "candidate_prob_gap",
    "top1_prob",
)


def as_float_array(values) -> np.ndarray:
    """Accept a NumPy array or any array-like stand-in (including test doubles)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("logits must be a vector of first-step scores")
    return array.reshape(-1)


def softmax(logits) -> np.ndarray:
    scores = as_float_array(logits)
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("softmax received non-finite logits")
    return weights / total


def entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def extract_features(history_length: int, logits) -> dict[str, float]:
    """Features available after the shared first decoding step.

    ``candidate_prob_gap`` is the probability difference between the best and
    second-best first-step tokens. The predictor uses it, together with
    history length and entropy, to estimate whether a wider beam is worth it.
    """
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    probs = softmax(logits)
    if probs.size < 2:
        raise ValueError("first-step distribution needs at least two candidates")
    order = np.argsort(-probs)
    top1 = float(probs[order[0]])
    gap = float(probs[order[0]] - probs[order[1]])
    features = {
        "history_length": float(history_length),
        "first_step_entropy": entropy(probs),
        "candidate_prob_gap": gap,
        "top1_prob": top1,
    }
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"feature extraction dropped {missing}")
    return features


def stack_features(rows: list[dict[str, float]]) -> np.ndarray:
    if not rows:
        raise ValueError("no feature rows to stack")
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
