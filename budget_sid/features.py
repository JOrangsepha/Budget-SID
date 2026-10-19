"""Cheap first-step features for the benefit predictor."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "history_length",
    "first_step_entropy",
    "candidate_prob_gap",
    "top1_prob",
)
# Ranks are 1-based after sorting first-step probabilities descending.
# Rank 20 is the last code a beam-20 first step keeps. Ranks 21–60 are the
# extra mass a beam-60 first step can still spend. These three columns are
# the boundary arm. They are not imputed when the logit vector is absent.
BOUNDARY_TOP_RANK = 20
BOUNDARY_TAIL_RANK = 60
BOUNDARY_FEATURE_NAMES: tuple[str, ...] = (
    "top20_cum_prob",
    "prob_gap_p20_p21",
    "prob_sum_ranks_21_60",
)
ROUTING_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + BOUNDARY_FEATURE_NAMES
# "Was the target item's hierarchy-0 code inside the top 20?" reads the label.
# It can be an offline attribution check. It is not a routing feature.


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


def ranked_probabilities(logits) -> np.ndarray:
    """First-step probabilities, highest rank first.

    Rank 1 is index 0. Ties keep the earlier logit index (stable mergesort).
    A vector shorter than rank 60 cannot name the beam-20/60 cut. It is not
    padded with zeros.
    """
    probs = softmax(logits)
    if probs.size < BOUNDARY_TAIL_RANK:
        raise ValueError(
            f"boundary features need at least {BOUNDARY_TAIL_RANK} first-step candidates "
            f"(ranks 1..{BOUNDARY_TAIL_RANK}), got {probs.size}. Refusing to pad with zeros."
        )
    order = np.argsort(-probs, kind="mergesort")
    return probs[order]


def extract_boundary_features(history_length: int, logits) -> dict[str, float]:
    """Base first-step features plus the beam-20/60 boundary cut.

    ``top20_cum_prob`` is the probability mass on ranks 1..20.
    ``prob_gap_p20_p21`` is p(rank 20) − p(rank 21).
    ``prob_sum_ranks_21_60`` is the mass on ranks 21..60, the codes a beam-60
    first step can still reach after beam 20 has taken its twenty.
    """
    base = extract_features(history_length, logits)
    ranked = ranked_probabilities(logits)
    top = ranked[:BOUNDARY_TOP_RANK]
    tail = ranked[BOUNDARY_TOP_RANK:BOUNDARY_TAIL_RANK]
    features = {
        **base,
        "top20_cum_prob": float(top.sum()),
        "prob_gap_p20_p21": float(ranked[BOUNDARY_TOP_RANK - 1] - ranked[BOUNDARY_TOP_RANK]),
        "prob_sum_ranks_21_60": float(tail.sum()),
    }
    missing = [name for name in ROUTING_FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"boundary feature extraction dropped {missing}")
    return features


def stack_features(rows: list[dict[str, float]]) -> np.ndarray:
    if not rows:
        raise ValueError("no feature rows to stack")
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)


def stack_named_features(rows: list[dict], names) -> np.ndarray:
    """Stack a subset of ``ROUTING_FEATURE_NAMES`` in the given order.

    P2 uses this when the joined file has only some of the first-step
    features. Missing logit columns are left out. They are not imputed.
    Boundary columns are included only when the caller lists them.
    """
    names = tuple(names)
    if not rows:
        raise ValueError("no feature rows to stack")
    if not names:
        raise ValueError("no feature names to stack")
    unknown = [name for name in names if name not in ROUTING_FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown feature names: {unknown}")
    matrix = []
    for row in rows:
        try:
            matrix.append([float(row[name]) for name in names])
        except KeyError as exc:
            raise ValueError(f"feature row is missing {exc}") from exc
    return np.array(matrix, dtype=np.float64)
