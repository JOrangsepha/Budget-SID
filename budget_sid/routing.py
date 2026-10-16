"""Choose a beam after one shared first step. Never rerun the search."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from budget_sid.constants import TOP_K
from budget_sid.errors import RoutingProtocolError
from budget_sid.features import extract_features


@dataclass
class RouteDecision:
    beam_width: int
    predicted_delta: float | None
    policy: str


@dataclass
class RouteResult:
    recommendations: list
    beam_width: int
    policy: str
    predicted_delta: float | None
    features: dict
    latency_s: float
    latency_breakdown: dict
    first_step_calls: int
    continuation_calls: int


def run_small_then_rerun_large(*_args, **_kwargs):
    """Retained so call sites cannot quietly implement the forbidden schedule."""
    raise RoutingProtocolError(
        "Do not finish the small beam and then rerun the large beam. "
        "Decode the first step once, then continue at the chosen width."
    )


def filter_legal(candidates, catalog, top_k: int = TOP_K) -> list:
    allowed = {_item_id(item) for item in catalog}
    chosen = []
    seen = set()
    for item in candidates:
        item_id = _item_id(item)
        if item_id in allowed and item_id not in seen:
            chosen.append(item_id)
            seen.add(item_id)
        if len(chosen) >= top_k:
            break
    return chosen


def _item_id(item):
    if isinstance(item, bool):
        return item
    if isinstance(item, (int, np.integer)):
        return int(item)
    return item


def route_request(recommender, history, policy, catalog, top_k: int = TOP_K) -> RouteResult:
    """Shared first step, then one continuation at the width the policy picks.

    Latency includes the first step, the policy, the legal-id filter, and the
    continuation. A second full decode is not performed.
    """
    first_calls_before = len(getattr(recommender, "first_calls", []))
    continue_calls_before = len(getattr(recommender, "continue_calls", []))

    started = time.perf_counter()
    state = recommender.first_step(history)
    after_first = time.perf_counter()

    features = extract_features(_history_length(history), state.logits)
    decision = policy.decide(features)
    after_policy = time.perf_counter()

    recommendations = recommender.continue_from_first_step(state, decision.beam_width)
    after_continue = time.perf_counter()
    legal = filter_legal(recommendations, catalog, top_k=top_k)
    finished = time.perf_counter()

    first_s = max(after_first - started, float(getattr(state, "elapsed_s", 0.0)))
    continuation_s = max(
        after_continue - after_policy,
        float(getattr(recommender, "last_continuation_s", 0.0)),
    )
    policy_s = after_policy - after_first
    filter_s = finished - after_continue
    breakdown = {
        "first_step": first_s,
        "route_decision": policy_s,
        "continuation": continuation_s,
        "legal_filter": filter_s,
    }
    first_calls = len(getattr(recommender, "first_calls", [])) - first_calls_before
    continuation_calls = len(getattr(recommender, "continue_calls", [])) - continue_calls_before
    if first_calls > 1 or continuation_calls > 1:
        raise RoutingProtocolError("routing decoded more than one first step or continuation")
    return RouteResult(
        recommendations=legal,
        beam_width=int(decision.beam_width),
        policy=decision.policy,
        predicted_delta=decision.predicted_delta,
        features=features,
        latency_s=sum(breakdown.values()),
        latency_breakdown=breakdown,
        first_step_calls=first_calls if hasattr(recommender, "first_calls") else 1,
        continuation_calls=continuation_calls if hasattr(recommender, "continue_calls") else 1,
    )


def _history_length(history) -> int:
    if hasattr(history, "items"):
        return len(tuple(history.items))
    sequence = tuple(history)
    if sequence and isinstance(sequence[0], str) and sequence[0].startswith("user:"):
        return len(sequence) - 1
    return len(sequence)
