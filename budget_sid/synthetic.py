"""Synthetic Beauty-sized requests for CPU smoke and the stop-gate tests.

Most validation users gain nothing from a wider beam. A small group is missed
by beams 20 and 40 and found by beam 60, which is the concentrated-gain case
the stop-gate is willing to continue past.
"""

from __future__ import annotations

import numpy as np

from budget_sid.features import extract_features

CATALOG = list(range(40))


def _logits(peaked: bool, width: int = 16) -> np.ndarray:
    if peaked:
        logits = np.zeros(width, dtype=np.float64)
        logits[0] = 8.0
        return logits
    return np.zeros(width, dtype=np.float64)


def _ranking(relevant: int, hit: bool) -> list[int]:
    rest = [item for item in CATALOG if item != relevant]
    if hit:
        return [relevant, *rest]
    return [*rest, relevant]


def make_population(n_easy: int = 20, n_hard: int = 4, n_test: int = 8):
    """Validation rows the stop-gate can pass, plus a locked test split."""
    rows = []
    for index in range(n_easy + n_hard):
        hard = index < n_hard
        user_id = f"val-{index:02d}"
        relevant = index + 1
        peaked = not hard
        logits = _logits(peaked)
        history_items = [] if hard else [relevant + 100, relevant + 101, relevant + 102, relevant + 103]
        history = ("user:" + user_id, *history_items)
        ndcg = {20: 0.0, 40: 0.0, 60: 1.0} if hard else {20: 1.0, 40: 1.0, 60: 1.0}
        rows.append(
            _row(
                user_id=user_id,
                split="val",
                history=history,
                relevant=relevant,
                logits=logits,
                ndcg=ndcg,
                long_tail=hard,
            )
        )
    for index in range(n_test):
        user_id = f"test-{index:02d}"
        relevant = 30 + index
        logits = _logits(peaked=index % 2 == 0)
        history = ("user:" + user_id, 7, 8, 9) if index % 2 == 0 else ("user:" + user_id,)
        ndcg = {20: 1.0, 40: 1.0, 60: 1.0} if index % 2 == 0 else {20: 0.0, 40: 0.0, 60: 1.0}
        rows.append(
            _row(
                user_id=user_id,
                split="test",
                history=history,
                relevant=relevant,
                logits=logits,
                ndcg=ndcg,
                long_tail=index % 2 == 1,
            )
        )
    return rows


def uniform_gain_population(n_users: int = 24, delta: float = 0.1):
    """Every validation user improves by the same amount. Gain is not concentrated."""
    rows = []
    for index in range(n_users):
        user_id = f"flat-{index:02d}"
        relevant = 1
        rows.append(
            _row(
                user_id=user_id,
                split="val",
                history=("user:" + user_id, 2, 3, 4, 5, 6),
                relevant=relevant,
                logits=_logits(peaked=True),
                # Same Δ on every user, while beam 40 is still far below beam 60.
                # The oracle would help, but the gain is not concentrated.
                ndcg={20: 0.0, 40: float(delta), 60: 1.0},
                long_tail=False,
            )
        )
    return rows


def concentrated_tail_population(n_positive: int = 16, n_decoy: int = 24, n_negative: int = 24, n_zero: int = 24, n_test: int = 4):
    """Validation requests whose positive Δ sits on a minority of users.

    Positives are a short-history, moderate-entropy cluster. High-entropy
    decoys and negatives have no positive Δ, so an entropy ranking spends the
    quota on them. A ridge fit on the four first-step features can separate
    the positive cluster. Uniform beam 60 loses to uniform beam 40 because
    the negative cluster outweighs the tail. Test users are disjoint from
    validation and must stay out of the fit.
    """
    classes = (
        ("positive", n_positive, _TAIL_POSITIVE),
        ("decoy", n_decoy, _TAIL_DECOY),
        ("negative", n_negative, _TAIL_NEGATIVE),
        ("zero", n_zero, _TAIL_ZERO),
    )
    rows = []
    for kind, count, spec in classes:
        for index in range(count):
            rows.append(_tail_row(f"{kind}-{index:02d}", "val", spec))
    for index in range(n_test):
        rows.append(_tail_row(f"test-{index:02d}", "test", _TAIL_POSITIVE))
    return rows


_TAIL_POSITIVE = {
    "history_length": 1.0,
    "first_step_entropy": 0.40,
    "candidate_prob_gap": 0.10,
    "top1_prob": 0.20,
    "ndcg": {20: 0.0, 40: 0.05, 60: 0.80},
}
_TAIL_DECOY = {
    "history_length": 20.0,
    "first_step_entropy": 2.20,
    "candidate_prob_gap": 0.80,
    "top1_prob": 0.15,
    "ndcg": {20: 0.15, 40: 0.15, 60: 0.15},
}
_TAIL_NEGATIVE = {
    "history_length": 10.0,
    "first_step_entropy": 1.80,
    "candidate_prob_gap": 0.50,
    "top1_prob": 0.40,
    "ndcg": {20: 0.60, 40: 0.70, 60: 0.10},
}
_TAIL_ZERO = {
    "history_length": 8.0,
    "first_step_entropy": 0.02,
    "candidate_prob_gap": 0.95,
    "top1_prob": 0.97,
    "ndcg": {20: 0.30, 40: 0.30, 60: 0.30},
}


def _tail_row(user_id: str, split: str, spec: dict) -> dict:
    length = int(spec["history_length"])
    history = list(range(1, length + 1))
    ndcg = spec["ndcg"]
    features = {
        "history_length": float(spec["history_length"]),
        "first_step_entropy": float(spec["first_step_entropy"]),
        "candidate_prob_gap": float(spec["candidate_prob_gap"]),
        "top1_prob": float(spec["top1_prob"]),
    }
    beams = {}
    for beam, value in ndcg.items():
        beams[str(beam)] = {"ndcg@20": float(value)}
    return {
        "request_id": f"{split}:{user_id}:{length}",
        "user_id": user_id,
        "split": split,
        "history": history,
        "target_position": length,
        "features": features,
        "beams": beams,
        "delta_ndcg@20": float(ndcg[60] - ndcg[20]),
    }


def selective_oracle_counterexample():
    """Ten validation requests: two gain from beam 60, eight are hurt by it.

    Mean ΔNDCG@20 is −0.6. Filling the beam-40 budget (five large beams)
    upgrades hurt requests and scores 0.7, which loses to uniform beam 40
    (0.8). Upgrading only the two positive requests scores 1.0 at mean beam
    28, which is inside the budget and beats both.
    """
    rows = []
    for index in range(2):
        rows.append(_beam_row(f"gain-{index:02d}", {20: 0.0, 40: 0.0, 60: 1.0}))
    for index in range(8):
        rows.append(_beam_row(f"hurt-{index:02d}", {20: 1.0, 40: 1.0, 60: 0.0}))
    return rows


def _beam_row(user_id: str, ndcg: dict):
    return {
        "user_id": user_id,
        "split": "val",
        "ndcg_by_beam": {int(beam): float(value) for beam, value in ndcg.items()},
        "delta_ndcg": float(ndcg[60] - ndcg[20]),
    }


def oracle_tied_population(n_easy: int = 20, n_hard: int = 4):
    """Wide-beam gain exists, but beam 40 already captured it, so the oracle ties."""
    rows = []
    for index in range(n_easy + n_hard):
        hard = index < n_hard
        user_id = f"tie-{index:02d}"
        relevant = index + 1
        ndcg = {20: 0.0, 40: 1.0, 60: 1.0} if hard else {20: 1.0, 40: 1.0, 60: 1.0}
        rows.append(
            _row(
                user_id=user_id,
                split="val",
                history=("user:" + user_id, 3, 4),
                relevant=relevant,
                logits=_logits(peaked=not hard),
                ndcg=ndcg,
                long_tail=hard,
            )
        )
    return rows


def _row(user_id, split, history, relevant, logits, ndcg, long_tail):
    item_history = [item for item in history if not (isinstance(item, str) and item.startswith("user:"))]
    features = extract_features(len(item_history), logits)
    return {
        "user_id": user_id,
        "split": split,
        "history": tuple(history),
        "target": relevant,
        "relevant": [relevant],
        "logits": logits,
        "features": features,
        "delta_ndcg": float(ndcg[60] - ndcg[20]),
        "ndcg_by_beam": dict(ndcg),
        "rankings": {
            20: _ranking(relevant, hit=ndcg[20] >= 1.0),
            40: _ranking(relevant, hit=ndcg[40] >= 1.0),
            60: _ranking(relevant, hit=ndcg[60] >= 1.0),
        },
        "history_length": len(item_history),
        "long_tail": long_tail,
    }
