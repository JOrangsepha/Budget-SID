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
