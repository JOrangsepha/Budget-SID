import numpy as np
import pytest

from budget_sid.benefit import (
    fit_benefit_predictor,
    mean_beam_width,
    select_threshold,
    train_on_split_a,
)
from budget_sid.data_audit import assert_history_excludes_target, audit_requests
from budget_sid.errors import HistoryLeakError, TestLeakError
from budget_sid.features import extract_features, stack_features
from budget_sid.splits import split_val_users
from budget_sid.synthetic import make_population


class MockTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def test_first_step_entropy_and_gap():
    uniform = extract_features(2, np.zeros(4))
    peaked = extract_features(8, np.array([12.0, 0.0, 0.0, 0.0]))
    assert uniform["first_step_entropy"] == pytest.approx(np.log(4))
    assert uniform["candidate_prob_gap"] == pytest.approx(0.0)
    assert peaked["first_step_entropy"] < 0.05
    assert peaked["candidate_prob_gap"] > 0.9
    assert uniform["first_step_entropy"] > peaked["first_step_entropy"]


def test_features_accept_mock_tensor():
    features = extract_features(4, MockTensor([5.0, 0.0, 0.0, 0.0]))
    assert features["history_length"] == 4.0
    assert features["candidate_prob_gap"] > 0.5
    assert features["top1_prob"] > features["candidate_prob_gap"]


def test_val_user_split_is_disjoint_and_complete():
    users = [f"u{index}" for index in range(10)]
    split_a, split_b = split_val_users(users, seed=1)
    assert split_a.isdisjoint(split_b)
    assert split_a | split_b == set(users)
    assert len(split_a) == 5
    again_a, again_b = split_val_users(list(reversed(users)), seed=1)
    assert split_a == again_a
    assert split_b == again_b


def test_benefit_predictor_fits_split_a_signal():
    rng = np.random.default_rng(0)
    rows = []
    for index in range(40):
        entropy = float(rng.uniform(0.0, 2.0))
        gap = float(rng.uniform(0.0, 1.0))
        features = {
            "history_length": float(rng.integers(1, 30)),
            "first_step_entropy": entropy,
            "candidate_prob_gap": gap,
            "top1_prob": float(np.clip(0.9 - 0.5 * gap, 0.05, 0.99)),
        }
        rows.append(
            {
                "user_id": f"u{index}",
                "split": "val",
                "features": features,
                "delta_ndcg": 0.7 * entropy - 0.4 * gap,
            }
        )
    rows.append(
        {
            "user_id": "test-heldout",
            "split": "test",
            "features": rows[0]["features"],
            "delta_ndcg": 1.0,
        }
    )
    trained = train_on_split_a(rows, seed=0)
    assert "test-heldout" not in trained.split_a
    assert "test-heldout" not in trained.split_b
    train_rows = [row for row in rows if row["user_id"] in trained.split_a]
    predicted = trained.predictor.predict(stack_features([row["features"] for row in train_rows]))
    target = np.array([row["delta_ndcg"] for row in train_rows])
    correlation = float(np.corrcoef(predicted, target)[0, 1])
    assert correlation > 0.9


def test_test_split_cannot_train_or_set_threshold():
    rows = make_population()
    trained = train_on_split_a(rows, seed=42)
    test_ids = {row["user_id"] for row in rows if row["split"] == "test"}
    assert trained.split_a.isdisjoint(test_ids)
    assert trained.split_b.isdisjoint(test_ids)
    leaked = list(rows)
    leaked.append({**rows[0], "split": "test"})
    with pytest.raises(TestLeakError):
        train_on_split_a(leaked, seed=42)


def test_threshold_meets_average_budget():
    scores = np.arange(10, dtype=float)
    threshold = select_threshold(scores)
    assert mean_beam_width(scores, threshold, small_beam=20, large_beam=60) == pytest.approx(40.0)
    assert int(np.sum(scores >= threshold)) == 5


def test_threshold_refuses_ties_that_break_the_budget():
    scores = np.ones(10)
    threshold = select_threshold(scores)
    assert mean_beam_width(scores, threshold, small_beam=20, large_beam=60) <= 40.0
    assert int(np.sum(scores >= threshold)) == 0


def test_history_must_exclude_target():
    with pytest.raises(HistoryLeakError):
        assert_history_excludes_target([1, 2, 3], 2)
    assert audit_requests([{"history": [1, 4], "target": 2}]) == 1


def test_direct_fit_uses_delta_not_confidence():
    features = np.array(
        [
            [1.0, 0.1, 0.9, 0.95],
            [1.0, 1.5, 0.1, 0.3],
            [8.0, 0.1, 0.8, 0.9],
            [8.0, 1.6, 0.05, 0.25],
        ]
    )
    # High confidence (large gap) is paired with zero gain. High entropy is the gain.
    target = np.array([0.0, 1.0, 0.0, 1.0])
    predictor = fit_benefit_predictor(features, target, alpha=1e-3)
    scored = predictor.predict(features)
    assert scored[1] > scored[0]
    assert scored[3] > scored[2]
