"""Predict ΔNDCG@20 of a large beam versus a small beam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from budget_sid.constants import LARGE_BEAM, REFERENCE_BEAM, SMALL_BEAM
from budget_sid.errors import TestLeakError
from budget_sid.features import FEATURE_NAMES, stack_features
from budget_sid.splits import assert_test_unused, split_val_users


@dataclass
class BenefitPredictor:
    """Ridge regression on standardized first-step features.

    The target is the gain from spending a wider beam, not the model's
    confidence in a single recommendation.
    """

    coef: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} features, got {rows.shape[1]}")
        scaled = (rows - self.mean) / self.std
        design = np.concatenate([scaled, np.ones((scaled.shape[0], 1))], axis=1)
        return design @ self.coef


def fit_benefit_predictor(
    features: np.ndarray,
    delta_ndcg: np.ndarray,
    alpha: float = 1e-2,
) -> BenefitPredictor:
    rows = np.asarray(features, dtype=np.float64)
    target = np.asarray(delta_ndcg, dtype=np.float64).reshape(-1)
    if rows.ndim != 2 or rows.shape[0] != target.shape[0]:
        raise ValueError("features and delta_ndcg must describe the same requests")
    if rows.shape[0] < 2:
        raise ValueError("need at least two split-A requests to fit the predictor")
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = (rows - mean) / std
    design = np.concatenate([scaled, np.ones((scaled.shape[0], 1))], axis=1)
    penalty = alpha * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    coef = np.linalg.solve(gram, design.T @ target)
    return BenefitPredictor(coef=coef, mean=mean, std=std)


def n_large_for_budget(
    n_requests: int,
    small_beam: int = SMALL_BEAM,
    large_beam: int = LARGE_BEAM,
    reference_beam: int = REFERENCE_BEAM,
) -> int:
    """How many requests may take the large beam without exceeding the reference cost."""
    if not (small_beam < reference_beam < large_beam):
        raise ValueError("budget requires small < reference < large beam")
    if n_requests < 0:
        raise ValueError("n_requests must be non-negative")
    fraction = (reference_beam - small_beam) / (large_beam - small_beam)
    return int(np.floor(fraction * n_requests + 1e-12))


def select_threshold(
    scores: np.ndarray,
    small_beam: int = SMALL_BEAM,
    large_beam: int = LARGE_BEAM,
    reference_beam: int = REFERENCE_BEAM,
) -> float:
    """Highest threshold whose ``score >= threshold`` set stays inside the budget.

    Ties that would push the average beam above the reference are left on the
    small beam. The returned threshold is computed only from the scores passed
    in, which must already be validation split B.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot select a threshold without split-B scores")
    n_large = n_large_for_budget(values.size, small_beam, large_beam, reference_beam)
    order = np.argsort(-values, kind="mergesort")
    ranked = values[order]
    if n_large <= 0:
        return float(ranked[0] + 1.0)
    cutoff = float(ranked[n_large - 1])
    n_at_or_above = int(np.sum(values >= cutoff - 1e-12))
    if n_at_or_above > n_large:
        strictly_above = ranked[ranked > cutoff + 1e-12]
        if strictly_above.size == 0:
            return float(cutoff + 1.0)
        return float(strictly_above[-1])
    return cutoff


def mean_beam_width(scores: np.ndarray, threshold: float, small_beam: int, large_beam: int) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    widths = np.where(values >= threshold, large_beam, small_beam)
    return float(widths.mean())


@dataclass
class BenefitTrainingResult:
    predictor: BenefitPredictor
    threshold: float
    split_a: set[str]
    split_b: set[str]
    n_train_rows: int


def train_on_split_a(records, seed: int = 42) -> BenefitTrainingResult:
    """Fit on validation users A and choose the budget threshold on users B.

    ``records`` are mappings with ``user_id``, ``split``, ``features`` and
    ``delta_ndcg``. Any ``split == "test"`` row, or any test id overlapping A
    or B, aborts the run.
    """
    rows = list(records)
    test_ids = [str(row["user_id"]) for row in rows if row["split"] == "test"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if any(row["split"] == "test" for row in val_rows):
        raise TestLeakError("test rows were mixed into the validation training set")
    split_a, split_b = split_val_users((row["user_id"] for row in val_rows), seed=seed)
    assert_test_unused(split_a | split_b, test_ids)
    train_rows = [row for row in val_rows if str(row["user_id"]) in split_a]
    budget_rows = [row for row in val_rows if str(row["user_id"]) in split_b]
    if any(row["split"] != "val" for row in train_rows + budget_rows):
        raise TestLeakError("only validation users may train the predictor or set the threshold")
    predictor = fit_benefit_predictor(
        stack_features([row["features"] for row in train_rows]),
        np.array([row["delta_ndcg"] for row in train_rows], dtype=np.float64),
    )
    budget_scores = predictor.predict(stack_features([row["features"] for row in budget_rows]))
    threshold = select_threshold(budget_scores)
    return BenefitTrainingResult(
        predictor=predictor,
        threshold=threshold,
        split_a=split_a,
        split_b=split_b,
        n_train_rows=len(train_rows),
    )
