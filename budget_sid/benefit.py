"""Predict ΔNDCG@20 of a large beam versus a small beam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from budget_sid.constants import LARGE_BEAM, REFERENCE_BEAM, SMALL_BEAM
from budget_sid.errors import TestLeakError
from budget_sid.features import FEATURE_NAMES, stack_features
from budget_sid.splits import USER_PROTOCOLS, assert_test_events_unused, assert_test_unused, split_val_users


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


# Depth and iteration caps for the optional tree model. It fits whatever
# columns the caller passes, the same list as ridge. Boundary columns are
# included only when that list contains them.
HGB_MAX_DEPTH = 3
HGB_MAX_ITER = 50


def try_fit_hist_gradient_boosting(
    features: np.ndarray,
    delta_ndcg: np.ndarray,
    *,
    max_depth: int = HGB_MAX_DEPTH,
    max_iter: int = HGB_MAX_ITER,
    random_state: int = 0,
):
    """Fit HistGradientBoostingRegressor on the same columns as ridge.

    Returns ``(model, info)``. ``model`` is None when scikit-learn is not
    installed or the fit fails. Callers keep the ridge predictor either way.
    ``max_depth`` and ``max_iter`` stay at the caps above unless a test asks
    otherwise. Boundary columns are included when the caller puts them in
    ``features``.
    """
    limit = {
        "model": "HistGradientBoostingRegressor",
        "max_depth": int(max_depth),
        "max_iter": int(max_iter),
        "early_stopping": False,
        "same_features_as_ridge": True,
        "boundary_features": False,
    }
    if int(max_depth) < 1 or int(max_iter) < 1:
        raise ValueError("max_depth and max_iter must be positive")
    if int(max_depth) > HGB_MAX_DEPTH or int(max_iter) > HGB_MAX_ITER:
        raise ValueError(
            f"HistGradientBoostingRegressor is capped at max_depth={HGB_MAX_DEPTH} "
            f"and max_iter={HGB_MAX_ITER}"
        )
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        return None, {
            **limit,
            "status": "unavailable",
            "split_b_mse": None,
            "reason": "scikit-learn is not installed; HistGradientBoostingRegressor was not fit",
            "error": type(exc).__name__,
        }
    rows = np.asarray(features, dtype=np.float64)
    target = np.asarray(delta_ndcg, dtype=np.float64).reshape(-1)
    if rows.ndim != 2 or rows.shape[0] != target.shape[0]:
        raise ValueError("features and delta_ndcg must describe the same requests")
    if rows.shape[0] < 2:
        raise ValueError("need at least two split-A requests to fit the predictor")
    try:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=int(max_depth),
            max_iter=int(max_iter),
            early_stopping=False,
            learning_rate=0.1,
            min_samples_leaf=1,
            random_state=int(random_state),
        )
        model.fit(rows, target)
    except (TypeError, ValueError, RuntimeError) as exc:
        return None, {
            **limit,
            "status": "unavailable",
            "split_b_mse": None,
            "reason": "HistGradientBoostingRegressor fit failed: " + str(exc),
            "error": type(exc).__name__,
        }
    return model, {
        **limit,
        "status": "fit",
        "random_state": int(random_state),
        "n_train_rows": int(rows.shape[0]),
        "split_b_mse": None,
        "reason": (
            "Same joined columns as ridge, with max_depth and max_iter capped. "
            "Boundary columns are included when the joined feature file has them."
        ),
    }


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


def train_on_split_a(records, seed: int = 42, *, user_protocol: str = "disjoint_users") -> BenefitTrainingResult:
    """Fit on validation users A and choose the budget threshold on users B.

    ``records`` are mappings with ``user_id``, ``split``, ``features`` and
    ``delta_ndcg``. Test rows are never fit. ``user_protocol="disjoint_users"``
    (the scripted smoke) also rejects a test user id inside A or B.
    ``user_protocol="temporal_loo"`` allows that shared id and instead rejects
    a reused test ``request_id``. Beauty leave-one-out needs the latter.
    """
    if user_protocol not in USER_PROTOCOLS:
        raise ValueError(f"unknown user_protocol {user_protocol!r}; expected one of {USER_PROTOCOLS}")
    rows = list(records)
    test_ids = [str(row["user_id"]) for row in rows if row["split"] == "test"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if any(row["split"] == "test" for row in val_rows):
        raise TestLeakError("test rows were mixed into the validation training set")
    split_a, split_b = split_val_users((row["user_id"] for row in val_rows), seed=seed)
    if user_protocol == "disjoint_users":
        assert_test_unused(split_a | split_b, test_ids)
    else:
        relevant = [row for row in rows if row["split"] in {"val", "test"}]
        missing = [row for row in relevant if not row.get("request_id")]
        if missing:
            raise ValueError("temporal_loo requires request_id on every val and test row")
        test_events = [str(row["request_id"]) for row in relevant if row["split"] == "test"]
        used_events = [str(row["request_id"]) for row in val_rows]
        assert_test_events_unused(used_events, test_events)
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
