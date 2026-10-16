"""Validation-user split A/B. The test split never enters this helper's outputs."""

from __future__ import annotations

import numpy as np

from budget_sid.errors import TestLeakError


def split_val_users(
    user_ids,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split validation users into disjoint A and B.

    A trains the benefit regressor. B selects the routing threshold under the
    average-budget constraint. The split is a seeded shuffle of the unique ids,
    so it does not depend on row order.
    """
    unique = sorted({str(user_id) for user_id in user_ids})
    if len(unique) < 2:
        raise ValueError("need at least two validation users to form splits A and B")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_a = len(unique) // 2
    split_a = {unique[int(index)] for index in order[:n_a]}
    split_b = {unique[int(index)] for index in order[n_a:]}
    if split_a & split_b:
        raise RuntimeError("validation splits A and B overlap")
    if split_a | split_b != set(unique):
        raise RuntimeError("validation splits A and B dropped a user")
    return split_a, split_b


def assert_test_unused(used_user_ids, test_user_ids) -> None:
    leaked = {str(user_id) for user_id in used_user_ids} & {str(user_id) for user_id in test_user_ids}
    if leaked:
        sample = ", ".join(sorted(leaked)[:8])
        raise TestLeakError(f"test users leaked into training or threshold selection: {sample}")
