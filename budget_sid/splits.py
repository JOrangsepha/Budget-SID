"""Validation-user split A/B, plus the two user-overlap policies.

``assert_test_unused`` is the disjoint-user lock for the scripted benefit
smoke: each user belongs to one split, so a shared id is a leak.

Beauty is temporal leave-one-out. The same user id appears in training,
evaluation, and testing. That overlap is not label leakage. Label leakage is
the target interaction event sitting inside the history used to predict it.
Use ``assert_test_events_unused`` for that protocol.
"""

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
    """Raise when a user id used to fit or tune also appears in the test ids.

    This is the disjoint-user lock. Do not call it on Beauty leave-one-out
    ids: those users are supposed to occur in more than one split.
    """
    leaked = shared_user_ids(used_user_ids, test_user_ids)
    if leaked:
        sample = ", ".join(sorted(leaked)[:8])
        raise TestLeakError(f"test users leaked into training or threshold selection: {sample}")


USER_PROTOCOLS = ("disjoint_users", "temporal_loo")


def shared_user_ids(left, right) -> set[str]:
    return {str(user_id) for user_id in left} & {str(user_id) for user_id in right}


def assert_test_events_unused(used_event_ids, test_event_ids) -> None:
    """Temporal leave-one-out lock.

    The same user may have a validation event and a later test event.
    Using the test event itself to fit the benefit model or to pick the
    threshold is the leak this helper rejects.
    """
    leaked = {str(event_id) for event_id in used_event_ids} & {
        str(event_id) for event_id in test_event_ids
    }
    if leaked:
        sample = ", ".join(sorted(leaked)[:8])
        raise TestLeakError(f"test events leaked into training or threshold selection: {sample}")


def describe_split_users(rows) -> dict:
    """Count user ids shared across splits. Overlap is not label leakage.

    Rows need ``user_id`` and ``split`` (``train``, ``val``, or ``test``).
    This helper does not raise.
    """
    grouped: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        split = str(row["split"])
        if split not in grouped:
            continue
        grouped[split].add(str(row["user_id"]))
    return {
        "protocol": "temporal_leave_one_out",
        "user_overlap_is_label_leakage": False,
        "label_leakage_definition": (
            "the target interaction event appears in the history used to predict it"
        ),
        "disjoint_user_helper": "assert_test_unused",
        "disjoint_user_helper_applies_to": (
            "scripted benefit smoke, where each user belongs to one split"
        ),
        "n_users": {split: len(users) for split, users in grouped.items()},
        "n_shared_val_test": len(shared_user_ids(grouped["val"], grouped["test"])),
        "n_shared_train_test": len(shared_user_ids(grouped["train"], grouped["test"])),
        "n_shared_train_val": len(shared_user_ids(grouped["train"], grouped["val"])),
        "n_shared_all_three": len(grouped["train"] & grouped["val"] & grouped["test"]),
    }
