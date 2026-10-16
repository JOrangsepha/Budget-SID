"""Checks that a request's history does not already contain its target."""

from __future__ import annotations

from budget_sid.errors import HistoryLeakError


def assert_history_excludes_target(history, target) -> None:
    if target in history:
        raise HistoryLeakError(
            "history contains the interaction being predicted; "
            "drop the target before training or evaluation"
        )


def audit_requests(requests) -> int:
    """Return the number of requests that passed the history audit."""
    count = 0
    for request in requests:
        assert_history_excludes_target(request["history"], request["target"])
        count += 1
    return count
