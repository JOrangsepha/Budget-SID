"""Checks that a request's history does not already contain its target."""

from __future__ import annotations

from budget_sid.errors import HistoryLeakError


def assert_history_excludes_target(history, target) -> None:
    """Item-id check used by the scripted smoke path.

    A repeated purchase of the same item fails this helper. Beauty leave-one-out
    allows that. Use ``assert_target_event_not_in_history`` for the event check.
    """
    if target in history:
        raise HistoryLeakError(
            "history contains the interaction being predicted; "
            "drop the target before training or evaluation"
        )


def assert_target_event_not_in_history(history_positions, target_position: int) -> None:
    """The held-out interaction event must not be one of the history events.

    ``history_positions`` are indexes of events that were shown to the model.
    ``target_position`` is the index of the event being predicted. The same item
    id at an earlier index is a repeated purchase, not leakage.
    """
    position = int(target_position)
    if position < 0:
        raise HistoryLeakError(f"target event position must be >= 0, got {position}")
    positions = {int(index) for index in history_positions}
    if position in positions:
        raise HistoryLeakError(
            "history contains the interaction being predicted; "
            f"target event at position {position} is in history"
        )


def audit_requests(requests) -> int:
    """Return the number of requests that passed the history audit."""
    count = 0
    for request in requests:
        assert_history_excludes_target(request["history"], request["target"])
        count += 1
    return count
