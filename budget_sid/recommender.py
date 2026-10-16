"""Scripted CPU recommender used by ``--smoke`` and the unit tests.

The scripted model never downloads Amazon reviews or a text encoder. A later
GPU run swaps in the lightweight transformer described by the Beauty config
without changing the first-step routing protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from budget_sid.constants import TOP_K


@dataclass
class FirstStepState:
    history: tuple
    logits: np.ndarray
    elapsed_s: float = 0.0
    continued: bool = False


@dataclass
class ScriptedRecommender:
    """Deterministic stand-in with an explicit shared-first-step API."""

    catalog: Sequence[int]
    seed: int = 0
    first_step_s: float = 0.001
    first_calls: list = field(default_factory=list)
    continue_calls: list = field(default_factory=list)
    full_decode_calls: int = 0
    last_continuation_s: float = 0.0

    def first_step(self, history: Sequence) -> FirstStepState:
        frozen = tuple(history)
        self.first_calls.append(frozen)
        rng = np.random.default_rng(self._digest(frozen + ("seed", self.seed)))
        logits = rng.normal(size=16)
        item_count = len(frozen) - 1 if frozen and isinstance(frozen[0], str) else len(frozen)
        if item_count <= 3:
            logits = logits * 0.05
        return FirstStepState(history=frozen, logits=logits, elapsed_s=self.first_step_s)

    def continue_from_first_step(self, state: FirstStepState, beam_width: int) -> list[int]:
        if state.continued:
            raise RuntimeError("first-step state was continued twice")
        state.continued = True
        self.continue_calls.append(int(beam_width))
        self.last_continuation_s = self.first_step_s * (int(beam_width) / 20.0)
        return self._ranking(state.history, int(beam_width))

    def decode_full(self, history: Sequence, beam_width: int) -> list[int]:
        """The forbidden path: a fresh decode that ignores the shared first step."""
        self.full_decode_calls += 1
        self.first_step(history)
        return self._ranking(tuple(history), int(beam_width))

    def _ranking(self, history: tuple, beam_width: int) -> list[int]:
        catalog = [int(item) for item in self.catalog]
        preference = np.random.default_rng(self._digest(history + ("seed", self.seed))).permutation(len(catalog))
        # Wider beams stay closer to a stable preference order. Top-20 reads
        # the prefix, so a larger beam is a different ranking, not a longer list.
        if beam_width >= 60:
            chosen = preference
        elif beam_width >= 40:
            shift = max(1, len(catalog) // 8)
            chosen = np.concatenate([preference[shift:], preference[:shift]])
        else:
            shift = max(1, len(catalog) // 3)
            chosen = np.concatenate([preference[shift:], preference[:shift]])
        return [int(catalog[int(index)]) for index in chosen[: max(TOP_K, beam_width)]]

    @staticmethod
    def _digest(history: tuple) -> int:
        payload = repr(tuple(history)).encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:8], 16)


@dataclass
class LookupRecommender:
    """Replay precomputed first-step logits and per-beam rankings."""

    rows: Sequence[dict]
    first_calls: list = field(default_factory=list)
    continue_calls: list = field(default_factory=list)
    full_decode_calls: int = 0
    last_continuation_s: float = 0.0

    def __post_init__(self) -> None:
        self._by_user = {row["user_id"]: row for row in self.rows}

    def first_step(self, history: Sequence) -> FirstStepState:
        user_id = _user_id(history)
        row = self._by_user[user_id]
        self.first_calls.append(user_id)
        return FirstStepState(history=tuple(history), logits=np.asarray(row["logits"]), elapsed_s=0.002)

    def continue_from_first_step(self, state: FirstStepState, beam_width: int) -> list[int]:
        if state.continued:
            raise RuntimeError("first-step state was continued twice")
        state.continued = True
        user_id = _user_id(state.history)
        self.continue_calls.append(int(beam_width))
        self.last_continuation_s = 0.002 * (int(beam_width) / 20.0)
        return list(self._by_user[user_id]["rankings"][int(beam_width)])

    def decode_full(self, history: Sequence, beam_width: int) -> list[int]:
        self.full_decode_calls += 1
        state = self.first_step(history)
        return self.continue_from_first_step(state, beam_width)


def _user_id(history: Sequence) -> str:
    head = tuple(history)[0]
    if isinstance(head, str) and head.startswith("user:"):
        return head.split("user:", 1)[1]
    return str(head)
