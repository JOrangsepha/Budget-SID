"""Stop if a wider beam has no concentrated gain, or if an oracle budget loses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from budget_sid.benefit import n_large_for_budget
from budget_sid.constants import LARGE_BEAM, REFERENCE_BEAM, SMALL_BEAM
from budget_sid.errors import TestLeakError

EXIT_OK = 0
EXIT_STOP = 2


@dataclass
class StopGateResult:
    exit_code: int
    concentrated_gain: bool
    oracle_beats_reference: bool
    mean_delta_ndcg: float
    oracle_ndcg: float
    reference_ndcg: float
    positive_user_fraction: float
    top_mass_share: float
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK


def has_concentrated_gain(
    deltas,
    top_fraction: float = 0.2,
    min_positive_mass: float = 0.5,
    max_positive_user_fraction: float = 0.5,
) -> tuple[bool, dict]:
    """Return whether positive ΔNDCG@20 is uneven across requests.

    Uneven means either a minority of requests hold every positive Δ, or the
    highest-Δ slice (``top_fraction`` of all requests) holds at least
    ``min_positive_mass`` of the total positive gain.

    The mean over all requests is reported and is not a pass/fail check.
    Selective allocation can still help when that mean is near zero or
    negative, because the requests that lose on a wide beam stay on the
    small beam.
    """
    values = np.asarray(list(deltas), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return False, {"mean_delta": 0.0, "positive_user_fraction": 0.0, "top_mass_share": 0.0}
    mean_delta = float(values.mean())
    positive = values[values > 0.0]
    positive_fraction = float(positive.size / values.size)
    stats = {
        "mean_delta": mean_delta,
        "positive_user_fraction": positive_fraction,
        "top_mass_share": 0.0,
    }
    if positive.size == 0:
        return False, stats
    n_top = max(1, int(np.ceil(top_fraction * values.size)))
    ranked = np.sort(values)[::-1]
    top_slice = ranked[:n_top]
    top_gain = float(top_slice[top_slice > 0.0].sum())
    mass_share = float(top_gain / float(positive.sum()))
    stats["top_mass_share"] = mass_share
    uneven = positive_fraction <= max_positive_user_fraction or mass_share + 1e-12 >= min_positive_mass
    return bool(uneven), stats


def request_ndcg(row: dict, beam: int) -> float:
    """Read NDCG@20 for ``beam`` from a stop-gate row or a per-request JSONL row."""
    table = row.get("ndcg_by_beam")
    if isinstance(table, dict) and (beam in table or str(beam) in table):
        return float(table[beam] if beam in table else table[str(beam)])
    beams = row.get("beams") or {}
    cell = beams.get(str(beam), beams.get(beam))
    if isinstance(cell, dict) and cell.get("ndcg@20") is not None:
        return float(cell["ndcg@20"])
    request = row.get("request_id", row.get("user_id"))
    raise ValueError(f"request {request} is missing nDCG@20 for beam {beam}")


def budgeted_positive_assignment(
    records,
    small_beam: int = SMALL_BEAM,
    large_beam: int = LARGE_BEAM,
    reference_beam: int = REFERENCE_BEAM,
) -> dict:
    """Upgrade only positive-Δ requests, and only up to the average-beam cap.

    The cap is the largest count whose average beam width is still at or
    below ``reference_beam``. Requests with ΔNDCG@20 ≤ 0 stay on the small
    beam even when that leaves budget unused. This function does not look at
    ``split``; callers that must stay off the test set check that themselves.
    """
    rows = list(records)
    if not rows:
        raise ValueError("oracle allocation needs validation requests")
    deltas = np.array(
        [request_ndcg(row, large_beam) - request_ndcg(row, small_beam) for row in rows],
        dtype=np.float64,
    )
    user_ids = [str(row["user_id"]) for row in rows]
    n_budget = n_large_for_budget(len(rows), small_beam, large_beam, reference_beam)
    order = sorted(range(len(rows)), key=lambda index: (-float(deltas[index]), user_ids[index]))
    chosen = np.array([request_ndcg(row, small_beam) for row in rows], dtype=np.float64)
    upgraded = []
    for index in order:
        if len(upgraded) >= n_budget:
            break
        if not (float(deltas[index]) > 0.0):
            break
        chosen[index] = request_ndcg(rows[index], large_beam)
        upgraded.append(index)
    reference = np.array([request_ndcg(row, reference_beam) for row in rows], dtype=np.float64)
    small = np.array([request_ndcg(row, small_beam) for row in rows], dtype=np.float64)
    large = np.array([request_ndcg(row, large_beam) for row in rows], dtype=np.float64)
    oracle_ndcg = float(chosen.mean())
    reference_ndcg = float(reference.mean())
    n_upgraded = len(upgraded)
    mean_beam = float((n_upgraded * large_beam + (len(rows) - n_upgraded) * small_beam) / len(rows))
    return {
        "oracle_ndcg": oracle_ndcg,
        "reference_ndcg": reference_ndcg,
        "small_ndcg": float(small.mean()),
        "large_ndcg": float(large.mean()),
        "n_budget": int(n_budget),
        "n_upgraded": int(n_upgraded),
        "n_positive": int(np.sum(deltas > 0.0)),
        "mean_beam": mean_beam,
        "beats_reference": bool(oracle_ndcg > reference_ndcg),
        "upgraded_user_ids": [user_ids[index] for index in upgraded],
    }


def oracle_allocation(
    records,
    small_beam: int = SMALL_BEAM,
    large_beam: int = LARGE_BEAM,
    reference_beam: int = REFERENCE_BEAM,
):
    """Score the budgeted positive-Δ assignment against uniform ``reference_beam``.

    The assignment does not fill unused budget with negative-Δ requests.
    """
    rows = list(records)
    if not rows:
        raise ValueError("oracle allocation needs validation requests")
    if any(row.get("split") == "test" for row in rows):
        raise TestLeakError("the stop-gate oracle reads validation requests only")
    assigned = budgeted_positive_assignment(
        rows,
        small_beam=small_beam,
        large_beam=large_beam,
        reference_beam=reference_beam,
    )
    return assigned["oracle_ndcg"], assigned["reference_ndcg"], assigned["beats_reference"]


def run_stop_gate(records, top_fraction: float = 0.2, min_positive_mass: float = 0.5) -> StopGateResult:
    records = list(records)
    rows = [row for row in records if row.get("split", "val") == "val"]
    if len(rows) != len(records):
        raise TestLeakError("stop-gate input must be the validation split")
    deltas = [request_ndcg(row, LARGE_BEAM) - request_ndcg(row, SMALL_BEAM) for row in rows]
    concentrated, stats = has_concentrated_gain(
        deltas,
        top_fraction=top_fraction,
        min_positive_mass=min_positive_mass,
    )
    oracle_ndcg, reference_ndcg, beats = oracle_allocation(rows)
    reasons = []
    if not concentrated:
        reasons.append("no_concentrated_gain")
    if not beats:
        reasons.append("oracle_not_better_than_uniform_beam_40")
    return StopGateResult(
        exit_code=EXIT_OK if concentrated and beats else EXIT_STOP,
        concentrated_gain=concentrated,
        oracle_beats_reference=beats,
        mean_delta_ndcg=stats["mean_delta"],
        oracle_ndcg=oracle_ndcg,
        reference_ndcg=reference_ndcg,
        positive_user_fraction=stats["positive_user_fraction"],
        top_mass_share=stats["top_mass_share"],
        reasons=reasons,
    )
