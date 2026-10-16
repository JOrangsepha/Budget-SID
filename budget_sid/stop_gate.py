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
    min_mean_delta: float = 0.0,
) -> tuple[bool, dict]:
    """Gain is concentrated when it is real and uneven across requests.

    Uneven means either a minority of requests are the only ones with positive
    ΔNDCG@20, or the highest-Δ slice (``top_fraction``) holds at least
    ``min_positive_mass`` of the total positive gain.
    """
    values = np.asarray(list(deltas), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return False, {"mean_delta": 0.0, "positive_user_fraction": 0.0, "top_mass_share": 0.0}
    mean_delta = float(values.mean())
    positive = values[values > 0]
    positive_fraction = float(positive.size / values.size)
    if positive.size == 0 or mean_delta <= min_mean_delta:
        return False, {
            "mean_delta": mean_delta,
            "positive_user_fraction": positive_fraction,
            "top_mass_share": 0.0,
        }
    n_top = max(1, int(np.ceil(top_fraction * values.size)))
    ranked = np.sort(values)[::-1]
    top_positive = ranked[:n_top]
    top_positive = top_positive[top_positive > 0].sum()
    mass_share = float(top_positive / positive.sum())
    uneven = positive_fraction <= max_positive_user_fraction or mass_share + 1e-12 >= min_positive_mass
    return uneven, {
        "mean_delta": mean_delta,
        "positive_user_fraction": positive_fraction,
        "top_mass_share": mass_share,
    }


def oracle_allocation(records, small_beam: int = SMALL_BEAM, large_beam: int = LARGE_BEAM, reference_beam: int = REFERENCE_BEAM):
    """Give the large beam to the requests with the largest true ΔNDCG@20.

    The number of large-beam requests is the largest count whose average beam
    width stays at or below ``reference_beam``. Compared with giving every
    request beam ``reference_beam``.
    """
    rows = list(records)
    if not rows:
        raise ValueError("oracle allocation needs validation requests")
    if any(row.get("split") == "test" for row in rows):
        raise TestLeakError("the stop-gate oracle reads validation requests only")
    deltas = np.array(
        [float(row["ndcg_by_beam"][large_beam]) - float(row["ndcg_by_beam"][small_beam]) for row in rows],
        dtype=np.float64,
    )
    user_ids = [str(row["user_id"]) for row in rows]
    n_large = n_large_for_budget(len(rows), small_beam, large_beam, reference_beam)
    order = sorted(range(len(rows)), key=lambda index: (-deltas[index], user_ids[index]))
    chosen = np.array([float(row["ndcg_by_beam"][small_beam]) for row in rows], dtype=np.float64)
    for index in order[:n_large]:
        chosen[index] = float(rows[index]["ndcg_by_beam"][large_beam])
    reference = np.array([float(row["ndcg_by_beam"][reference_beam]) for row in rows], dtype=np.float64)
    oracle_ndcg = float(chosen.mean())
    reference_ndcg = float(reference.mean())
    return oracle_ndcg, reference_ndcg, oracle_ndcg > reference_ndcg


def run_stop_gate(records, top_fraction: float = 0.2, min_positive_mass: float = 0.5) -> StopGateResult:
    records = list(records)
    rows = [row for row in records if row.get("split", "val") == "val"]
    if len(rows) != len(records):
        raise TestLeakError("stop-gate input must be the validation split")
    deltas = [
        float(row["ndcg_by_beam"][LARGE_BEAM]) - float(row["ndcg_by_beam"][SMALL_BEAM]) for row in rows
    ]
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
