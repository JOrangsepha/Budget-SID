import pytest

from budget_sid.errors import TestLeakError
from budget_sid.experiment import command_evaluate_budgets
from budget_sid.stop_gate import (
    EXIT_OK,
    EXIT_STOP,
    budgeted_positive_assignment,
    has_concentrated_gain,
    oracle_allocation,
    run_stop_gate,
)
from budget_sid.synthetic import (
    make_population,
    oracle_tied_population,
    selective_oracle_counterexample,
    uniform_gain_population,
)


def test_stop_gate_passes_when_gain_is_concentrated():
    rows = [row for row in make_population() if row["split"] == "val"]
    result = run_stop_gate(rows)
    assert result.exit_code == EXIT_OK
    assert result.concentrated_gain is True
    assert result.oracle_beats_reference is True
    assert result.reasons == []
    assert result.oracle_ndcg == pytest.approx(1.0)
    assert result.reference_ndcg == pytest.approx(20 / 24)
    assert result.mean_delta_ndcg == pytest.approx(4 / 24)


def test_stop_gate_exit_2_when_gain_not_concentrated():
    result = run_stop_gate(uniform_gain_population())
    assert result.exit_code == EXIT_STOP
    assert result.concentrated_gain is False
    assert result.oracle_beats_reference is True
    assert result.reasons == ["no_concentrated_gain"]


def test_stop_gate_exit_2_when_oracle_loses_to_beam_40():
    result = run_stop_gate(oracle_tied_population())
    assert result.exit_code == EXIT_STOP
    assert result.concentrated_gain is True
    assert result.oracle_beats_reference is False
    assert result.reasons == ["oracle_not_better_than_uniform_beam_40"]
    assert result.oracle_ndcg == pytest.approx(result.reference_ndcg)


def test_stop_gate_rejects_test_split():
    with pytest.raises(TestLeakError):
        run_stop_gate(make_population())


def _force_fill_ndcg(rows):
    """Rejected policy: always spend the full large-beam cap, including negative Δ."""
    from budget_sid.benefit import n_large_for_budget

    deltas = [row["ndcg_by_beam"][60] - row["ndcg_by_beam"][20] for row in rows]
    n_large = n_large_for_budget(len(rows))
    order = sorted(range(len(rows)), key=lambda index: (-deltas[index], rows[index]["user_id"]))
    upgraded = set(order[:n_large])
    chosen = [
        row["ndcg_by_beam"][60] if index in upgraded else row["ndcg_by_beam"][20]
        for index, row in enumerate(rows)
    ]
    return sum(chosen) / len(chosen)


def test_selective_oracle_beats_force_fill_when_mean_delta_is_negative():
    """10 requests: 2 gain from beam 60, 8 are hurt. Mean Δ is -0.6.

    Force-filling five large beams scores 0.7 and loses to uniform beam 40
    (0.8). Upgrading only the two positive requests scores 1.0 at mean beam 28.
    """
    rows = selective_oracle_counterexample()
    deltas = [row["delta_ndcg"] for row in rows]
    concentrated, stats = has_concentrated_gain(deltas)
    assert stats["mean_delta"] == pytest.approx(-0.6)
    assert stats["positive_user_fraction"] == pytest.approx(0.2)
    assert concentrated is True

    assigned = budgeted_positive_assignment(rows)
    assert assigned["n_budget"] == 5
    assert assigned["n_upgraded"] == 2
    assert assigned["n_positive"] == 2
    assert assigned["upgraded_user_ids"] == ["gain-00", "gain-01"]
    assert assigned["mean_beam"] == pytest.approx(28.0)
    assert assigned["mean_beam"] <= 40.0
    assert assigned["oracle_ndcg"] == pytest.approx(1.0)
    assert assigned["reference_ndcg"] == pytest.approx(0.8)
    assert assigned["small_ndcg"] == pytest.approx(0.8)
    assert assigned["large_ndcg"] == pytest.approx(0.2)
    assert assigned["beats_reference"] is True

    force_fill = _force_fill_ndcg(rows)
    assert force_fill == pytest.approx(0.7)
    assert assigned["oracle_ndcg"] > force_fill
    assert force_fill < assigned["reference_ndcg"]

    oracle_ndcg, reference_ndcg, beats = oracle_allocation(rows)
    assert oracle_ndcg == pytest.approx(1.0)
    assert reference_ndcg == pytest.approx(0.8)
    assert beats is True

    result = run_stop_gate(rows)
    assert result.exit_code == EXIT_OK
    assert result.concentrated_gain is True
    assert result.oracle_beats_reference is True
    assert result.mean_delta_ndcg == pytest.approx(-0.6)
    assert result.reasons == []


def test_concentrated_gain_when_mean_delta_is_zero():
    concentrated, stats = has_concentrated_gain([1.0] * 5 + [-1.0] * 5)
    assert stats["mean_delta"] == pytest.approx(0.0)
    assert concentrated is True


def test_oracle_uses_budget_on_positive_delta_only():
    rows = []
    for index in range(2):
        rows.append(
            {
                "user_id": f"big-{index}",
                "split": "val",
                "ndcg_by_beam": {20: 0.0, 40: 0.0, 60: 1.0},
            }
        )
    for index in range(8):
        rows.append(
            {
                "user_id": f"small-{index:02d}",
                "split": "val",
                "ndcg_by_beam": {20: 0.0, 40: 0.0, 60: 0.2},
            }
        )
    assigned = budgeted_positive_assignment(rows)
    assert assigned["n_positive"] == 10
    assert assigned["n_budget"] == 5
    assert assigned["n_upgraded"] == 5
    assert assigned["mean_beam"] == pytest.approx(40.0)
    assert assigned["oracle_ndcg"] == pytest.approx(0.26)
    assert assigned["upgraded_user_ids"] == ["big-0", "big-1", "small-00", "small-01", "small-02"]


def test_oracle_allocation_rejects_test_rows():
    rows = selective_oracle_counterexample()
    rows[0] = {**rows[0], "split": "test"}
    with pytest.raises(TestLeakError):
        oracle_allocation(rows)


@pytest.mark.parametrize("population", [uniform_gain_population, oracle_tied_population])
def test_gate_failure_does_not_load_test(beauty_config, population):
    def load_test():
        raise AssertionError("test loader called after gate failure")

    payload = command_evaluate_budgets(
        beauty_config,
        True,
        records=population(),
        load_test=load_test,
    )
    assert payload["exit_code"] == EXIT_STOP
    assert payload["ok"] is False
    assert payload["test_scored"] is False
    assert "test_metrics" not in payload
    assert payload["stop_gate"]["exit_code"] == EXIT_STOP


def test_gate_pass_loads_test_once(beauty_config):
    rows = make_population()
    calls = {"n": 0}

    def load_test():
        calls["n"] += 1
        return [row for row in rows if row["split"] == "test"]

    payload = command_evaluate_budgets(beauty_config, True, records=rows, load_test=load_test)
    assert calls["n"] == 1
    assert payload["exit_code"] == EXIT_OK
    assert payload["test_scored"] is True
    assert "test_metrics" in payload
