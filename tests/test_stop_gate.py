import pytest

from budget_sid.errors import TestLeakError
from budget_sid.stop_gate import EXIT_OK, EXIT_STOP, run_stop_gate
from budget_sid.synthetic import make_population, oracle_tied_population, uniform_gain_population


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
