import numpy as np
import pytest

from budget_sid.baselines import BenefitAwarePolicy, EntropyPolicy, RandomBudgetPolicy, build_policies
from budget_sid.benefit import fit_benefit_predictor
from budget_sid.errors import RoutingProtocolError
from budget_sid.features import extract_features
from budget_sid.recommender import FirstStepState, ScriptedRecommender
from budget_sid.routing import route_request, run_small_then_rerun_large
from budget_sid.synthetic import CATALOG


class CountingRecommender:
    def __init__(self):
        self.first_calls = []
        self.continue_calls = []
        self.full_decode_calls = 0
        self.last_continuation_s = 0.5

    def first_step(self, history):
        self.first_calls.append(tuple(history))
        return FirstStepState(
            history=tuple(history),
            logits=np.array([3.0, 0.0, 0.0, 0.0]),
            elapsed_s=0.2,
        )

    def continue_from_first_step(self, state, beam_width):
        self.continue_calls.append(int(beam_width))
        return [999, *range(1, 40)]

    def decode_full(self, history, beam_width):
        self.full_decode_calls += 1
        self.first_step(history)
        return self.continue_from_first_step(FirstStepState(tuple(history), np.zeros(4)), beam_width)


def test_router_shared_first_step_does_not_rerun():
    recommender = CountingRecommender()
    policy = build_policies(
        predictor=fit_benefit_predictor(np.zeros((2, 4)), np.array([0.0, 1.0])),
        threshold=10.0,
        entropy_threshold=0.5,
        seed=0,
    )["fixed_20"]
    result = route_request(recommender, [1, 2, 3, 4], policy, catalog=list(range(30)))
    assert recommender.full_decode_calls == 0
    assert result.first_step_calls == 1
    assert result.continuation_calls == 1
    assert result.beam_width == 20
    assert 999 not in result.recommendations
    assert len(result.recommendations) == 20
    assert result.latency_s >= 0.7
    assert set(result.latency_breakdown) == {"first_step", "route_decision", "continuation", "legal_filter"}


def test_forbidden_small_then_large_raises():
    with pytest.raises(RoutingProtocolError):
        run_small_then_rerun_large([1, 2], small=20, large=60)


def test_baselines_return_top20_legal_ids():
    predictor = fit_benefit_predictor(
        np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        np.array([0.0, 1.0]),
    )
    policies = build_policies(predictor, threshold=0.0, entropy_threshold=0.5, seed=0)
    expected = {"fixed_20": 20, "fixed_40": 40, "fixed_60": 60}
    assert set(policies) == {"fixed_20", "fixed_40", "fixed_60", "random_budget", "entropy", "benefit_aware"}
    for name, policy in policies.items():
        recommender = CountingRecommender()
        result = route_request(recommender, [1, 2, 3, 4, 5], policy, catalog=list(range(30)))
        assert recommender.full_decode_calls == 0
        assert len(result.recommendations) == 20
        assert all(item in range(30) for item in result.recommendations)
        if name in expected:
            assert result.beam_width == expected[name]
        else:
            assert result.beam_width in (20, 60)


def test_random_budget_mean_width():
    policy = RandomBudgetPolicy(seed=0)
    widths = [policy.decide({"first_step_entropy": 0.0}).beam_width for _ in range(4000)]
    assert set(widths) <= {20, 60}
    assert abs(float(np.mean(widths)) - 40.0) < 1.5


def test_entropy_routes_uncertain_requests_wider():
    policy = EntropyPolicy(threshold=0.5)
    uncertain = extract_features(1, np.zeros(8))
    certain = extract_features(12, np.array([10.0, 0, 0, 0, 0, 0, 0, 0], dtype=float))
    assert policy.decide(uncertain).beam_width == 60
    assert policy.decide(certain).beam_width == 20


def test_benefit_policy_uses_predicted_gain():
    predictor = fit_benefit_predictor(
        np.array(
            [
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 2.0, 0.0, 0.2],
            ]
        ),
        np.array([0.0, 1.0]),
        alpha=1e-4,
    )
    policy = BenefitAwarePolicy(predictor=predictor, threshold=0.5)
    low = {"history_length": 1.0, "first_step_entropy": 0.0, "candidate_prob_gap": 1.0, "top1_prob": 1.0}
    high = {"history_length": 1.0, "first_step_entropy": 2.0, "candidate_prob_gap": 0.0, "top1_prob": 0.2}
    assert policy.decide(low).beam_width == 20
    assert policy.decide(high).beam_width == 60
    assert policy.decide(high).predicted_delta > policy.decide(low).predicted_delta


def test_scripted_recommender_is_deterministic():
    left = ScriptedRecommender(catalog=CATALOG, seed=1)
    right = ScriptedRecommender(catalog=CATALOG, seed=1)
    left_state = left.first_step([1, 2, 3])
    right_state = right.first_step([1, 2, 3])
    assert np.allclose(left_state.logits, right_state.logits)
    assert left.continue_from_first_step(left_state, 40) == right.continue_from_first_step(right_state, 40)
    assert left.full_decode_calls == 0
