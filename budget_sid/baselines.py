"""Fixed, random, entropy, and benefit-aware beam policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from budget_sid.constants import LARGE_BEAM, REFERENCE_BEAM, SMALL_BEAM
from budget_sid.routing import RouteDecision


@dataclass
class FixedBeamPolicy:
    beam_width: int
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"fixed_{self.beam_width}"

    def decide(self, features: dict) -> RouteDecision:
        del features
        return RouteDecision(int(self.beam_width), None, self.name)


@dataclass
class RandomBudgetPolicy:
    """Assign the large beam with the probability that matches the reference cost."""

    seed: int
    small_beam: int = SMALL_BEAM
    large_beam: int = LARGE_BEAM
    reference_beam: int = REFERENCE_BEAM
    name: str = "random_budget"

    def __post_init__(self) -> None:
        self.p_large = (self.reference_beam - self.small_beam) / (self.large_beam - self.small_beam)
        self._rng = np.random.default_rng(self.seed)

    def decide(self, features: dict) -> RouteDecision:
        del features
        width = self.large_beam if float(self._rng.random()) < self.p_large else self.small_beam
        return RouteDecision(int(width), None, self.name)


@dataclass
class EntropyPolicy:
    threshold: float
    small_beam: int = SMALL_BEAM
    large_beam: int = LARGE_BEAM
    name: str = "entropy"

    def decide(self, features: dict) -> RouteDecision:
        width = self.large_beam if features["first_step_entropy"] >= self.threshold else self.small_beam
        return RouteDecision(int(width), None, self.name)


@dataclass
class BenefitAwarePolicy:
    predictor: object
    threshold: float
    small_beam: int = SMALL_BEAM
    large_beam: int = LARGE_BEAM
    name: str = "benefit_aware"

    def decide(self, features: dict) -> RouteDecision:
        from budget_sid.features import stack_features

        score = float(self.predictor.predict(stack_features([features]))[0])
        width = self.large_beam if score >= self.threshold else self.small_beam
        return RouteDecision(int(width), score, self.name)


def build_policies(predictor, threshold: float, entropy_threshold: float, seed: int) -> dict:
    return {
        "fixed_20": FixedBeamPolicy(SMALL_BEAM),
        "fixed_40": FixedBeamPolicy(REFERENCE_BEAM),
        "fixed_60": FixedBeamPolicy(LARGE_BEAM),
        "random_budget": RandomBudgetPolicy(seed=seed),
        "entropy": EntropyPolicy(threshold=entropy_threshold),
        "benefit_aware": BenefitAwarePolicy(predictor=predictor, threshold=threshold),
    }
