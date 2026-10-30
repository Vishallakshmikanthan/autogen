"""Decision-theoretic utility functions and Expected Utility maximization (SYS-10, Module M11)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from autogen.intelligence.decision.contracts import UtilityWeights
from autogen.intelligence.probabilistic.distributions import DiscreteDistribution

S = TypeVar("S")
A = TypeVar("A")


class MultiAttributeUtility:
    """Multi-attribute linear utility function evaluating state features."""

    def __init__(self, weights: UtilityWeights | None = None) -> None:
        self.weights = weights or UtilityWeights()

    def evaluate(
        self,
        energy: float,
        health: float,
        food: float,
        hazard_risk: float,
    ) -> float:
        """Computes scalar utility: U(s) = w_e * E + w_h * H + w_f * F - w_r * Risk."""
        return (
            self.weights.energy * energy
            + self.weights.health * health
            + self.weights.food * food
            - self.weights.hazard_risk * hazard_risk
        )


def expected_utility(
    action: A,
    outcome_distribution: DiscreteDistribution[S] | Mapping[S, float],
    utility_fn: Callable[[S], float],
) -> float:
    """Calculates Expected Utility: EU(a) = sum_{s'} P(s' | a) * U(s')."""
    eu = 0.0
    if isinstance(outcome_distribution, DiscreteDistribution):
        for outcome in outcome_distribution.support():
            p = outcome_distribution.prob(outcome)
            if p > 0.0:
                eu += p * utility_fn(outcome)
    else:
        for outcome, p in outcome_distribution.items():
            if p > 0.0:
                eu += p * utility_fn(outcome)
    return eu


def argmax_expected_utility(
    candidate_actions: Sequence[A],
    transition_model: Callable[[A], DiscreteDistribution[S] | Mapping[S, float]],
    utility_fn: Callable[[S], float],
) -> tuple[A, float, dict[A, float]]:
    """Applies the Principle of Maximum Expected Utility (MEU) across candidate actions.

    Returns:
        (best_action, max_eu, all_action_utilities)
    """
    if not candidate_actions:
        raise ValueError("Candidate actions cannot be empty.")

    utilities: dict[A, float] = {}
    for act in candidate_actions:
        dist = transition_model(act)
        eu = expected_utility(act, dist, utility_fn)
        utilities[act] = eu

    best_action = max(candidate_actions, key=lambda a: utilities[a])
    return best_action, utilities[best_action], utilities

