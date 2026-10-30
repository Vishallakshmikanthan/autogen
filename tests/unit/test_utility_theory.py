"""Unit tests for multi-attribute utility theory and Maximum Expected Utility (Module M11)."""

from __future__ import annotations

import pytest

from autogen.intelligence.decision.contracts import UtilityWeights
from autogen.intelligence.decision.utility import (
    MultiAttributeUtility,
    argmax_expected_utility,
    expected_utility,
)
from autogen.intelligence.probabilistic.distributions import DiscreteDistribution


def test_multi_attribute_utility_linear_combination() -> None:
    """Verifies that MultiAttributeUtility applies weights correctly."""
    weights = UtilityWeights(energy=2.0, health=1.0, food=0.5, hazard_risk=10.0)
    mau = MultiAttributeUtility(weights)

    # U = 2*50 + 1*100 + 0.5*10 - 10*0.2 = 100 + 100 + 5 - 2 = 203.0
    u = mau.evaluate(energy=50.0, health=100.0, food=10.0, hazard_risk=0.2)
    assert abs(u - 203.0) < 1e-10


def test_expected_utility_meu_selection() -> None:
    """Verifies Principle of Maximum Expected Utility (MEU) across lottery distributions."""
    # Action A: 50% chance of 100, 50% chance of 0 -> EU = 50
    # Action B: 100% chance of 60 -> EU = 60 (Superior)
    candidates = ["A", "B"]
    models = {
        "A": DiscreteDistribution({100: 0.5, 0: 0.5}),
        "B": DiscreteDistribution({60: 1.0}),
    }

    best_act, max_eu, all_eu = argmax_expected_utility(
        candidate_actions=candidates,
        transition_model=lambda a: models[a],
        utility_fn=lambda val: float(val),
    )

    assert best_act == "B"
    assert abs(max_eu - 60.0) < 1e-10
    assert abs(all_eu["A"] - 50.0) < 1e-10

