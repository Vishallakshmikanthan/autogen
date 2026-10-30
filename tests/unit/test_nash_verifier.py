"""Independent Epsilon-Nash equilibrium verifier tests (Module M07)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.game_theory.contracts import MixedStrategyProfile
from autogen.intelligence.game_theory.matrix_games import (
    make_matching_pennies,
    make_prisoners_dilemma,
)
from autogen.intelligence.game_theory.verifier import (
    is_nash_equilibrium,
    verify_equilibrium_inequality,
)


def test_verifier_accepts_valid_nash() -> None:
    """Verifies that the verifier correctly accepts true pure and mixed equilibria."""
    pd_game = make_prisoners_dilemma()
    # (Defect, Defect)
    valid_profile = MixedStrategyProfile(
        probabilities=[
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ]
    )
    is_nash, report = verify_equilibrium_inequality(pd_game, valid_profile)
    assert is_nash is True
    assert report["max_deviation_incentive"] <= 1e-5


def test_verifier_rejects_non_nash_profile() -> None:
    """Verifies that the verifier detects profitable deviations in non-equilibria."""
    pd_game = make_prisoners_dilemma()
    # (Cooperate, Cooperate) -> not Nash, deviation incentive is Temptation (5) - Reward (3) = 2.0
    non_nash_profile = MixedStrategyProfile(
        probabilities=[
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
        ]
    )
    is_nash, report = verify_equilibrium_inequality(pd_game, non_nash_profile)
    assert is_nash is False
    assert pytest.approx(report["max_deviation_incentive"], abs=1e-5) == 2.0


def test_verifier_rejects_perturbed_mixed_nash() -> None:
    """Perturbing Matching Pennies (0.5, 0.5) to (0.6, 0.4) creates a profitable deviation for the opponent."""
    mp_game = make_matching_pennies()
    perturbed_profile = MixedStrategyProfile(
        probabilities=[
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
        ]
    )
    # Player 2 now has incentive to play Tails (prob=1) against (0.6, 0.4)
    is_nash = is_nash_equilibrium(mp_game, perturbed_profile, epsilon=1e-5)
    assert is_nash is False


def test_verifier_rejects_malformed_probabilities() -> None:
    """Asserts that invalid probabilities (sum != 1 or negative) are rejected."""
    pd_game = make_prisoners_dilemma()
    bad_sum = MixedStrategyProfile(
        probabilities=[
            np.array([0.5, 0.3]),  # sum = 0.8
            np.array([0.5, 0.5]),
        ]
    )
    assert is_nash_equilibrium(pd_game, bad_sum) is False

    negative_p = MixedStrategyProfile(
        probabilities=[
            np.array([-0.2, 1.2]),
            np.array([0.5, 0.5]),
        ]
    )
    assert is_nash_equilibrium(pd_game, negative_p) is False

