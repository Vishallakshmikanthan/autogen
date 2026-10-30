"""Game theory solvers, canonical games, and Nash equilibrium oracle tests (Module M07)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.game_theory.contracts import NormalFormGame
from autogen.intelligence.game_theory.matrix_games import (
    make_coordination_game,
    make_hawk_dove,
    make_matching_pennies,
    make_prisoners_dilemma,
    make_stag_hunt,
)
from autogen.intelligence.game_theory.solvers import (
    find_best_responses,
    find_dominant_strategies,
    find_pure_nash_equilibria,
    solve_2x2_mixed_nash,
    solve_bimatrix_support_enumeration,
)
from autogen.intelligence.game_theory.verifier import is_nash_equilibrium


def test_prisoners_dilemma_dominant_and_unique_pure_nash() -> None:
    """Oracle verification: Prisoner's Dilemma unique pure Nash is (Defect, Defect)."""
    game = make_prisoners_dilemma()

    # Invariant 1: Defect (action index 1) strictly dominates Cooperate (index 0) for both players
    dom1 = find_dominant_strategies(game, player_idx=0)
    dom2 = find_dominant_strategies(game, player_idx=1)
    assert dom1["strictly_dominant"] == 1
    assert dom2["strictly_dominant"] == 1

    # Invariant 2: Unique pure strategy Nash equilibrium is (1, 1) -> (Defect, Defect)
    pure_eqs = find_pure_nash_equilibria(game)
    assert len(pure_eqs) == 1
    assert pure_eqs[0].pure_action_indices() == [1, 1]
    assert pure_eqs[0].expected_payoffs == [1.0, 1.0]

    # Verify with independent verifier
    assert is_nash_equilibrium(game, pure_eqs[0])


def test_matching_pennies_unique_mixed_nash_oracle() -> None:
    """Oracle verification: Matching Pennies has no pure Nash and unique mixed Nash ((0.5, 0.5), (0.5, 0.5))."""
    game = make_matching_pennies()

    # Invariant 1: No pure Nash equilibria exist
    pure_eqs = find_pure_nash_equilibria(game)
    assert len(pure_eqs) == 0

    # Invariant 2: Analytic 2x2 solver finds (0.5, 0.5)
    mixed_eqs = solve_2x2_mixed_nash(game)
    assert len(mixed_eqs) == 1
    p1_strat = mixed_eqs[0].probabilities[0]
    p2_strat = mixed_eqs[0].probabilities[1]

    np.testing.assert_allclose(p1_strat, [0.5, 0.5], atol=1e-5)
    np.testing.assert_allclose(p2_strat, [0.5, 0.5], atol=1e-5)
    assert pytest.approx(mixed_eqs[0].expected_payoffs[0], abs=1e-5) == 0.0
    assert pytest.approx(mixed_eqs[0].expected_payoffs[1], abs=1e-5) == 0.0

    # Invariant 3: Support enumeration reproduces the identical mixed equilibrium
    supp_eqs = solve_bimatrix_support_enumeration(game)
    assert len(supp_eqs) == 1
    np.testing.assert_allclose(supp_eqs[0].probabilities[0], [0.5, 0.5], atol=1e-5)


def test_hawk_dove_equilibria() -> None:
    """Verifies Hawk-Dove has 2 pure equilibria and 1 mixed equilibrium."""
    # V=4, C=6 -> p_hawk = V / C = 4 / 6 = 2/3
    game = make_hawk_dove(resource_value=4.0, injury_cost=6.0)

    # Pure Nash: (Hawk, Dove) and (Dove, Hawk)
    pure_eqs = find_pure_nash_equilibria(game)
    assert len(pure_eqs) == 2
    actions = [eq.pure_action_indices() for eq in pure_eqs]
    assert [0, 1] in actions  # (Hawk, Dove)
    assert [1, 0] in actions  # (Dove, Hawk)

    # Mixed Nash: P(Hawk) = 2/3, P(Dove) = 1/3
    mixed_eqs = solve_2x2_mixed_nash(game)
    assert len(mixed_eqs) == 1
    np.testing.assert_allclose(mixed_eqs[0].probabilities[0], [2.0 / 3.0, 1.0 / 3.0], atol=1e-5)
    np.testing.assert_allclose(mixed_eqs[0].probabilities[1], [2.0 / 3.0, 1.0 / 3.0], atol=1e-5)


def test_support_enumeration_3x3_rock_paper_scissors() -> None:
    """Verifies support enumeration on a 3x3 game (Rock-Paper-Scissors)."""
    # Payoffs: Win +1, Lose -1, Tie 0
    # Actions: [0: Rock, 1: Paper, 2: Scissors]
    rps_mat1 = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ], dtype=np.float64)
    rps_mat2 = -rps_mat1

    game = NormalFormGame(
        players=["Player 1", "Player 2"],
        actions=[["Rock", "Paper", "Scissors"], ["Rock", "Paper", "Scissors"]],
        payoff_matrices=[rps_mat1, rps_mat2],
    )

    eqs = solve_bimatrix_support_enumeration(game)
    assert len(eqs) == 1
    expected_prob = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(eqs[0].probabilities[0], expected_prob, atol=1e-4)
    np.testing.assert_allclose(eqs[0].probabilities[1], expected_prob, atol=1e-4)

