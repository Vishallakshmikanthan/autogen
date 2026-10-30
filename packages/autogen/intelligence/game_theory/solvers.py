"""Game theory solvers: Best response, Dominant strategies, Pure Nash, 2x2 Analytic Mixed Nash, and Support Enumeration (SYS-10, Module M07)."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from autogen.intelligence.game_theory.contracts import MixedStrategyProfile, NormalFormGame
from autogen.intelligence.game_theory.verifier import is_nash_equilibrium


def find_best_responses(
    game: NormalFormGame,
    player_idx: int,
    opponent_strategies: Sequence[np.ndarray],
    tol: float = 1e-6,
) -> list[int]:
    """Finds all pure action indices that maximize expected payoff against opponent strategies."""
    num_acts = len(game.actions[player_idx])
    payoffs = np.zeros(num_acts, dtype=np.float64)

    for a_idx in range(num_acts):
        pure_act = np.zeros(num_acts, dtype=np.float64)
        pure_act[a_idx] = 1.0

        full_profile = list(opponent_strategies)
        full_profile.insert(player_idx, pure_act)

        payoffs[a_idx] = game.expected_payoff(player_idx, full_profile)

    max_val = np.max(payoffs)
    return [i for i, val in enumerate(payoffs) if np.isclose(val, max_val, atol=tol)]


def find_dominant_strategies(
    game: NormalFormGame,
    player_idx: int,
    tol: float = 1e-6,
) -> dict[str, int | None]:
    """Determines if the player has a strictly or weakly dominant pure strategy."""
    num_acts = len(game.actions[player_idx])
    other_players = [i for i in range(game.num_players) if i != player_idx]
    other_act_ranges = [range(len(game.actions[i])) for i in other_players]

    strictly_dominant: int | None = None
    weakly_dominant: int | None = None

    for candidate_a in range(num_acts):
        is_strict = True
        is_weak = True
        has_strict_advantage = False

        for other_a in range(num_acts):
            if candidate_a == other_a:
                continue

            for opp_joint in itertools.product(*other_act_ranges):
                joint_cand = list(opp_joint)
                joint_cand.insert(player_idx, candidate_a)

                joint_other = list(opp_joint)
                joint_other.insert(player_idx, other_a)

                payoff_cand = game.payoff(player_idx, joint_cand)
                payoff_other = game.payoff(player_idx, joint_other)

                if payoff_cand <= payoff_other + tol:
                    is_strict = False
                if payoff_cand < payoff_other - tol:
                    is_weak = False
                if payoff_cand > payoff_other + tol:
                    has_strict_advantage = True

        if is_strict and strictly_dominant is None:
            strictly_dominant = candidate_a
        if is_weak and has_strict_advantage and weakly_dominant is None:
            weakly_dominant = candidate_a

    return {
        "strictly_dominant": strictly_dominant,
        "weakly_dominant": weakly_dominant,
    }


def find_pure_nash_equilibria(game: NormalFormGame) -> list[MixedStrategyProfile]:
    """Computes all pure strategy Nash equilibria by exhaustive joint action enumeration."""
    equilibria: list[MixedStrategyProfile] = []
    act_ranges = [range(len(game.actions[i])) for i in range(game.num_players)]

    for joint_profile in itertools.product(*act_ranges):
        is_nash = True
        for i in range(game.num_players):
            # Opponent profile
            opp_strats: list[np.ndarray] = []
            for j in range(game.num_players):
                if j != i:
                    p = np.zeros(len(game.actions[j]), dtype=np.float64)
                    p[joint_profile[j]] = 1.0
                    opp_strats.append(p)

            br_acts = find_best_responses(game, i, opp_strats)
            if joint_profile[i] not in br_acts:
                is_nash = False
                break

        if is_nash:
            probs: list[np.ndarray] = []
            for i, act_idx in enumerate(joint_profile):
                p = np.zeros(len(game.actions[i]), dtype=np.float64)
                p[act_idx] = 1.0
                probs.append(p)

            payoffs = [game.expected_payoff(i, probs) for i in range(game.num_players)]
            profile = MixedStrategyProfile(probabilities=probs, expected_payoffs=payoffs)
            equilibria.append(profile)

    return equilibria


def solve_2x2_mixed_nash(game: NormalFormGame) -> list[MixedStrategyProfile]:
    """Analytically solves for strictly mixed Nash equilibrium in 2-player 2x2 bimatrix games."""
    if game.num_players != 2 or len(game.actions[0]) != 2 or len(game.actions[1]) != 2:
        raise ValueError("solve_2x2_mixed_nash requires exactly 2 players with 2 actions each.")

    a = game.payoff_matrices[0]
    b = game.payoff_matrices[1]

    # Player 1 plays [p, 1-p] such that Player 2 is indifferent:
    # p*B00 + (1-p)*B10 = p*B01 + (1-p)*B11
    # p * ((B00 - B10) - (B01 - B11)) = B11 - B10
    delta_b = (b[0, 0] - b[1, 0]) - (b[0, 1] - b[1, 1])
    # Player 2 plays [q, 1-q] such that Player 1 is indifferent:
    # q*A00 + (1-q)*A01 = q*A10 + (1-q)*A11
    # q * ((A00 - A01) - (A10 - A11)) = A11 - A01
    delta_a = (a[0, 0] - a[0, 1]) - (a[1, 0] - a[1, 1])

    if np.isclose(delta_b, 0.0, atol=1e-9) or np.isclose(delta_a, 0.0, atol=1e-9):
        return []

    p = (b[1, 1] - b[1, 0]) / delta_b
    q = (a[1, 1] - a[0, 1]) / delta_a

    if 1e-6 < p < 1.0 - 1e-6 and 1e-6 < q < 1.0 - 1e-6:
        prob1 = np.array([p, 1.0 - p], dtype=np.float64)
        prob2 = np.array([q, 1.0 - q], dtype=np.float64)
        profile = MixedStrategyProfile(
            probabilities=[prob1, prob2],
            expected_payoffs=[
                float(prob1 @ a @ prob2),
                float(prob1 @ b @ prob2),
            ],
        )
        if is_nash_equilibrium(game, profile):
            return [profile]

    return []


def solve_bimatrix_support_enumeration(
    game: NormalFormGame,
    tol: float = 1e-5,
) -> list[MixedStrategyProfile]:
    """Support enumeration algorithm computing all pure and mixed Nash equilibria for 2-player bimatrix games."""
    if game.num_players != 2:
        raise ValueError("Support enumeration is specified for 2-player bimatrix games.")

    m = len(game.actions[0])
    n = len(game.actions[1])
    a = game.payoff_matrices[0]
    b = game.payoff_matrices[1]

    found_equilibria: list[MixedStrategyProfile] = []

    # Iterate over all possible support sizes k = 1 .. min(m, n)
    for k in range(1, min(m, n) + 1):
        for supp1 in itertools.combinations(range(m), k):
            for supp2 in itertools.combinations(range(n), k):
                # 1. Solve for Player 2's strategy q on supp2:
                # Player 1 is indifferent between all actions in supp1.
                # If k == 1: q is pure action in supp2
                # If k > 1: (k-1) equations of equal payoff + 1 equation for sum(q) = 1
                q_sub = _solve_indifference_system(a, supp1, supp2)
                if q_sub is None:
                    continue

                # 2. Solve for Player 1's strategy p on supp1:
                # Player 2 is indifferent between all actions in supp2.
                # Notice b.T gives payoffs when player 2 is row player!
                p_sub = _solve_indifference_system(b.T, supp2, supp1)
                if p_sub is None:
                    continue

                # Reconstruct full strategy vectors
                p_full = np.zeros(m, dtype=np.float64)
                for idx, act in enumerate(supp1):
                    p_full[act] = p_sub[idx]

                q_full = np.zeros(n, dtype=np.float64)
                for idx, act in enumerate(supp2):
                    q_full[act] = q_sub[idx]

                profile = MixedStrategyProfile(
                    probabilities=[p_full, q_full],
                    expected_payoffs=[
                        float(p_full @ a @ q_full),
                        float(p_full @ b @ q_full),
                    ],
                )

                # Verify Nash condition across ALL actions (both inside and outside support)
                if is_nash_equilibrium(game, profile, epsilon=tol):
                    # Check if already found (avoid duplicate profiles)
                    if not _contains_profile(found_equilibria, profile):
                        found_equilibria.append(profile)

    return found_equilibria


def _solve_indifference_system(
    payoff_matrix: np.ndarray,
    my_supp: tuple[int, ...],
    opp_supp: tuple[int, ...],
) -> np.ndarray | None:
    """Solves the linear system ensuring the row player is indifferent across my_supp when opp plays opp_supp."""
    k = len(opp_supp)
    if k == 1:
        return np.array([1.0], dtype=np.float64)

    # We need: for i in 1..k-1:
    # sum_j (M[my_supp[0], opp_supp[j]] - M[my_supp[i], opp_supp[j]]) * q_j = 0
    # and sum_j q_j = 1
    mat = np.zeros((k, k), dtype=np.float64)
    rhs = np.zeros(k, dtype=np.float64)

    base_row = my_supp[0]
    for eq_idx in range(k - 1):
        other_row = my_supp[eq_idx + 1]
        for col_idx, opp_col in enumerate(opp_supp):
            mat[eq_idx, col_idx] = payoff_matrix[base_row, opp_col] - payoff_matrix[other_row, opp_col]

    # Normalization equation: sum(q) = 1
    mat[k - 1, :] = 1.0
    rhs[k - 1] = 1.0

    try:
        # Solve linear system
        solution = np.linalg.solve(mat, rhs)
        # Check that probabilities are strictly positive on support (>= 1e-6)
        if np.all(solution >= -1e-6) and np.isclose(np.sum(solution), 1.0, atol=1e-5):
            solution = np.maximum(0.0, solution)
            solution = solution / np.sum(solution)
            return solution
        return None
    except np.linalg.LinAlgError:
        return None


def _contains_profile(
    profiles: list[MixedStrategyProfile],
    candidate: MixedStrategyProfile,
    tol: float = 1e-4,
) -> bool:
    for p in profiles:
        match = True
        for i in range(len(p.probabilities)):
            if not np.allclose(p.probabilities[i], candidate.probabilities[i], atol=tol):
                match = False
                break
        if match:
            return True
    return False

