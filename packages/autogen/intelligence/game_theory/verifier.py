"""Independent equilibrium verification and inequality checker (SYS-10, Module M07)."""

from __future__ import annotations

from typing import Any

import numpy as np

from autogen.intelligence.game_theory.contracts import MixedStrategyProfile, NormalFormGame


def is_nash_equilibrium(
    game: NormalFormGame,
    profile: MixedStrategyProfile,
    epsilon: float = 1e-5,
) -> bool:
    """Verifies the fundamental Nash equilibrium inequality condition:

        U_i(s_i*, s_-i*) >= U_i(s_i, s_-i*) - epsilon

    for every player i and all unilateral pure deviation strategies s_i in A_i.
    """
    valid, _ = verify_equilibrium_inequality(game, profile, epsilon=epsilon)
    return valid


def verify_equilibrium_inequality(
    game: NormalFormGame,
    profile: MixedStrategyProfile,
    epsilon: float = 1e-5,
) -> tuple[bool, dict[str, Any]]:
    """Evaluates expected payoffs and maximal deviation incentives for every player."""
    report: dict[str, Any] = {
        "is_nash": True,
        "max_deviation_incentive": 0.0,
        "player_reports": [],
    }

    # Verify probability simplex validity
    for i, p in enumerate(profile.probabilities):
        if not np.isclose(np.sum(p), 1.0, atol=1e-5):
            report["is_nash"] = False
            report["error"] = f"Player {i} strategy probabilities do not sum to 1: sum={np.sum(p)}"
            return False, report
        if np.any(p < -1e-6):
            report["is_nash"] = False
            report["error"] = f"Player {i} strategy contains negative probabilities"
            return False, report

    # Evaluate each player's deviation incentives
    for i in range(game.num_players):
        u_expected = game.expected_payoff(i, profile.probabilities)
        num_acts = len(game.actions[i])
        max_incentive = float("-inf")
        best_deviation_act = -1

        for a_idx in range(num_acts):
            # Pure deviation for player i
            pure_dev = np.zeros(num_acts, dtype=np.float64)
            pure_dev[a_idx] = 1.0

            dev_profile = list(profile.probabilities)
            dev_profile[i] = pure_dev

            u_dev = game.expected_payoff(i, dev_profile)
            incentive = u_dev - u_expected

            if incentive > max_incentive:
                max_incentive = incentive
                best_deviation_act = a_idx

        player_rep = {
            "player_index": i,
            "player_name": game.players[i],
            "expected_payoff": round(u_expected, 6),
            "max_incentive": round(max_incentive, 6),
            "best_deviation_action": game.actions[i][best_deviation_act],
            "violates_nash": max_incentive > epsilon,
        }
        report["player_reports"].append(player_rep)

        if max_incentive > report["max_deviation_incentive"]:
            report["max_deviation_incentive"] = max_incentive

        if max_incentive > epsilon:
            report["is_nash"] = False

    return bool(report["is_nash"]), report

