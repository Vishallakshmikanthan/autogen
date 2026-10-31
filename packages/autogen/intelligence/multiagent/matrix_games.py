"""Repeated normal-form matrix games: Prisoner's Dilemma, Stag Hunt, and Matching Pennies (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np


class MatrixGame:
    """Two-player bimatrix normal-form stage game."""

    def __init__(self, name: str, payoff_matrix: np.ndarray, action_names: tuple[str, ...]) -> None:
        self.name = name
        # payoff_matrix shape: (num_actions_1, num_actions_2, 2)
        self.payoffs = np.asarray(payoff_matrix, dtype=np.float64)
        self.action_names = action_names

    def get_payoffs(self, action_1: int, action_2: int) -> tuple[float, float]:
        """Returns (reward_1, reward_2) for chosen action indices."""
        return float(self.payoffs[action_1, action_2, 0]), float(self.payoffs[action_1, action_2, 1])

    @classmethod
    def prisoners_dilemma(cls) -> MatrixGame:
        """Standard Prisoner's Dilemma (T=5, R=3, P=1, S=0). Defection is strictly dominant."""
        # Actions: 0 = Cooperate (C), 1 = Defect (D)
        matrix = np.array([
            [[3.0, 3.0], [0.0, 5.0]],  # C vs C, C vs D
            [[5.0, 0.0], [1.0, 1.0]],  # D vs C, D vs D
        ])
        return cls("PrisonersDilemma", matrix, ("Cooperate", "Defect"))

    @classmethod
    def stag_hunt(cls) -> MatrixGame:
        """Coordination game: Stag=4 (Pareto optimal), Hare=2 (Risk dominant)."""
        # Actions: 0 = Stag, 1 = Hare
        matrix = np.array([
            [[4.0, 4.0], [0.0, 3.0]],  # Stag vs Stag, Stag vs Hare
            [[3.0, 0.0], [2.0, 2.0]],  # Hare vs Stag, Hare vs Hare
        ])
        return cls("StagHunt", matrix, ("Stag", "Hare"))

    @classmethod
    def matching_pennies(cls) -> MatrixGame:
        """Zero-sum game with no pure Nash equilibrium."""
        # Actions: 0 = Heads, 1 = Tails
        matrix = np.array([
            [[1.0, -1.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ])
        return cls("MatchingPennies", matrix, ("Heads", "Tails"))
