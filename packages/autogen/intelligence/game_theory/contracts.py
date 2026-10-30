"""Game theory contracts, data models, and protocols (SYS-10, Module M07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class NormalFormGame:
    """Strategic normal-form game representation G = (N, A, U)."""

    players: list[str]
    actions: list[list[str]]  # actions[player_idx] = list of action names
    payoff_matrices: list[np.ndarray]  # payoff_matrices[player_idx] has shape (len(A_0), len(A_1), ...)

    @property
    def num_players(self) -> int:
        return len(self.players)

    def payoff(self, player_idx: int, action_indices: Sequence[int]) -> float:
        """Returns the payoff for player_idx given a joint pure action profile."""
        return float(self.payoff_matrices[player_idx][tuple(action_indices)])

    def expected_payoff(self, player_idx: int, strategy_profile: Sequence[np.ndarray]) -> float:
        """Computes the expected payoff for player_idx under a mixed strategy profile."""
        # For 2-player games: E[U_1] = p^T * A * q, E[U_2] = p^T * B * q
        if self.num_players == 2:
            p = strategy_profile[0]
            q = strategy_profile[1]
            matrix = self.payoff_matrices[player_idx]
            return float(p @ matrix @ q)

        # General N-player expectation via tensor contraction
        weights = strategy_profile[0]
        for strat in strategy_profile[1:]:
            weights = np.tensordot(weights, strat, axes=0)
        return float(np.sum(self.payoff_matrices[player_idx] * weights))


@dataclass(slots=True)
class MixedStrategyProfile:
    """Strategy profile containing a probability distribution over actions per player."""

    probabilities: list[np.ndarray]  # probabilities[i] has length len(A_i), sum == 1.0
    expected_payoffs: list[float] = field(default_factory=list)

    @property
    def is_pure(self) -> bool:
        """Returns True if every player plays a pure action with probability 1.0."""
        for prob in self.probabilities:
            if not np.any(np.isclose(prob, 1.0, atol=1e-5)):
                return False
        return True

    def pure_action_indices(self) -> list[int] | None:
        """Returns list of pure action indices if all players play pure strategies, else None."""
        indices: list[int] = []
        for prob in self.probabilities:
            max_idx = int(np.argmax(prob))
            if np.isclose(prob[max_idx], 1.0, atol=1e-5):
                indices.append(max_idx)
            else:
                return None
        return indices

