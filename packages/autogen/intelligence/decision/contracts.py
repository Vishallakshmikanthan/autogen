"""Decision theory and Markov Decision Process (MDP) contracts (SYS-10, Module M11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypeVar, cast, runtime_checkable
import numpy as np

S = TypeVar("S", contravariant=True)
A = TypeVar("A")


@dataclass(frozen=True, slots=True)
class TabularMDP:
    """Explicit finite discrete Markov Decision Process <S, A, T, R, gamma>."""

    num_states: int
    num_actions: int
    states: tuple[str, ...]
    actions: tuple[str, ...]
    transition_tensor: np.ndarray  # shape: (num_states, num_actions, num_states), T[s, a, s']
    reward_tensor: np.ndarray      # shape: (num_states, num_actions, num_states), R[s, a, s']
    gamma: float                   # discount factor in [0, 1]
    terminal_states: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.transition_tensor.shape != (self.num_states, self.num_actions, self.num_states):
            raise ValueError(f"Transition tensor shape {self.transition_tensor.shape} != ({self.num_states}, {self.num_actions}, {self.num_states})")
        if self.reward_tensor.shape != (self.num_states, self.num_actions, self.num_states):
            raise ValueError(f"Reward tensor shape {self.reward_tensor.shape} != ({self.num_states}, {self.num_actions}, {self.num_states})")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"Discount factor gamma must be in [0, 1], got {self.gamma}")

        # Check normalization of transitions: sum_{s'} T(s, a, s') == 1.0 for non-terminals
        row_sums = np.sum(self.transition_tensor, axis=2)
        for s in range(self.num_states):
            if s not in self.terminal_states:
                for a in range(self.num_actions):
                    if not np.isclose(row_sums[s, a], 1.0, atol=1e-5):
                        raise ValueError(f"Transition probabilities from state {s} action {a} sum to {row_sums[s, a]}, expected 1.0")

    @property
    def expected_reward(self) -> np.ndarray:
        """Computes expected immediate reward R(s, a) = sum_{s'} T(s, a, s') * R(s, a, s').

        Shape: (num_states, num_actions).
        """
        return cast(np.ndarray, np.sum(self.transition_tensor * self.reward_tensor, axis=2))


@dataclass(frozen=True, slots=True)
class MDPSolution:
    """Exact optimal solution bundle for a Markov Decision Process."""

    v_star: np.ndarray              # shape: (num_states,)
    q_star: np.ndarray              # shape: (num_states, num_actions)
    policy: np.ndarray              # shape: (num_states,), dtype=int
    iterations: int
    max_residual: float
    converged: bool
    runtime_sec: float
    state_labels: tuple[str, ...] = ()
    action_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    """Multi-attribute utility weights for state evaluation."""

    energy: float = 1.0
    health: float = 2.0
    food: float = 1.5
    hazard_risk: float = 5.0


@runtime_checkable
class UtilityFunction(Protocol[S]):
    """Callable mapping state to scalar utility U(s) -> R."""

    def __call__(self, state: S) -> float:
        ...
