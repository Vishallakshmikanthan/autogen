"""Data contracts and interfaces for Reinforcement Learning Engine (SYS-12, Module M13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypeVar, runtime_checkable
import numpy as np


@dataclass(frozen=True, slots=True)
class StepResult:
    """Standard Gym-style transition tuple produced by environment adapters."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class StateEncoder(Protocol):
    """Encodes rich simulation observations into discrete keys or feature arrays."""

    def encode(self, observation: Any) -> Any:
        ...

    @property
    def num_states(self) -> int | None:
        ...


@dataclass(frozen=True, slots=True)
class EvaluationMetric:
    """Evaluation summary over held-out evaluation seeds."""

    mean_return: float
    std_return: float
    mean_length: float
    td_error_mean: float
    epsilon: float


@dataclass
class QTable:
    """Tabular Q-value storage with automatic state initialization."""

    num_actions: int = 4
    default_value: float = 0.0
    values: dict[Any, np.ndarray] = field(default_factory=dict)

    def get_q(self, state: Any) -> np.ndarray:
        if state not in self.values:
            self.values[state] = np.full(self.num_actions, self.default_value, dtype=np.float64)
        return self.values[state]

    def set_q(self, state: Any, action: int, value: float) -> None:
        if state not in self.values:
            self.values[state] = np.full(self.num_actions, self.default_value, dtype=np.float64)
        self.values[state][action] = value

    def best_action(self, state: Any, rng: np.random.Generator | None = None) -> int:
        q_vals = self.get_q(state)
        max_q = float(np.max(q_vals))
        best_actions = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
        if len(best_actions) == 1:
            return int(best_actions[0])
        if rng is not None:
            return int(rng.choice(best_actions))
        return int(best_actions[0])


@runtime_checkable
class RLAlgorithm(Protocol):
    """Protocol for tabular reinforcement learning control algorithms."""

    def select_action(self, state: Any, explore: bool = True) -> int:
        ...

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        ...

