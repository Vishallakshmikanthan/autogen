"""Probabilistic contracts and data interfaces (SYS-09, Module M10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

T = TypeVar("T")  # Outcome / event type
S = TypeVar("S")  # State type
O = TypeVar("O")  # Observation type


@runtime_checkable
class Distribution(Protocol[T]):
    """Generic probability distribution protocol."""

    def prob(self, outcome: T) -> float:
        """Evaluates probability mass P(X = outcome)."""
        ...

    def log_prob(self, outcome: T) -> float:
        """Evaluates log-probability log P(X = outcome)."""
        ...

    def sample(self, rng: np.random.Generator) -> T:
        """Draws an empirical sample according to the distribution."""
        ...

    def support(self) -> tuple[T, ...]:
        """Returns the support of the distribution (all outcomes with non-zero probability)."""
        ...

    def entropy(self) -> float:
        """Computes the Shannon entropy H(X) in bits."""
        ...


@runtime_checkable
class BeliefState(Protocol[S]):
    """Belief state probability distribution over hidden world states."""

    def prob(self, state: S) -> float:
        """Returns belief b(s) = P(S_t = s | o_{1:t})."""
        ...

    def update(self, observation: Any, action: Any | None = None) -> BeliefState[S]:
        """Transitions belief to b_{t+1} using motion model and sensory observation."""
        ...

    def entropy(self) -> float:
        """Computes Shannon entropy H(b) of the belief state."""
        ...

    def most_likely_states(self, k: int = 5) -> list[tuple[S, float]]:
        """Returns the top-k most likely states with their corresponding probabilities."""
        ...


@dataclass(frozen=True, slots=True)
class HMMResult:
    """Output bundle of Hidden Markov Model exact inference algorithms."""

    forward_probs: np.ndarray  # shape (T, num_states)
    backward_probs: np.ndarray  # shape (T, num_states)
    smoothed_posteriors: np.ndarray  # shape (T, num_states)
    most_likely_path: tuple[int, ...]
    log_likelihood: float
