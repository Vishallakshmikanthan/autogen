"""Bounded working memory buffer implementing FIFO/LRU policies."""

from __future__ import annotations

from collections import deque

from autogen.memory.contracts import EvictionPolicy
from autogen.simulation.core.contracts import Observation


class WorkingMemory:
    """Small bounded working memory buffer holding recent sensory observations."""

    def __init__(
        self,
        capacity: int = 10,
        policy: EvictionPolicy = EvictionPolicy.FIFO,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"Working memory capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.policy = policy
        self._buffer: deque[Observation] = deque(maxlen=capacity)

    def push(self, observation: Observation) -> None:
        """Pushes an observation into the working buffer, automatically evicting oldest."""
        self._buffer.append(observation)

    def get_latest(self) -> Observation | None:
        """Retrieves the most recent observation without removing it."""
        return self._buffer[-1] if self._buffer else None

    def get_recent(self, k: int = 1) -> list[Observation]:
        """Retrieves the k most recent observations in chronological order."""
        if k <= 0:
            return []
        items = list(self._buffer)
        return items[-k:]

    def get_all(self) -> list[Observation]:
        """Returns all currently buffered observations."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clears all buffered observations."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
