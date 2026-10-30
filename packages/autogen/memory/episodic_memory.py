"""Bounded episodic memory buffer for experience transitions."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from autogen.memory.contracts import EpisodicTransition


class EpisodicMemory:
    """Bounded episodic memory buffer recording (t, o, a, r, o', done) transitions."""

    def __init__(self, capacity: int = 500) -> None:
        if capacity < 1:
            raise ValueError(f"Episodic memory capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._buffer: deque[EpisodicTransition] = deque(maxlen=capacity)

    def push(self, transition: EpisodicTransition) -> None:
        """Appends an experience transition to episodic memory."""
        self._buffer.append(transition)

    def get_latest(self) -> EpisodicTransition | None:
        """Returns the most recent experience transition."""
        return self._buffer[-1] if self._buffer else None

    def get_recent(self, limit: int = 50) -> list[EpisodicTransition]:
        """Returns up to limit most recent transitions in chronological order."""
        if limit <= 0:
            return []
        items = list(self._buffer)
        return items[-limit:]

    def get_all(self) -> list[EpisodicTransition]:
        """Returns a list of all stored experience transitions."""
        return list(self._buffer)

    def filter_by_metadata(self, key: str, value: Any) -> list[EpisodicTransition]:
        """Retrieves transitions matching a specific metadata key-value pair."""
        return [t for t in self._buffer if t.metadata.get(key) == value]

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> list[EpisodicTransition]:
        """Uniformly samples a random mini-batch of transitions."""
        if not self._buffer:
            return []
        actual_size = min(batch_size, len(self._buffer))
        generator = rng or np.random.default_rng()
        indices = generator.choice(len(self._buffer), size=actual_size, replace=False)
        items = list(self._buffer)
        return [items[i] for i in indices]

    def clear(self) -> None:
        """Clears all stored episodic experiences."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
