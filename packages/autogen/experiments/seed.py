"""Deterministic seed generation and train/eval seed separation (SYS-16)."""

from __future__ import annotations

import numpy as np


class SeedSplitter:
    """Provides deterministic, strictly segregated RNG seed streams for training and evaluation."""

    def __init__(self, root_seed: int = 42) -> None:
        self.root_seed = root_seed
        self._seq = np.random.SeedSequence(root_seed)

    def spawn_seeds(self, count: int) -> list[int]:
        """Spawns a sequence of distinct child integer seeds."""
        children = self._seq.spawn(count)
        return [int(child.generate_state(1)[0]) for child in children]

    @staticmethod
    def get_range_seeds(start: int, count: int) -> list[int]:
        """Returns deterministic contiguous sequence of seeds [start, start + count)."""
        return list(range(start, start + count))

