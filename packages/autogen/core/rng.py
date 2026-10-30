"""Deterministic pseudo-random number generator management.

Follows ADR-004: Every stochastic decision flows from an explicit seed
through named RNG streams spawned via numpy.random.SeedSequence.
Training seeds and evaluation seeds are strictly partitioned.
"""

from __future__ import annotations

import numpy as np


class SeedManager:
    """Manages independent, deterministic pseudo-random number streams."""

    def __init__(self, master_seed: int = 42) -> None:
        if master_seed < 0:
            raise ValueError(f"master_seed must be non-negative, got {master_seed}")
        self._master_seed = master_seed
        self._root_seq = np.random.SeedSequence(master_seed)
        self._named_streams: dict[str, np.random.Generator] = {}

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def spawn_stream(self, name: str) -> np.random.Generator:
        """Spawns an independent Generator for a named subsystem.

        If a stream with this name was already spawned, it returns that existing stream
        to preserve its sequential state.
        """
        if name in self._named_streams:
            return self._named_streams[name]

        child_seq = self._root_seq.spawn(1)[0]
        gen = np.random.default_rng(child_seq)
        self._named_streams[name] = gen
        return gen

    def get_train_generator(self, run_index: int) -> np.random.Generator:
        """Spawns a dedicated Generator for training run with index `run_index`."""
        return self.spawn_stream(f"train_run_{run_index}")

    def get_eval_generator(self, run_index: int) -> np.random.Generator:
        """Spawns a dedicated Generator for evaluation run with index `run_index`."""
        return self.spawn_stream(f"eval_run_{run_index}")
