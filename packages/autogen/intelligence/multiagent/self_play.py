"""Competitive self-play with opponent pool, frozen policy snapshots, and Elo tracking (SYS-14, Module M15)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PolicySnapshot:
    """Frozen snapshot of an agent policy checkpoint."""

    snapshot_id: str
    iteration: int
    parameters: dict[str, Any]
    elo_rating: float = 1200.0
    wins: int = 0
    losses: int = 0
    draws: int = 0


class OpponentPool:
    """Historical pool of frozen opponent snapshots for non-stationary self-play evaluation."""

    def __init__(self, k_factor: float = 32.0, seed: int = 42) -> None:
        self.k_factor = float(k_factor)
        self.rng = np.random.default_rng(seed)
        self.snapshots: list[PolicySnapshot] = []

    def add_snapshot(
        self,
        snapshot_id: str,
        iteration: int,
        parameters: dict[str, Any],
        initial_elo: float = 1200.0,
    ) -> PolicySnapshot:
        """Saves a frozen deep-copied parameter snapshot into the opponent pool."""
        snap = PolicySnapshot(
            snapshot_id=snapshot_id,
            iteration=iteration,
            parameters=copy.deepcopy(parameters),
            elo_rating=initial_elo,
        )
        self.snapshots.append(snap)
        return snap

    def sample_opponent(self, strategy: str = "uniform") -> PolicySnapshot:
        """Samples an opponent snapshot from the pool."""
        if not self.snapshots:
            raise ValueError("Opponent pool is empty")

        if strategy == "latest":
            return self.snapshots[-1]
        elif strategy == "random" or strategy == "uniform":
            idx = int(self.rng.integers(0, len(self.snapshots)))
            return self.snapshots[idx]
        else:
            # Elo softmax sampling (prioritize strong opponents)
            ratings = np.array([s.elo_rating for s in self.snapshots])
            probs = np.exp((ratings - np.max(ratings)) / 400.0)
            probs /= np.sum(probs)
            idx = int(self.rng.choice(len(self.snapshots), p=probs))
            return self.snapshots[idx]

    def update_elo(
        self,
        snapshot_a: PolicySnapshot,
        snapshot_b: PolicySnapshot,
        score_a: float,  # 1.0 for A win, 0.5 for draw, 0.0 for B win
    ) -> tuple[float, float]:
        """Updates Elo ratings after a competitive match."""
        ra = snapshot_a.elo_rating
        rb = snapshot_b.elo_rating

        # Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        score_b = 1.0 - score_a

        new_ra = ra + self.k_factor * (score_a - ea)
        new_rb = rb + self.k_factor * (score_b - eb)

        snapshot_a.elo_rating = new_ra
        snapshot_b.elo_rating = new_rb

        if score_a > 0.5:
            snapshot_a.wins += 1
            snapshot_b.losses += 1
        elif score_a < 0.5:
            snapshot_a.losses += 1
            snapshot_b.wins += 1
        else:
            snapshot_a.draws += 1
            snapshot_b.draws += 1

        return new_ra, new_rb
