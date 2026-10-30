"""Grid Belief Filter (2D Spatial Histogram Filter) (SYS-09, Module M10)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autogen.intelligence.probabilistic.contracts import BeliefState
from autogen.simulation.state.types import Position


class GridBeliefFilter(BeliefState[Position]):
    """Spatial 2D histogram filter tracking probability distributions of hidden entities across a grid."""

    def __init__(
        self,
        width: int,
        height: int,
        initial_distribution: np.ndarray | None = None,
        motion_noise: float = 0.2,
    ) -> None:
        self.width = width
        self.height = height
        self.motion_noise = motion_noise

        if initial_distribution is not None:
            if initial_distribution.shape != (height, width):
                raise ValueError(f"Initial distribution shape {initial_distribution.shape} must match ({height}, {width})")
            total = float(np.sum(initial_distribution))
            if total <= 0.0:
                raise ValueError("Initial distribution sum must be positive.")
            self._grid = np.array(initial_distribution, dtype=np.float64) / total
        else:
            # Uniform prior across all grid cells
            self._grid = np.full((height, width), 1.0 / (width * height), dtype=np.float64)

    def prob(self, state: Position) -> float:
        """Returns the probability belief b(x, y) of the state position."""
        if 0 <= state.x < self.width and 0 <= state.y < self.height:
            return float(self._grid[state.y, state.x])
        return 0.0

    def predict(self) -> None:
        """Propagates belief through the transition motion model T(s' | s).

        Model: target stays in place with prob (1 - motion_noise), or moves uniformly to an adjacent cell.
        """
        new_grid = np.zeros_like(self._grid)
        p_stay = 1.0 - self.motion_noise
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        p_move = self.motion_noise / len(deltas)

        for y in range(self.height):
            for x in range(self.width):
                b_xy = self._grid[y, x]
                if b_xy <= 0.0:
                    continue

                # Stay in place
                new_grid[y, x] += b_xy * p_stay

                # Move to neighbors
                for dx, dy in deltas:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        new_grid[ny, nx] += b_xy * p_move
                    else:
                        # Bounces back / stays if hitting wall
                        new_grid[y, x] += b_xy * p_move

        total = float(np.sum(new_grid))
        if total > 0.0:
            self._grid = new_grid / total

    def correct_sensor(
        self,
        sensor_pos: Position,
        sensor_radius: int,
        target_detected: bool,
        false_positive_rate: float = 0.05,
        false_negative_rate: float = 0.10,
    ) -> None:
        """Applies sensor observation likelihood O(o | s) and normalizes the belief."""
        likelihoods = np.zeros_like(self._grid)

        for y in range(self.height):
            for x in range(self.width):
                dist = abs(x - sensor_pos.x) + abs(y - sensor_pos.y)
                in_sensor_range = dist <= sensor_radius

                if in_sensor_range:
                    if target_detected:
                        # P(Detected | Target at (x, y)) = 1 - FN
                        likelihoods[y, x] = 1.0 - false_negative_rate
                    else:
                        # P(Not Detected | Target at (x, y)) = FN
                        likelihoods[y, x] = false_negative_rate
                else:
                    if target_detected:
                        # P(Detected | Target outside range) = FP
                        likelihoods[y, x] = false_positive_rate
                    else:
                        # P(Not Detected | Target outside range) = 1 - FP
                        likelihoods[y, x] = 1.0 - false_positive_rate

        updated = self._grid * likelihoods
        total = float(np.sum(updated))
        if total > 0.0:
            self._grid = updated / total
        else:
            # Degenerate case fallback: preserve prior
            pass

    def update(
        self,
        observation: tuple[Position, int, bool],
        action: Any | None = None,
    ) -> GridBeliefFilter:
        """Executes full filter step: predict motion model then correct with sensor reading.

        observation tuple: (sensor_pos, sensor_radius, target_detected)
        """
        self.predict()
        sensor_pos, radius, detected = observation
        self.correct_sensor(sensor_pos, radius, detected)
        return self

    def entropy(self) -> float:
        """Calculates 2D spatial Shannon entropy H(b) in bits."""
        ent = 0.0
        for p in self._grid.flat:
            if p > 0.0:
                ent -= float(p * math.log2(p))
        return ent

    def most_likely_states(self, k: int = 5) -> list[tuple[Position, float]]:
        """Finds top-k cells with highest probability."""
        flat_indices = np.argsort(self._grid.flat)[::-1][:k]
        results: list[tuple[Position, float]] = []
        for idx in flat_indices:
            y, x = divmod(int(idx), self.width)
            results.append((Position(x, y), float(self._grid[y, x])))
        return results

    def as_matrix(self) -> np.ndarray:
        """Returns a copy of the probability matrix for visual rendering and heatmaps."""
        return np.copy(self._grid)
