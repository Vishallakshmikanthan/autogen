"""Unit tests for the GridBeliefFilter (2D spatial histogram filter) (Module M10)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.probabilistic.grid_belief_filter import GridBeliefFilter
from autogen.simulation.state.types import Position


def test_grid_belief_filter_normalization_invariant() -> None:
    """Invariant: The spatial belief grid must always sum to 1.0 with non-negative entries."""
    filter_ = GridBeliefFilter(width=10, height=10)
    assert abs(np.sum(filter_.as_matrix()) - 1.0) < 1e-12

    # Run 10 random prediction and correction steps
    for step in range(10):
        sensor_pos = Position(step % 10, (step * 2) % 10)
        filter_.update((sensor_pos, 3, step % 2 == 0))
        mat = filter_.as_matrix()
        assert np.all(mat >= 0.0)
        assert abs(np.sum(mat) - 1.0) < 1e-10


def test_perfect_sensor_collapses_entropy() -> None:
    """Invariant: Repeated detections by an informative sensor decrease spatial entropy toward 0."""
    filter_ = GridBeliefFilter(width=8, height=8, motion_noise=0.01)
    initial_entropy = filter_.entropy()

    target_pos = Position(4, 4)
    # Consecutive detections directly centered at target_pos with pinpoint accuracy (radius=0)
    for _ in range(5):
        filter_.correct_sensor(sensor_pos=target_pos, sensor_radius=0, target_detected=True)

    final_entropy = filter_.entropy()
    assert final_entropy < initial_entropy
    # Probability at target cell must be dominant
    top_pos, top_prob = filter_.most_likely_states(k=1)[0]
    assert top_pos == target_pos
    assert top_prob > 0.80


def test_useless_sensor_maintains_motion_diffusion() -> None:
    """Invariant: When sensor is uninformative (equal likelihoods), belief evolves according to pure diffusion."""
    filter_ = GridBeliefFilter(width=10, height=10, motion_noise=0.2)
    # Start with a concentrated delta at center
    init_grid = np.zeros((10, 10))
    init_grid[5, 5] = 1.0
    filter_ = GridBeliefFilter(width=10, height=10, initial_distribution=init_grid, motion_noise=0.2)

    ent_0 = filter_.entropy()
    # Predict without informative sensor updates -> diffusion increases entropy
    for _ in range(5):
        filter_.predict()

    ent_diffused = filter_.entropy()
    assert ent_diffused > ent_0
