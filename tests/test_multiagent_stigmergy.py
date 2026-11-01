"""Unit and mathematical tests for PheromoneField stigmergy dynamics (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.multiagent.stigmergy import PheromoneField
from autogen.simulation.state.types import Position


@pytest.mark.mathematical
def test_stigmergy_deposition_and_evaporation():
    """Verifies chemical concentration deposit bounds and exponential evaporation."""
    field = PheromoneField(width=10, height=10, evaporation_rate=0.1, diffusion_rate=0.0)
    pos = Position(x=5, y=5)

    # Deposit
    field.deposit(pos, amount=50.0)
    assert field.sense(pos) == 50.0

    # Evaporate one step: 50 * (1 - 0.1) = 45.0
    field.step()
    assert np.isclose(field.sense(pos), 45.0)


@pytest.mark.mathematical
def test_stigmergy_laplacian_diffusion():
    """Verifies that discrete 2D Laplacian diffusion spreads concentration to adjacent neighbors."""
    field = PheromoneField(width=7, height=7, evaporation_rate=0.0, diffusion_rate=0.1)
    pos = Position(x=3, y=3)

    field.deposit(pos, amount=40.0)
    field.step()

    # Center loses concentration to 4 neighbors: (3, 2), (3, 4), (2, 3), (4, 3)
    center_val = field.sense(pos)
    assert center_val < 40.0

    up_val = field.sense(Position(x=3, y=2))
    right_val = field.sense(Position(x=4, y=3))
    assert up_val > 0.0
    assert right_val > 0.0
    assert np.isclose(up_val, right_val)


@pytest.mark.unit
def test_stigmergy_gradient_points_uphill():
    """Verifies that spatial gradient vector points towards higher concentration peak."""
    field = PheromoneField(width=10, height=10, evaporation_rate=0.0, diffusion_rate=0.0)

    # Peak placed at (7, 5)
    field.deposit(Position(x=7, y=5), amount=80.0)

    # Point to the left of peak at (6, 5)
    gx, gy = field.gradient(Position(x=6, y=5))
    # Concentration to the right is higher, so grad_x must be positive
    assert gx > 0.0
    assert np.isclose(gy, 0.0)

