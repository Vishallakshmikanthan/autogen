"""Unit and mathematical tests for CommonsResourcePool and Tragedy of the Commons (SYS-14, Module M15)."""

from __future__ import annotations

import pytest

from autogen.intelligence.multiagent.commons import CommonsResourcePool


@pytest.mark.mathematical
def test_commons_sustainable_harvest_regrowth():
    """Verifies that moderate sustainable harvest allows resource pool to replenish."""
    pool = CommonsResourcePool(carrying_capacity=500.0, growth_rate=0.2, collapse_threshold=50.0)

    # Harvest below maximum sustainable yield
    for _ in range(10):
        harvested = pool.harvest("agent_sust", 10.0)
        assert harvested == 10.0
        info = pool.step_regrowth()
        assert not info["collapsed"]
        assert pool.current_stock > 100.0


@pytest.mark.mathematical
def test_commons_overharvest_tipping_point_collapse():
    """Verifies that over-harvest past the threshold triggers non-linear collapse and cessation of regrowth."""
    pool = CommonsResourcePool(carrying_capacity=500.0, growth_rate=0.2, collapse_threshold=100.0)

    # Aggressive over-harvest pushing pool down to 40 (below collapse_threshold 100)
    pool.harvest("greedy_agent_1", 260.0)
    pool.harvest("greedy_agent_2", 200.0)
    assert pool.current_stock == 40.0

    # Step regrowth: should trip collapse flag and yield zero regrowth
    info = pool.step_regrowth()
    assert info["collapsed"] is True
    assert info["regrowth"] == 0.0
    assert pool.current_stock == 40.0

    # Drain to 0
    pool.harvest("greedy_agent_1", 50.0)
    assert pool.current_stock == 0.0
    assert pool.depletion_step is not None
