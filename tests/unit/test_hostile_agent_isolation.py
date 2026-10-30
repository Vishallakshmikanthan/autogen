"""Unit test verifying that hostile agents cannot reach or mutate simulation internals."""

from __future__ import annotations

import pytest

from autogen.agents.base.dummy_agents import HostileAgent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


@pytest.mark.unit
def test_hostile_agent_cannot_mutate_observation_or_env():
    """Validates that immutable data contracts prevent hostile state mutation."""
    world = GridWorld(EnvironmentConfig(width=6, height=6, seed=42))
    world.reset()
    world.spawn_agent("hostile_one", energy=25.0)

    engine = SimulationEngine(environment=world, seed=42)
    hostile = HostileAgent("hostile_one")
    engine.register_agent(hostile)

    # Step engine — hostile agent will try to mutate observation.energy to 999999.0
    engine.step()

    # Verify that mutation was blocked by frozen model
    assert hostile.mutation_attempt_failed is True
    # Verify that ground truth in world state was NOT mutated
    actual_energy = world.state.agents["hostile_one"].energy
    assert actual_energy < 50.0  # Remained around initial energy minus stay cost
