"""Unit tests for conflict resolution policies."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.simulation.core.conflict_resolution import ConflictResolver
from autogen.simulation.core.contracts import Action, ConflictPolicy
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType, Position, TerrainType


@pytest.mark.unit
def test_conflict_cancel_all_policy():
    """Under CANCEL_ALL, two agents attempting to enter the same cell are both stopped."""
    config = EnvironmentConfig(
        width=6,
        height=6,
        seed=42,
        terrain_weights={
            TerrainType.GRASSLAND: 1.0,
            TerrainType.FOREST: 0.0,
            TerrainType.DESERT: 0.0,
            TerrainType.WATER: 0.0,
            TerrainType.MOUNTAIN: 0.0,
            TerrainType.OBSTACLE: 0.0,
        },
    )
    world = GridWorld(config)
    world.reset()

    # Place agent_a at (1, 2) and agent_b at (3, 2)
    # Both move toward target cell (2, 2)
    world.spawn_agent("agent_a", position=Position(1, 2))
    world.spawn_agent("agent_b", position=Position(3, 2))

    resolver = ConflictResolver(conflict_policy=ConflictPolicy.CANCEL_ALL)
    actions = {
        "agent_a": Action(agent_id="agent_a", action_type=ActionType.RIGHT),
        "agent_b": Action(agent_id="agent_b", action_type=ActionType.LEFT),
    }

    rng = np.random.default_rng(123)
    arbitrated = resolver.resolve_conflicts(world.state, actions, rng)

    assert arbitrated["agent_a"].action_type == ActionType.STAY
    assert arbitrated["agent_b"].action_type == ActionType.STAY
    assert arbitrated["agent_a"].parameters.get("collision_cancelled") is True
    assert arbitrated["agent_b"].parameters.get("collision_cancelled") is True


@pytest.mark.unit
def test_conflict_seeded_priority_policy():
    """Under SEEDED_PRIORITY, exactly one agent enters the target cell; other agent stays."""
    config = EnvironmentConfig(
        width=6,
        height=6,
        seed=42,
        terrain_weights={
            TerrainType.GRASSLAND: 1.0,
            TerrainType.FOREST: 0.0,
            TerrainType.DESERT: 0.0,
            TerrainType.WATER: 0.0,
            TerrainType.MOUNTAIN: 0.0,
            TerrainType.OBSTACLE: 0.0,
        },
    )
    world = GridWorld(config)
    world.reset()

    world.spawn_agent("agent_1", position=Position(1, 2))
    world.spawn_agent("agent_2", position=Position(3, 2))

    resolver = ConflictResolver(conflict_policy=ConflictPolicy.SEEDED_PRIORITY)
    actions = {
        "agent_1": Action(agent_id="agent_1", action_type=ActionType.RIGHT),
        "agent_2": Action(agent_id="agent_2", action_type=ActionType.LEFT),
    }

    rng = np.random.default_rng(999)
    arbitrated = resolver.resolve_conflicts(world.state, actions, rng)

    actions_types = [arbitrated["agent_1"].action_type, arbitrated["agent_2"].action_type]
    # Exactly one agent keeps its movement action, the other is substituted with STAY
    assert ActionType.STAY in actions_types
    assert (ActionType.RIGHT in actions_types) or (ActionType.LEFT in actions_types)
