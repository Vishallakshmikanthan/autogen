"""Unit tests for baseline agent architectures and internal map invariants."""

import pytest

from autogen.agents.baseline.model_based_reflex_agent import ModelBasedReflexAgent
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig, PerceptionMode
from autogen.simulation.perception.engine import PerceptionEngine
from autogen.simulation.state.types import ActionType, Position, TerrainType


def test_random_agent_deliberation():
    agent = RandomAgent(agent_id="rand_1", seed=777)
    obs = Observation(
        agent_id="rand_1",
        step=0,
        position=Position(5, 5),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        is_alive=True,
        visible_cells=(),
        visible_agents=(),
    )
    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    assert action.agent_id == "rand_1"
    assert action.action_type in [
        ActionType.UP,
        ActionType.DOWN,
        ActionType.LEFT,
        ActionType.RIGHT,
        ActionType.STAY,
        ActionType.COLLECT,
    ]
    assert trace.algorithm_name == "random_action_selection"
    assert trace.rationale["candidate_actions"][action.action_type.value] == pytest.approx(1.0 / 6)


def test_simple_reflex_agent_harvest_when_on_food():
    world = GridWorld(config=EnvironmentConfig(width=10, height=10, seed=1))
    world.reset(seed=1)
    agent_pos = Position(3, 3)

    # Set cell (3, 3) to walkable grassland with food
    grid = [list(r) for r in world.state.grid]
    grid[3][3] = grid[3][3].model_copy(update={"terrain": TerrainType.GRASSLAND, "food": 5.0})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("reflex_1", agent_pos)

    engine = PerceptionEngine()
    obs = engine.generate_observation(world.state, "reflex_1")

    agent = SimpleReflexAgent("reflex_1")
    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    assert action.action_type == ActionType.COLLECT
    assert trace.rationale["candidate_actions"][ActionType.COLLECT.value] == 100.0
    assert "harvesting" in trace.rationale["reason"]


def test_simple_reflex_agent_navigates_toward_visible_food():
    world = GridWorld(config=EnvironmentConfig(width=10, height=10, seed=1))
    world.reset(seed=1)
    agent_pos = Position(3, 3)

    # Clear food around (3, 3) except for (4, 3)
    grid = [list(r) for r in world.state.grid]
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            nx, ny = agent_pos.x + dx, agent_pos.y + dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                grid[ny][nx] = grid[ny][nx].model_copy(
                    update={"terrain": TerrainType.GRASSLAND, "food": 0.0}
                )

    # Place food solely at (4, 3) (to the right)
    grid[3][4] = grid[3][4].model_copy(update={"terrain": TerrainType.GRASSLAND, "food": 8.0})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("reflex_1", agent_pos)

    engine = PerceptionEngine()
    obs = engine.generate_observation(world.state, "reflex_1")

    agent = SimpleReflexAgent("reflex_1")
    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    assert action.action_type == ActionType.RIGHT
    assert "Navigating toward visible food" in trace.rationale["reason"]


def test_model_based_reflex_internal_map_matches_ground_truth_only_where_observed():
    """Invariant test: agent internal map equals ground truth ONLY where observed."""
    width, height = 25, 25
    cfg = EnvironmentConfig(width=width, height=height, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)

    agent_pos = Position(12, 12)
    grid = [list(r) for r in world.state.grid]
    grid[12][12] = grid[12][12].model_copy(update={"terrain": TerrainType.GRASSLAND})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("model_1", agent_pos)

    radius = 3
    engine = PerceptionEngine()
    p_cfg = PerceptionConfig(mode=PerceptionMode.LOCAL_WINDOW, radius=radius)
    obs = engine.generate_observation(world.state, "model_1", p_cfg)

    agent = ModelBasedReflexAgent("model_1")
    agent.observe(obs)
    agent.update_belief()

    # 1. Check all coordinates in internal map
    for (mx, my), cell_info in agent.internal_map.items():
        # Coordinate must have been within Chebyshev radius of agent_pos
        assert max(abs(mx - agent_pos.x), abs(my - agent_pos.y)) <= radius
        # Must exactly match ground truth cell in world state
        truth_cell = world.state.grid[my][mx]
        assert cell_info.terrain == truth_cell.terrain
        assert cell_info.food == truth_cell.food
        assert cell_info.walkable == TERRAIN_PROPERTIES[truth_cell.terrain].walkable

    # 2. Check unobserved coordinates outside window are NOT present in internal map
    unobserved_count = 0
    for y in range(height):
        for x in range(width):
            if max(abs(x - agent_pos.x), abs(y - agent_pos.y)) > radius:
                assert (x, y) not in agent.internal_map
                unobserved_count += 1

    assert unobserved_count > 0
    assert len(agent.internal_map) == (2 * radius + 1) ** 2
