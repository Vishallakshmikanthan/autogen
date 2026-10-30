"""Unit tests for the mathematical world model and environment engine."""

from __future__ import annotations

import pytest

from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import (
    TERRAIN_PROPERTIES,
    ActionType,
    Position,
)


@pytest.mark.unit
def test_out_of_bounds_rejection():
    config = EnvironmentConfig(width=8, height=8, seed=123)
    world = GridWorld(config)
    world.reset()

    # Spawn agent at (0, 0) if walkable, or find a boundary cell
    agent = world.spawn_agent(
        "agent_boundary", position=Position(0, 0) if world.is_walkable(Position(0, 0)) else None
    )
    initial_pos = agent.position

    # Attempt to step outside grid boundaries
    if initial_pos.x == 0:
        assert not world.validate_action("agent_boundary", ActionType.LEFT)
        state_after = world.apply_transitions({"agent_boundary": ActionType.LEFT})
        assert state_after.agents["agent_boundary"].position == initial_pos
    if initial_pos.y == 0:
        assert not world.validate_action("agent_boundary", ActionType.UP)
        state_after = world.apply_transitions({"agent_boundary": ActionType.UP})
        assert state_after.agents["agent_boundary"].position == initial_pos


@pytest.mark.unit
def test_stay_action_always_valid():
    world = GridWorld()
    world.reset()
    agent = world.spawn_agent("agent_stay")
    assert world.validate_action("agent_stay", ActionType.STAY)

    state = world.apply_transitions({"agent_stay": ActionType.STAY})
    assert state.agents["agent_stay"].position == agent.position
    # Stay incurs stay energy cost
    assert state.agents["agent_stay"].energy == pytest.approx(agent.energy - world.config.cost_stay)


@pytest.mark.unit
def test_obstacle_collision_rejection():
    world = GridWorld(EnvironmentConfig(width=10, height=10, seed=42))
    world.reset()

    # Find an obstacle cell adjacent to a walkable cell
    found_pair: tuple[Position, ActionType] | None = None
    for y in range(world.config.height):
        for x in range(world.config.width):
            pos = Position(x, y)
            if world.is_walkable(pos):
                for act, (dx, dy) in [
                    (ActionType.UP, (0, -1)),
                    (ActionType.DOWN, (0, 1)),
                    (ActionType.LEFT, (-1, 0)),
                    (ActionType.RIGHT, (1, 0)),
                ]:
                    target = pos.offset(dx, dy)
                    if world.is_in_bounds(target) and not world.is_walkable(target):
                        found_pair = (pos, act)
                        break
            if found_pair:
                break
        if found_pair:
            break

    if found_pair is not None:
        pos, illegal_act = found_pair
        world.spawn_agent("agent_collider", position=pos)
        assert not world.validate_action("agent_collider", illegal_act)

        next_state = world.apply_transitions({"agent_collider": illegal_act})
        # Movement into wall rejected, agent stays in place
        assert next_state.agents["agent_collider"].position == pos


@pytest.mark.unit
def test_energy_bounds_and_clamping():
    config = EnvironmentConfig(width=6, height=6, energy_max=50.0, seed=1)
    world = GridWorld(config)
    world.reset()
    world.spawn_agent("agent_energy", energy=5.0)

    # Perform multiple actions without eating to verify non-negative clamping
    for _ in range(50):
        state = world.apply_transitions({"agent_energy": ActionType.STAY})
        current_energy = state.agents["agent_energy"].energy
        assert 0.0 <= current_energy <= 50.0
        if current_energy == 0.0:
            break

    assert state.agents["agent_energy"].energy == 0.0


@pytest.mark.unit
def test_resource_regeneration_bounds():
    config = EnvironmentConfig(width=6, height=6, seed=99)
    world = GridWorld(config)
    initial_state = world.reset()

    # Step environment 50 times with no agents
    state = initial_state
    for _ in range(50):
        state = world.apply_transitions({})

    for y in range(config.height):
        for x in range(config.width):
            cell = state.grid[y][x]
            max_capacity = TERRAIN_PROPERTIES[cell.terrain].food_capacity
            assert 0.0 <= cell.food <= max_capacity + 1e-9


@pytest.mark.unit
def test_starvation_and_health_separation_rule():
    """Validates that health damage occurs IF AND ONLY IF energy is 0.0."""
    config = EnvironmentConfig(
        width=6,
        height=6,
        energy_max=10.0,
        health_max=100.0,
        starvation_damage=10.0,
        cost_stay=2.0,
        seed=10,
    )
    world = GridWorld(config)
    world.reset()

    agent = world.spawn_agent("test_agent", energy=4.0, health=100.0)
    assert agent.energy == 4.0
    assert agent.health == 100.0

    # Tick 1: Energy drops from 4.0 -> 2.0. Health MUST NOT take starvation damage.
    state1 = world.apply_transitions({"test_agent": ActionType.STAY})
    a1 = state1.agents["test_agent"]
    assert a1.energy == pytest.approx(2.0)
    assert a1.health == 100.0
    assert a1.starvation_ticks == 0

    # Tick 2: Energy drops from 2.0 -> 0.0. Energy hits zero, triggering tick 1 of starvation!
    state2 = world.apply_transitions({"test_agent": ActionType.STAY})
    a2 = state2.agents["test_agent"]
    assert a2.energy == 0.0
    assert a2.starvation_ticks == 1
    assert a2.health == pytest.approx(90.0)

    # Tick 3: Energy remains 0.0. Another starvation tick damages health by 10.0 -> 80.0
    state3 = world.apply_transitions({"test_agent": ActionType.STAY})
    a3 = state3.agents["test_agent"]
    assert a3.energy == 0.0
    assert a3.starvation_ticks == 2
    assert a3.health == pytest.approx(80.0)
