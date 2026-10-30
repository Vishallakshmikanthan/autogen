"""Property-based tests for world connectivity and invariant preservation."""

from __future__ import annotations

from collections import deque

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from autogen.core.rng import SeedManager
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.generator import ProceduralWorldGenerator
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import (
    TERRAIN_PROPERTIES,
    ActionType,
)


def count_connected_components(grid_world: GridWorld) -> int:
    """Helper to verify that all walkable cells form exactly one connected component."""
    state = grid_world.state
    w, h = state.width, state.height
    visited: set[tuple[int, int]] = set()
    components = 0

    for y in range(h):
        for x in range(w):
            cell = state.grid[y][x]
            if TERRAIN_PROPERTIES[cell.terrain].walkable and (x, y) not in visited:
                components += 1
                queue = deque([(x, y)])
                visited.add((x, y))

                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and (nx, ny) not in visited
                            and TERRAIN_PROPERTIES[state.grid[ny][nx].terrain].walkable
                        ):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
    return components


@pytest.mark.property_based
def test_connectivity_guarantee_across_worlds():
    """Invariant: The procedural generator produces a 100% connected walkable topology.

    Tested across 1,000 distinct seeds.
    """
    for seed in range(1000):
        config = EnvironmentConfig(width=8, height=8, seed=seed)
        seed_mgr = SeedManager(seed)
        gen = ProceduralWorldGenerator(config, seed_mgr.spawn_stream("generator"))
        cell_grid = gen.generate()

        # Check BFS connectivity on cell_grid
        visited: set[tuple[int, int]] = set()
        components = 0
        w, h = config.width, config.height

        for y in range(h):
            for x in range(w):
                cell = cell_grid[y][x]
                if TERRAIN_PROPERTIES[cell.terrain].walkable and (x, y) not in visited:
                    components += 1
                    queue = deque([(x, y)])
                    visited.add((x, y))

                    while queue:
                        cx, cy = queue.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 <= nx < w
                                and 0 <= ny < h
                                and (nx, ny) not in visited
                                and TERRAIN_PROPERTIES[cell_grid[ny][nx].terrain].walkable
                            ):
                                visited.add((nx, ny))
                                queue.append((nx, ny))

        assert components == 1, f"Seed {seed} generated {components} disconnected components!"


@pytest.mark.property_based
@given(
    actions=st.lists(
        st.sampled_from(
            [
                ActionType.UP,
                ActionType.DOWN,
                ActionType.LEFT,
                ActionType.RIGHT,
                ActionType.STAY,
                ActionType.COLLECT,
            ]
        ),
        min_size=10,
        max_size=50,
    )
)
@settings(max_examples=25)
def test_random_action_sequences_preserve_state_invariants(actions: list[ActionType]):
    """Property test: Arbitrary action sequences never violate fundamental physical bounds."""
    config = EnvironmentConfig(width=10, height=10, seed=42)
    world = GridWorld(config)
    world.reset()
    world.spawn_agent("agent_0")

    for act in actions:
        state = world.apply_transitions({"agent_0": act})
        agent = state.agents["agent_0"]

        # Invariants:
        assert 0.0 <= agent.energy <= config.energy_max
        assert 0.0 <= agent.health <= config.health_max
        assert 0 <= agent.position.x < config.width
        assert 0 <= agent.position.y < config.height
        assert world.is_walkable(agent.position)
