"""Integration test: Multi-agent simulation with LogicAgent and symbolic safety filters (Module M08)."""

from __future__ import annotations

import pytest

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.symbolic.logic_agent import LogicAgent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position, TERRAIN_PROPERTIES
from autogen.simulation.state.world_state import canonical_world_hash


def _build_symbolic_simulation(seed: int = 42) -> SimulationEngine:
    config = EnvironmentConfig(
        width=15,
        height=15,
        seed=seed,
    )
    world = GridWorld(config)
    world.reset()

    logic_agent = LogicAgent(agent_id="logic_agent_1")
    random_agent = RandomAgent(agent_id="random_agent_1", seed=seed)

    world.spawn_agent("logic_agent_1")
    world.spawn_agent("random_agent_1")

    engine = SimulationEngine(environment=world, seed=seed)
    p_cfg = PerceptionConfig(radius=5)

    engine.register_agent(logic_agent, p_cfg)
    engine.register_agent(random_agent, p_cfg)

    return engine


def test_symbolic_simulation_determinism() -> None:
    """Verifies that running the simulation with LogicAgent is strictly deterministic."""
    engine1 = _build_symbolic_simulation(seed=101)
    engine2 = _build_symbolic_simulation(seed=101)

    hashes1: list[str] = []
    hashes2: list[str] = []

    for _ in range(50):
        s1 = engine1.step()
        s2 = engine2.step()
        hashes1.append(canonical_world_hash(s1))
        hashes2.append(canonical_world_hash(s2))

    assert hashes1 == hashes2, "Non-deterministic simulation divergence detected!"


def test_logic_agent_never_enters_impassable_cells() -> None:
    """Invariant: LogicAgent must never occupy an impassable obstacle or water cell throughout execution."""
    engine = _build_symbolic_simulation(seed=202)

    for _ in range(30):
        state = engine.step()
        agent = state.agents.get("logic_agent_1")
        if agent is not None and agent.is_alive:
            cell = state.get_cell(agent.position)
            assert TERRAIN_PROPERTIES[cell.terrain].walkable is True, (
                f"LogicAgent entered unwalkable cell at {agent.position} ({cell.terrain})"
            )

