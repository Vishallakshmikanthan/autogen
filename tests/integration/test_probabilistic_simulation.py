"""Integration test: BayesianHazardAgent in SimulationEngine with strict determinism."""

from __future__ import annotations

import pytest

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.probabilistic.bayesian_hazard_agent import BayesianHazardAgent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig


def _build_probabilistic_simulation(seed: int = 42) -> SimulationEngine:
    config = EnvironmentConfig(width=12, height=12, seed=seed)
    world = GridWorld(config)
    world.reset()

    hazard_agent = BayesianHazardAgent(agent_id="bayes_agent_1", grid_width=12, grid_height=12)
    opponent = RandomAgent(agent_id="predator_1", seed=seed)

    world.spawn_agent("bayes_agent_1")
    world.spawn_agent("predator_1")

    engine = SimulationEngine(environment=world, seed=seed)
    p_cfg = PerceptionConfig(radius=4)
    engine.register_agent(hazard_agent, p_cfg)
    engine.register_agent(opponent, p_cfg)

    return engine


def test_probabilistic_simulation_determinism() -> None:
    """Verifies that running a simulation with BayesianHazardAgent is bit-identical across runs."""
    engine1 = _build_probabilistic_simulation(seed=777)
    engine2 = _build_probabilistic_simulation(seed=777)

    hashes1: list[str] = []
    hashes2: list[str] = []

    for _ in range(30):
        s1 = engine1.step()
        s2 = engine2.step()
        hashes1.append(canonical_world_hash(s1))
        hashes2.append(canonical_world_hash(s2))

    assert hashes1 == hashes2, "Non-deterministic simulation divergence in probabilistic agent simulation!"
