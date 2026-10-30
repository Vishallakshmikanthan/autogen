"""Integration test: MDPAgent executing in SimulationEngine with strict determinism."""

from __future__ import annotations

import pytest

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.decision.mdp_agent import MDPAgent
from autogen.intelligence.decision.gridworld_mdp import GridWorldMDPBuilder
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig


def _build_mdp_simulation(seed: int = 42) -> SimulationEngine:
    config = EnvironmentConfig(width=6, height=6, seed=seed)
    world = GridWorld(config)
    world.reset()

    # Build MDP for this 6x6 grid
    mdp = GridWorldMDPBuilder.build_grid_mdp(
        width=6,
        height=6,
        walls=set(),
        terminals={(5, 5): 10.0},
        step_cost=-0.1,
        slip_prob=0.1,
        gamma=0.95,
    )

    mdp_agent = MDPAgent(agent_id="mdp_agent_1", mdp=mdp)
    random_agent = RandomAgent(agent_id="rand_agent_1", seed=seed)

    world.spawn_agent("mdp_agent_1")
    world.spawn_agent("rand_agent_1")

    engine = SimulationEngine(environment=world, seed=seed)
    p_cfg = PerceptionConfig(radius=4)
    engine.register_agent(mdp_agent, p_cfg)
    engine.register_agent(random_agent, p_cfg)

    return engine


def test_mdp_simulation_determinism() -> None:
    """Verifies that running a simulation with MDPAgent is bit-identical across runs."""
    engine1 = _build_mdp_simulation(seed=333)
    engine2 = _build_mdp_simulation(seed=333)

    hashes1: list[str] = []
    hashes2: list[str] = []

    for _ in range(25):
        s1 = engine1.step()
        s2 = engine2.step()
        hashes1.append(canonical_world_hash(s1))
        hashes2.append(canonical_world_hash(s2))

    assert hashes1 == hashes2, "Simulation diverged non-deterministically with MDPAgent!"

