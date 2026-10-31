"""Integration test: RL agents in deterministic SimulationEngine (SYS-12, Module M13)."""

from __future__ import annotations

import pytest

from autogen.agents.rl.dyna_agent import DynaAgent
from autogen.agents.rl.q_learning_agent import QLearningAgent
from autogen.intelligence.rl.dyna import DynaQ
from autogen.intelligence.rl.tabular import QLearning
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def test_q_learning_agent_simulation_run() -> None:
    """Verifies that QLearningAgent runs in SimulationEngine and emits authentic DecisionTraces."""
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    ql = QLearning(num_actions=4, alpha=0.2, gamma=0.95, epsilon=0.1, seed=42)
    agent = QLearningAgent(agent_id="rl_bot", q_learning=ql)
    world.spawn_agent("rl_bot")

    engine = SimulationEngine(environment=world, agents={"rl_bot": agent}, seed=42)

    for _ in range(15):
        engine.step()
        trace = agent.latest_trace
        assert trace is not None
        assert trace.agent_id == "rl_bot"
        assert trace.algorithm_name == "TabularQLearning"
        assert "is_exploratory" in trace.metrics
        assert "epsilon" in trace.metrics
        assert "candidate_actions" in trace.rationale


def test_dyna_agent_simulation_run() -> None:
    """Verifies that DynaAgent runs in SimulationEngine with authentic DecisionTraces."""
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    dyna = DynaQ(num_actions=4, alpha=0.2, gamma=0.95, n_planning_steps=5, seed=42)
    agent = DynaAgent(agent_id="dyna_bot", dyna_q=dyna)
    world.spawn_agent("dyna_bot")

    engine = SimulationEngine(environment=world, agents={"dyna_bot": agent}, seed=42)

    for _ in range(15):
        engine.step()
        trace = agent.latest_trace
        assert trace is not None
        assert trace.agent_id == "dyna_bot"
        assert trace.algorithm_name == "ModelBasedDynaQ"
        assert "n_planning_steps" in trace.metrics


def test_rl_agent_simulation_determinism() -> None:
    """Verifies bit-identical simulation states across two runs with identical seeds."""
    def run_sim(seed: int) -> list[str]:
        config = EnvironmentConfig(width=6, height=6, seed=seed)
        world = GridWorld(config)
        world.reset()

        ql = QLearning(num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.05, seed=seed)
        agent = QLearningAgent(agent_id="agent_det", q_learning=ql)
        world.spawn_agent("agent_det")

        engine = SimulationEngine(environment=world, agents={"agent_det": agent}, seed=seed)
        hashes = []
        for _ in range(15):
            s = engine.step()
            hashes.append(canonical_world_hash(s))
        return hashes

    h1 = run_sim(42)
    h2 = run_sim(42)
    assert h1 == h2

