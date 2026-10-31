"""Integration test: Multi-Agent systems in deterministic SimulationEngine (SYS-14, Module M15)."""

from __future__ import annotations

from autogen.agents.multiagent.cooperative_agent import CooperativeAgent
from autogen.agents.multiagent.iql_agent import IQLAgent
from autogen.agents.multiagent.marl_agent import MARLAgent
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.intelligence.multiagent.communication import CommunicationBus
from autogen.intelligence.multiagent.stigmergy import PheromoneField
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def test_multiagent_simulation_run() -> None:
    """Verifies that IQLAgent and CooperativeAgent run concurrently in SimulationEngine emitting authentic DecisionTraces."""
    config = EnvironmentConfig(width=8, height=8, seed=42)
    world = GridWorld(config)
    world.reset()

    field = PheromoneField(width=8, height=8)
    bus = CommunicationBus()

    iql_bot = IQLAgent(agent_id="iql_bot")
    coop_bot = CooperativeAgent(agent_id="coop_bot", pheromone_field=field, comm_bus=bus)

    world.spawn_agent("iql_bot")
    world.spawn_agent("coop_bot")

    engine = SimulationEngine(environment=world, agents={"iql_bot": iql_bot, "coop_bot": coop_bot}, seed=42)

    for _ in range(15):
        engine.step()

        t_iql = iql_bot.latest_trace
        assert t_iql is not None
        assert t_iql.algorithm_name == "IndependentQLearning"
        assert "candidate_actions" in t_iql.rationale

        t_coop = coop_bot.latest_trace
        assert t_coop is not None
        assert t_coop.algorithm_name == "CooperativeCollective"
        assert "pheromone_conc" in t_coop.metrics


def test_multiagent_agent_registry_creation() -> None:
    """Verifies declarative factory instantiation of multi-agent types via AgentRegistry."""
    cfg_iql = AgentConfig(agent_id="reg_iql", agent_type="iql_agent", seed=10)
    agent_iql = AgentRegistry.create(cfg_iql)
    assert isinstance(agent_iql, IQLAgent)

    cfg_marl = AgentConfig(agent_id="reg_marl", agent_type="marl_agent", seed=20)
    agent_marl = AgentRegistry.create(cfg_marl)
    assert isinstance(agent_marl, MARLAgent)

    cfg_coop = AgentConfig(agent_id="reg_coop", agent_type="cooperative_agent", seed=30)
    agent_coop = AgentRegistry.create(cfg_coop)
    assert isinstance(agent_coop, CooperativeAgent)


def test_multiagent_simulation_determinism() -> None:
    """Verifies bit-identical simulation states across two multi-agent runs with identical seeds."""
    def run_sim(seed: int) -> list[str]:
        config = EnvironmentConfig(width=6, height=6, seed=seed)
        world = GridWorld(config)
        world.reset()

        a1 = IQLAgent(agent_id="agent_1")
        a2 = IQLAgent(agent_id="agent_2")
        world.spawn_agent("agent_1")
        world.spawn_agent("agent_2")

        engine = SimulationEngine(environment=world, agents={"agent_1": a1, "agent_2": a2}, seed=seed)
        hashes = []
        for _ in range(15):
            s = engine.step()
            hashes.append(canonical_world_hash(s))
        return hashes

    h1 = run_sim(42)
    h2 = run_sim(42)
    assert h1 == h2
    assert len(h1) == 15
