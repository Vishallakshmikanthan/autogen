"""Integration test: Neural agents in deterministic SimulationEngine (SYS-13, Module M14)."""

from __future__ import annotations

from autogen.agents.neural.dqn_agent import DQNAgent
from autogen.agents.neural.ppo_agent import PPOAgent
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.intelligence.neural.dqn import DeepQNetwork
from autogen.intelligence.neural.policy_gradient import PPO
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def test_dqn_agent_simulation_run() -> None:
    """Verifies that DQNAgent executes in SimulationEngine and emits authentic DecisionTraces."""
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    dqn = DeepQNetwork(state_dim=4, num_actions=4, lr=0.005, seed=42)
    agent = DQNAgent(agent_id="dqn_bot", dqn=dqn, initial_epsilon=0.1)
    world.spawn_agent("dqn_bot")

    engine = SimulationEngine(environment=world, agents={"dqn_bot": agent}, seed=42)

    for _ in range(15):
        engine.step()
        trace = agent.latest_trace
        assert trace is not None
        assert trace.agent_id == "dqn_bot"
        assert trace.algorithm_name == "DeepQNetwork"
        assert "mean_q" in trace.metrics
        assert "max_q" in trace.metrics
        assert "candidate_actions" in trace.rationale


def test_ppo_agent_simulation_run() -> None:
    """Verifies that PPOAgent executes in SimulationEngine with authentic DecisionTraces."""
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    ppo = PPO(state_dim=4, num_actions=4, seed=42)
    agent = PPOAgent(agent_id="ppo_bot", ppo=ppo, update_interval=5)
    world.spawn_agent("ppo_bot")

    engine = SimulationEngine(environment=world, agents={"ppo_bot": agent}, seed=42)

    for _ in range(15):
        engine.step()
        trace = agent.latest_trace
        assert trace is not None
        assert trace.agent_id == "ppo_bot"
        assert trace.algorithm_name == "PPOPolicyGradient"
        assert "state_value" in trace.metrics
        assert "entropy" in trace.metrics
        assert "candidate_actions" in trace.rationale


def test_neural_agent_registry_creation() -> None:
    """Verifies instantiation of DQNAgent and PPOAgent through AgentRegistry."""
    dqn_cfg = AgentConfig(agent_id="reg_dqn", agent_type="dqn_agent", seed=100)
    dqn_agent = AgentRegistry.create(dqn_cfg)
    assert isinstance(dqn_agent, DQNAgent)

    ppo_cfg = AgentConfig(agent_id="reg_ppo", agent_type="ppo_agent", seed=200)
    ppo_agent = AgentRegistry.create(ppo_cfg)
    assert isinstance(ppo_agent, PPOAgent)


def test_neural_simulation_determinism() -> None:
    """Verifies bit-identical simulation states across two runs with identical seeds."""
    def run_sim(seed: int) -> list[str]:
        config = EnvironmentConfig(width=6, height=6, seed=seed)
        world = GridWorld(config)
        world.reset()

        dqn = DeepQNetwork(state_dim=4, num_actions=4, lr=0.005, seed=seed)
        agent = DQNAgent(agent_id="det_bot", dqn=dqn, initial_epsilon=0.05)
        world.spawn_agent("det_bot")

        engine = SimulationEngine(environment=world, agents={"det_bot": agent}, seed=seed)
        hashes = []
        for _ in range(15):
            s = engine.step()
            hashes.append(canonical_world_hash(s))
        return hashes

    h1 = run_sim(42)
    h2 = run_sim(42)
    assert h1 == h2
    assert len(h1) == 15
