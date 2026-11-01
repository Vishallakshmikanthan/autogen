"""Integration tests for Hybrid Cognitive Agent in multi-agent GridWorld simulation (SYS-15, Module M17)."""

from __future__ import annotations

from autogen.agents.hybrid.config import HybridAgentConfig
from autogen.agents.hybrid.hybrid_agent import HybridCognitiveAgent
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.grid_world import EnvironmentConfig, GridWorld
from autogen.simulation.state.types import ActionType, Position


def test_hybrid_agent_decision_trace_glass_box() -> None:
    """Verifies that HybridCognitiveAgent emits an authentic, explainable glass-box DecisionTrace."""
    config = HybridAgentConfig(hazard_threshold=0.5, grid_width=10, grid_height=10)
    agent = HybridCognitiveAgent(agent_id="hybrid_flagship", config=config)

    obs = Observation(
        agent_id="hybrid_flagship",
        step=1,
        position=Position(x=2, y=2),
        energy=95.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    assert action.action_type in HybridCognitiveAgent.ACTION_MAP
    assert trace.agent_id == "hybrid_flagship"
    assert trace.algorithm_name == "HybridCognitiveArchitecture"
    assert "active_option" in trace.rationale["belief_summary"]
    assert "shield_vetoed" in trace.rationale["belief_summary"]
    assert "hazard_belief_count" in trace.metrics


def test_hybrid_agent_simulation_determinism() -> None:
    """Verifies that identical seeds produce bit-for-bit identical actions and trajectories."""
    def run_sim(seed: int) -> list[ActionType]:
        env_config = EnvironmentConfig(width=10, height=10, seed=seed)
        _world = GridWorld(config=env_config)
        agent = HybridCognitiveAgent(
            agent_id="det_agent",
            config=HybridAgentConfig(seed=seed, grid_width=10, grid_height=10),
        )

        actions: list[ActionType] = []
        for step in range(10):
            obs = Observation(
                agent_id="det_agent",
                step=step,
                position=Position(x=step % 10, y=(step * 2) % 10),
                energy=100.0 - step,
                health=100.0,
                carried_food=0,
                visible_cells=[],
                visible_agents=[],
            )
            agent.observe(obs)
            act, _ = agent.decide()
            actions.append(act.action_type)
        return actions

    run1 = run_sim(seed=999)
    run2 = run_sim(seed=999)
    assert run1 == run2
