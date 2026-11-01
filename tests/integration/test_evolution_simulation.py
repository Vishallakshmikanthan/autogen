"""Integration tests for evolutionary and ALife agents in multi-agent GridWorld simulation (SYS-15, Module M16)."""

from __future__ import annotations

from autogen.agents.evolution.evolved_agent import EvolvedAgent
from autogen.agents.evolution.organism_agent import OrganismAgent
from autogen.intelligence.evolution.alife import OrganismTraits
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.grid_world import EnvironmentConfig, GridWorld
from autogen.simulation.state.types import ActionType, Position


def test_evolved_agent_decide_and_trace() -> None:
    """Verifies EvolvedAgent produces valid actions and authentic DecisionTraces."""
    controller = NeuralNetwork([
        DenseLayer(in_features=4, out_features=5, seed=42),
    ])

    agent = EvolvedAgent(
        agent_id="evolved_01",
        controller=controller,
        state_dim=4,
        generation=12,
        genome_id="gene_abc",
    )

    obs = Observation(
        agent_id="evolved_01",
        step=1,
        position=Position(x=2, y=3),
        energy=85.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    assert action.action_type in EvolvedAgent.ACTION_MAP
    assert trace.agent_id == "evolved_01"
    assert trace.algorithm_name == "EvolvedController"
    assert trace.metrics["generation"] == 12
    assert trace.rationale["belief_summary"]["genome_id"] == "gene_abc"


def test_organism_agent_metabolism_and_trace() -> None:
    """Verifies OrganismAgent deliberates with dynamic metabolic traits and emits glass-box traces."""
    traits = OrganismTraits(speed=1.5, vision_range=3, metabolic_rate=1.2)
    agent = OrganismAgent(agent_id="organism_01", traits=traits, generation=2)

    obs = Observation(
        agent_id="organism_01",
        step=1,
        position=Position(x=5, y=5),
        energy=60.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    agent.observe(obs)
    action, trace = agent.decide()

    assert action.action_type in OrganismAgent.ACTION_MAP
    assert trace.agent_id == "organism_01"
    assert trace.algorithm_name == "OrganismMetabolicController"
    assert trace.metrics["metabolic_rate"] == 1.2
    assert trace.rationale["belief_summary"]["vision_range"] == 3


def test_evolution_simulation_determinism() -> None:
    """Verifies that running two identical seeded simulations produces identical actions and trajectories."""
    def run_sim(seed: int) -> list[ActionType]:
        config = EnvironmentConfig(width=10, height=10, seed=seed)
        _world = GridWorld(config=config)
        controller = NeuralNetwork([DenseLayer(in_features=4, out_features=5, seed=seed)])
        agent = EvolvedAgent(agent_id="agent_det", controller=controller, state_dim=4)

        actions: list[ActionType] = []
        for step in range(10):
            obs = Observation(
                agent_id="agent_det",
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

    actions_run1 = run_sim(seed=777)
    actions_run2 = run_sim(seed=777)
    assert actions_run1 == actions_run2
