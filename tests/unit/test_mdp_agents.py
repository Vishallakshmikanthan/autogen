"""Unit tests for MDPAgent and UtilityAgent decision making and traces (Module M11)."""

from __future__ import annotations

import pytest

from autogen.agents.decision.mdp_agent import MDPAgent
from autogen.agents.decision.utility_agent import UtilityAgent
from autogen.intelligence.decision.gridworld_mdp import classic_russell_norvig_grid
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position


def test_mdp_agent_decides_optimal_action() -> None:
    """Verifies that MDPAgent looks up the exact optimal policy action and emits DecisionTrace."""
    mdp = classic_russell_norvig_grid(step_cost=-0.04, slip_prob=0.2, gamma=1.0)
    agent = MDPAgent(agent_id="mdp_tester", mdp=mdp)

    # State (0, 0)
    obs = Observation(
        step=1,
        agent_id="mdp_tester",
        position=Position(0, 0),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        visible_cells=(),
        visible_agents=(),
    )

    agent.observe(obs)
    action, trace = agent.decide()

    assert action.action_type in {ActionType.UP, ActionType.RIGHT}
    assert trace is not None
    assert trace.algorithm_name == "ExactMDPPolicy"
    assert "v_star" in trace.metrics


def test_utility_agent_decides_meu_action() -> None:
    """Verifies that UtilityAgent computes expected utilities and emits DecisionTrace."""
    agent = UtilityAgent(agent_id="utility_tester", grid_width=8, grid_height=8)

    obs = Observation(
        step=1,
        agent_id="utility_tester",
        position=Position(3, 3),
        energy=80.0,
        health=100.0,
        carried_food=0.0,
        visible_cells=(),
        visible_agents=(),
    )

    agent.observe(obs)
    action, trace = agent.decide()

    assert action.action_type in list(ActionType)
    assert trace is not None
    assert trace.algorithm_name == "MultiAttributeUtility"
    assert "best_eu" in trace.metrics
