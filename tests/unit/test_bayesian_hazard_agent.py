"""Unit tests for BayesianHazardAgent expected utility deliberation and decision traces."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.agents.probabilistic.bayesian_hazard_agent import BayesianHazardAgent
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import (
    ActionType,
    Position,
    TerrainType,
)


def test_bayesian_hazard_agent_avoids_high_belief_hazard() -> None:
    """Verifies that BayesianHazardAgent chooses actions minimizing expected hazard collision."""
    agent = BayesianHazardAgent(agent_id="hazard_avoider", grid_width=10, grid_height=10)

    # Set high probability of hazard directly North of agent at (5, 4)
    # Agent is at (5, 5). North is (5, 4).
    high_hazard_grid = np.full((10, 10), 0.001)
    high_hazard_grid[4, 5] = 0.90  # y=4, x=5
    agent.belief_filter._grid = high_hazard_grid / np.sum(high_hazard_grid)

    obs = Observation(
        step=1,
        agent_id="hazard_avoider",
        position=Position(5, 5),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        visible_cells=(),
        visible_agents=(),
    )

    agent.observe(obs)
    action, trace = agent.decide()

    # Agent must NOT move UP into the high hazard cell
    assert action.action_type != ActionType.UP
    assert trace is not None
    assert trace.algorithm_name == "BayesianBeliefFilter"
    assert "belief_summary" in trace.rationale
    assert "top_states" in trace.rationale["belief_summary"]
