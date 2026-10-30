"""Unit tests verifying DecisionTrace completeness and explainability contracts."""

import pytest

from autogen.agents.baseline.model_based_reflex_agent import ModelBasedReflexAgent
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.engine import PerceptionEngine
from autogen.simulation.state.types import Position, TerrainType


@pytest.mark.parametrize(
    "agent_cls,kwargs",
    [
        (RandomAgent, {"seed": 42}),
        (SimpleReflexAgent, {}),
        (ModelBasedReflexAgent, {}),
    ],
)
def test_decision_trace_completeness_and_schema(agent_cls, kwargs):
    cfg = EnvironmentConfig(width=15, height=15, seed=123)
    world = GridWorld(config=cfg)
    world.reset(seed=123)
    # Ensure (7, 7) is walkable
    grid = [list(r) for r in world.state.grid]
    grid[7][7] = grid[7][7].model_copy(update={"terrain": TerrainType.GRASSLAND, "food": 5.0})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("agent_test", Position(7, 7))

    engine = PerceptionEngine()
    obs = engine.generate_observation(world.state, "agent_test")

    agent = agent_cls(agent_id="agent_test", **kwargs)
    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    # Structural assertions
    assert trace.agent_id == "agent_test"
    assert trace.step_id == 0
    assert trace.timestamp_ns > 0
    assert len(trace.algorithm_name) > 0
    assert trace.decision_type == "action_selection"

    # Rationale assertions
    rationale = trace.rationale
    assert "mechanism" in rationale and len(rationale["mechanism"]) > 0
    assert "selected_action" in rationale
    assert rationale["selected_action"] == action.action_type.value
    assert "reason" in rationale and len(rationale["reason"]) > 0
    assert "candidate_actions" in rationale

    candidates = rationale["candidate_actions"]
    assert isinstance(candidates, dict)
    assert len(candidates) > 0
    # Every candidate score must be a valid float
    for _act_name, score in candidates.items():
        assert isinstance(score, (float, int))
