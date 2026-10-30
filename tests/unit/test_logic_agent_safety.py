"""Unit tests for SafetyRuleLayer and LogicAgent symbolic decision making (Module M08)."""

from __future__ import annotations

import pytest

from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.agents.symbolic.logic_agent import LogicAgent
from autogen.agents.symbolic.safety_layer import SafetyRuleLayer
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.simulation.state.world_state import CellState


def _make_observation_with_hazards() -> Observation:
    # Agent at (5, 5). Obstacle at (5, 4) (UP), Water at (6, 5) (RIGHT), Grassland elsewhere
    cells: list[CellState] = []
    for x in range(3, 8):
        for y in range(3, 8):
            p = Position(x, y)
            t = TerrainType.GRASSLAND
            if p == Position(5, 4):
                t = TerrainType.OBSTACLE
            elif p == Position(6, 5):
                t = TerrainType.WATER
            cells.append(CellState(position=p, terrain=t, food=0.0, movement_cost=1.0))

    return Observation(
        step=1,
        agent_id="logic_agent_1",
        position=Position(5, 5),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        visible_cells=tuple(cells),
        visible_agents=(),
    )


def test_safety_rule_layer_action_filtering() -> None:
    safety = SafetyRuleLayer()
    obs = _make_observation_with_hazards()
    safety.ground_perceptions(obs)

    # Moving UP into (5, 4) should be provably unsafe
    unsafe_up, proof_up = safety.is_action_provably_unsafe(Position(5, 5), ActionType.UP)
    assert unsafe_up is True
    assert proof_up is not None
    assert proof_up.is_proved is True

    # Moving RIGHT into (6, 5) should be provably unsafe
    unsafe_right, proof_right = safety.is_action_provably_unsafe(Position(5, 5), ActionType.RIGHT)
    assert unsafe_right is True
    assert proof_right is not None

    # Moving DOWN into (5, 6) is safe grassland
    unsafe_down, proof_down = safety.is_action_provably_unsafe(Position(5, 5), ActionType.DOWN)
    assert unsafe_down is False
    assert proof_down is None


def test_logic_agent_decision_and_trace() -> None:
    agent = LogicAgent(agent_id="logic_1")
    obs = _make_observation_with_hazards()

    agent.observe(obs)
    agent.update_belief()
    action, trace = agent.decide()

    # Invariant: Action must never be UP or RIGHT (the provably unsafe actions)
    assert action.action_type not in [ActionType.UP, ActionType.RIGHT]
    assert action.action_type in [ActionType.DOWN, ActionType.LEFT, ActionType.STAY]

    # Verify DecisionTrace
    assert trace.agent_id == "logic_1"
    assert trace.algorithm_name == "LogicDeduction"
    assert "provenance_steps_count" in trace.metrics
    assert "symbolic_justification" in trace.rationale
    justification = trace.rationale["symbolic_justification"]
    assert "UP" in justification["rejected_actions"]
    assert "RIGHT" in justification["rejected_actions"]


def test_logic_agent_registered_in_factory() -> None:
    registered = AgentRegistry.list_registered()
    assert "logic_agent" in registered

    agent = AgentRegistry.create(AgentConfig(agent_type="logic_agent", agent_id="la_1"))
    assert isinstance(agent, LogicAgent)

