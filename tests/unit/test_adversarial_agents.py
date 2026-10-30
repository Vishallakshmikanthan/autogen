"""Unit tests for adversarial and game-theoretic agents (Module M07)."""

from __future__ import annotations

import pytest

from autogen.agents.adversarial.predator_agent import PredatorAgent
from autogen.agents.adversarial.prey_agent import PreyAgent
from autogen.agents.adversarial.stochastic_opponent import StochasticOpponent
from autogen.agents.adversarial.strategic_forager import StrategicForager
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.simulation.state.world_state import CellState


def _make_dummy_obs(
    step: int = 1,
    agent_pos: Position = Position(5, 5),
    other_agents: list[tuple[str, Position]] | None = None,
    food_cells: list[Position] | None = None,
) -> Observation:
    cells: list[CellState] = []
    food_set = set(food_cells or [])
    for x in range(agent_pos.x - 2, agent_pos.x + 3):
        for y in range(agent_pos.y - 2, agent_pos.y + 3):
            p = Position(x, y)
            cells.append(
                CellState(
                    position=p,
                    terrain=TerrainType.GRASSLAND,
                    food=5.0 if p in food_set else 0.0,
                    movement_cost=1.0,
                )
            )

    return Observation(
        step=step,
        agent_id="test_agent",
        position=agent_pos,
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        visible_cells=tuple(cells),
        visible_agents=tuple(other_agents or []),
    )


def test_predator_agent_lookahead_and_trace() -> None:
    """Verifies that PredatorAgent observes prey, executes AlphaBeta search, and produces an authentic trace."""
    predator = PredatorAgent(agent_id="pred_1", max_lookahead_depth=3)

    # Prey positioned at (5, 7)
    obs = _make_dummy_obs(agent_pos=Position(5, 5), other_agents=[("prey_1", Position(5, 7))])

    predator.observe(obs)
    predator.update_belief()
    action, trace = predator.decide()

    assert action.action_type in [
        ActionType.UP,
        ActionType.DOWN,
        ActionType.LEFT,
        ActionType.RIGHT,
        ActionType.STAY,
    ]
    # Invariant: Predator must move downward toward prey at (5, 7)
    assert action.action_type == ActionType.DOWN

    # Verify authentic DecisionTrace
    assert trace.agent_id == "pred_1"
    assert trace.algorithm_name == "AlphaBetaSearch"
    assert "candidate_actions" in trace.rationale
    assert len(trace.rationale["candidate_actions"]) > 0
    assert "nodes_expanded" in trace.metrics
    assert trace.metrics["nodes_expanded"] > 0
    assert predator.last_search_tree is not None


def test_prey_agent_evasion_minimax_and_expectimax() -> None:
    """Verifies that PreyAgent evades observed predators using both Minimax and Expectimax."""
    prey_mm = PreyAgent(agent_id="prey_mm", max_lookahead_depth=2, use_expectimax=False)
    prey_ex = PreyAgent(agent_id="prey_ex", max_lookahead_depth=2, use_expectimax=True)

    # Predator close at (5, 4), prey at (5, 5)
    obs = _make_dummy_obs(agent_pos=Position(5, 5), other_agents=[("pred_1", Position(5, 4))])

    prey_mm.observe(obs)
    action_mm, trace_mm = prey_mm.decide()
    assert trace_mm.algorithm_name == "MinimaxSearch"
    assert len(trace_mm.rationale["candidate_actions"]) > 0

    prey_ex.observe(obs)
    action_ex, trace_ex = prey_ex.decide()
    assert trace_ex.algorithm_name == "ExpectimaxSearch"
    assert len(trace_ex.rationale["candidate_actions"]) > 0


def test_stochastic_opponent() -> None:
    """Verifies StochasticOpponent samples valid actions and populates selection probability."""
    opp = StochasticOpponent(agent_id="opp_1", seed=42)
    obs = _make_dummy_obs(agent_pos=Position(5, 5))

    opp.observe(obs)
    action, trace = opp.decide()

    assert trace.algorithm_name == "StochasticPolicy"
    assert "selection_probability" in trace.metrics
    assert 0.0 < float(trace.metrics["selection_probability"]) <= 1.0


def test_strategic_forager_hawk_dove_resolution() -> None:
    """Verifies StrategicForager initiates Hawk-Dove game upon contested encounter."""
    forager = StrategicForager(agent_id="forager_1", strategy_mode="hawk")
    obs = _make_dummy_obs(agent_pos=Position(5, 5), other_agents=[("rival_1", Position(5, 6))])

    forager.observe(obs)
    action, trace = forager.decide()

    assert trace.algorithm_name == "GameTheoreticForager"
    assert trace.metrics.get("strategic_disposition") == "Hawk"
    assert trace.metrics.get("contesting_agent_id") == "rival_1"


def test_adversarial_agents_registered_in_factory() -> None:
    """Verifies that all M07 agents are registered in AgentRegistry."""
    registered = AgentRegistry.list_registered()
    assert "predator" in registered
    assert "prey" in registered
    assert "stochastic_opponent" in registered
    assert "strategic_forager" in registered

    p_agent = AgentRegistry.create(AgentConfig(agent_type="predator", agent_id="p1"))
    assert isinstance(p_agent, PredatorAgent)

    prey_agent = AgentRegistry.create(AgentConfig(agent_type="prey", agent_id="prey1"))
    assert isinstance(prey_agent, PreyAgent)
