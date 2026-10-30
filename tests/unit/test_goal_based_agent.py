"""Unit tests for GoalBasedAgent: path planning, execution, cache invalidation, and replanning (Module M06)."""

from __future__ import annotations

from autogen.agents.goal_based import GoalBasedAgent
from autogen.core.types import DecisionTrace
from autogen.intelligence.search.informed import AStarSearch
from autogen.intelligence.search.uninformed import BreadthFirstSearch
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.simulation.state.world_state import CellState


def _make_dummy_observation(
    step: int,
    pos: Position,
    cells: list[CellState],
    energy: float = 100.0,
) -> Observation:
    return Observation(
        agent_id="test_goal_agent",
        step=step,
        position=pos,
        energy=energy,
        health=100.0,
        carried_food=0.0,
        is_alive=True,
        visible_cells=tuple(cells),
        visible_agents=(),
    )


def test_goal_based_agent_path_execution():
    """Agent plans path with A* to fixed goal and executes actions sequentially."""
    agent = GoalBasedAgent(
        agent_id="test_goal_agent",
        search_strategy=AStarSearch(),
        fixed_goal=Position(2, 0),
    )

    # Initial sensory observation at (0, 0)
    visible_cells = [
        CellState(
            position=Position(x, y),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        )
        for y in range(3)
        for x in range(3)
    ]
    obs_0 = _make_dummy_observation(0, Position(0, 0), visible_cells)

    agent.observe(obs_0)
    agent.update_belief()
    action_1, trace_1 = agent.decide()

    assert isinstance(trace_1, DecisionTrace)
    assert trace_1.rationale["selected_action"] == ActionType.RIGHT.value
    assert action_1.action_type == ActionType.RIGHT
    assert "nodes_expanded" in trace_1.metrics

    # Next step at (1, 0)
    obs_1 = _make_dummy_observation(1, Position(1, 0), visible_cells)
    agent.observe(obs_1)
    agent.update_belief()
    action_2, trace_2 = agent.decide()

    # Second step along cached path should also be RIGHT to reach (2, 0)
    assert action_2.action_type == ActionType.RIGHT

    # At goal (2, 0)
    obs_2 = _make_dummy_observation(2, Position(2, 0), visible_cells)
    agent.observe(obs_2)
    agent.update_belief()
    assert agent.current_goal is not None


def test_goal_based_agent_cache_invalidation_on_obstacle():
    """Agent detects unexpected obstacle on planned route and invalidates cached path to replan."""
    agent = GoalBasedAgent(
        agent_id="test_goal_agent",
        search_strategy=BreadthFirstSearch(),
        fixed_goal=Position(0, 2),
    )

    # Initially all observed cells appear walkable
    cells_step0 = [
        CellState(
            position=Position(0, 0),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        ),
        CellState(
            position=Position(0, 1),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        ),
        CellState(
            position=Position(0, 2),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        ),
        CellState(
            position=Position(1, 0),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        ),
        CellState(
            position=Position(1, 1),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        ),
        CellState(
            position=Position(1, 2),
            terrain=TerrainType.GRASSLAND,
            food=0.0,
            movement_cost=1.0,
        ),
    ]

    agent.observe(_make_dummy_observation(0, Position(0, 0), cells_step0))
    agent.update_belief()
    act, trace = agent.decide()

    # Initial plan goes DOWN to (0, 1) then DOWN to (0, 2)
    assert act.action_type == ActionType.DOWN
    assert len(agent._cached_path) == 1  # 1 remaining step (DOWN)

    # Now simulate new observation revealing that (0, 1) became blocked (e.g. wall/obstacle)
    cells_obstacle = [
        CellState(
            position=Position(0, 1),
            terrain=TerrainType.OBSTACLE,
            food=0.0,
            movement_cost=1.0,
        ),
    ]
    agent.observe(_make_dummy_observation(1, Position(0, 0), cells_obstacle))

    # The cached path toward (0, 1) should be invalidated immediately!
    assert len(agent._cached_path) == 0

    # Next decision must replan a detour via (1, 0)
    agent.update_belief()
    act_replan, trace_replan = agent.decide()
    assert act_replan.action_type == ActionType.RIGHT
