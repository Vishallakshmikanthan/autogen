"""Unit tests for GridSearchProblem adapter (Module M06)."""

from __future__ import annotations

from autogen.intelligence.search import GridSearchProblem
from autogen.simulation.state.types import ActionType, Position


def test_grid_search_problem_successors_and_bounds():
    """Verifies that successors stay within grid boundaries and exclude obstacles."""
    # 3x3 grid with center obstacle
    grid = [
        [1.0, 1.0, 1.0],
        [1.0, float("inf"), 1.0],  # (1, 1) is blocked
        [1.0, 1.0, 1.0],
    ]
    problem = GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(2, 2),
    )

    # From (0, 0), only (1, 0) [RIGHT] and (0, 1) [DOWN] are in-bounds
    succs = problem.successors(Position(0, 0))
    succ_actions = [s[0] for s in succs]
    succ_positions = [s[1] for s in succs]

    assert len(succs) == 2
    assert ActionType.RIGHT in succ_actions
    assert ActionType.DOWN in succ_actions
    assert Position(1, 0) in succ_positions
    assert Position(0, 1) in succ_positions

    # From (1, 0), DOWN leads to (1, 1) which is an obstacle, so it must be excluded
    succs_from_10 = problem.successors(Position(1, 0))
    pos_from_10 = [s[1] for s in succs_from_10]
    assert Position(1, 1) not in pos_from_10


def test_grid_search_problem_goal_and_heuristics():
    """Verifies goal predicate and heuristic dispatch."""
    grid = [[1.0, 1.0], [1.0, 1.0]]
    problem = GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(1, 1),
        heuristic_type="manhattan",
    )

    assert problem.is_goal(Position(1, 1)) is True
    assert problem.is_goal(Position(0, 0)) is False
    assert problem.heuristic(Position(0, 0)) == 2.0
    assert problem.heuristic(Position(1, 1)) == 0.0
