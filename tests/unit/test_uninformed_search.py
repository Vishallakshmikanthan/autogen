"""Unit tests for uninformed search algorithms (BFS, DFS, DLS, IDS, Bi-BFS, UCS) (Module M06)."""

from __future__ import annotations

from autogen.intelligence.search import (
    BidirectionalBFS,
    BreadthFirstSearch,
    DepthFirstSearch,
    DepthLimitedSearch,
    GridSearchProblem,
    IterativeDeepeningSearch,
    UniformCostSearch,
)
from autogen.simulation.state.types import Position


def _make_simple_grid(w: int = 5, h: int = 5) -> GridSearchProblem:
    # 5x5 grid with obstacle at (1, 1) and (1, 2)
    grid = [[1.0 for _ in range(w)] for _ in range(h)]
    grid[1][1] = float("inf")
    grid[2][1] = float("inf")
    return GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(4, 4),
    )


def _make_unreachable_grid() -> GridSearchProblem:
    # 3x3 grid where goal is surrounded by obstacles
    grid = [
        [1.0, 1.0, 1.0],
        [1.0, float("inf"), float("inf")],
        [1.0, float("inf"), 1.0],  # (2, 2) is enclosed
    ]
    return GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(2, 2),
    )


def test_bfs_shortest_path():
    """BFS finds optimal path in number of steps on uniform-cost grid."""
    problem = _make_simple_grid(5, 5)
    bfs = BreadthFirstSearch()
    result = bfs.solve(problem)

    assert result.success is True
    # Distance from (0, 0) to (4, 4) Manhattan is 8 steps
    assert len(result.path) == 8
    assert result.total_cost == 8.0
    assert result.states[0] == Position(0, 0)
    assert result.states[-1] == Position(4, 4)


def test_dfs_cycle_prevention_and_unreachable():
    """DFS finds a valid path and terminates safely on unreachable graphs."""
    problem = _make_simple_grid(5, 5)
    dfs = DepthFirstSearch(max_depth=50)
    result = dfs.solve(problem)

    assert result.success is True
    assert result.states[0] == Position(0, 0)
    assert result.states[-1] == Position(4, 4)

    # Test unreachable goal
    unreach = _make_unreachable_grid()
    unreach_res = dfs.solve(unreach)
    assert unreach_res.success is False
    assert unreach_res.total_cost == float("inf")


def test_depth_limited_search_cutoff():
    """DLS cutoffs when limit < optimal depth, and succeeds when limit >= optimal depth."""
    problem = _make_simple_grid(5, 5)

    # Optimal path is 8 steps
    dls_shallow = DepthLimitedSearch(limit=6)
    res_shallow = dls_shallow.solve(problem)
    assert res_shallow.success is False
    assert dls_shallow.cutoff_occurred is True

    dls_sufficient = DepthLimitedSearch(limit=8)
    res_sufficient = dls_sufficient.solve(problem)
    assert res_sufficient.success is True
    assert len(res_sufficient.path) == 8


def test_iterative_deepening_search_optimality():
    """IDS returns the shortest path in edge count while preserving low memory footprint."""
    problem = _make_simple_grid(5, 5)
    ids = IterativeDeepeningSearch(max_depth=15)
    result = ids.solve(problem)

    assert result.success is True
    assert len(result.path) == 8
    assert result.total_cost == 8.0


def test_bidirectional_bfs_connection():
    """Bidirectional BFS successfully joins forward and backward frontiers."""
    problem = _make_simple_grid(6, 6)
    bi_bfs = BidirectionalBFS()
    result = bi_bfs.solve(problem, goal_state=Position(5, 5))

    assert result.success is True
    assert result.states[0] == Position(0, 0)
    assert result.states[-1] == Position(5, 5)
    assert len(result.path) > 0


def test_uniform_cost_search_weighted_optimality():
    """UCS explores the cheaper weighted detour over a shorter expensive path."""
    # Start: (0, 0), Goal: (0, 2)
    # Direct route through (0, 1) has high cost 100.0
    # Detour through (1, 0) -> (1, 1) -> (1, 2) has cost 1.0 each (total 4.0)
    grid = [
        [1.0, 1.0],  # y=0: (0, 0)=1, (1, 0)=1
        [100.0, 1.0],  # y=1: (0, 1)=100, (1, 1)=1
        [1.0, 1.0],  # y=2: (0, 2)=1, (1, 2)=1
    ]
    problem = GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(0, 2),
    )

    ucs = UniformCostSearch()
    result = ucs.solve(problem)

    assert result.success is True
    # Expected path: (0,0) -> (1,0) [cost 1] -> (1,1) [cost 1] -> (1,2) [cost 1] -> (0,2) [cost 1] = 4.0
    assert result.total_cost == 4.0
    assert Position(0, 1) not in result.states

