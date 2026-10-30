"""Unit tests for informed heuristic search algorithms (Greedy, A*, Weighted A*, IDA*) (Module M06)."""

from __future__ import annotations

from autogen.intelligence.search import (
    AStarSearch,
    GreedyBestFirstSearch,
    GridSearchProblem,
    IterativeDeepeningAStar,
    UniformCostSearch,
    WeightedAStarSearch,
)
from autogen.simulation.state.types import Position


def _make_weighted_grid() -> GridSearchProblem:
    # 5x5 grid with high-cost center tile
    grid = [[1.0 for _ in range(5)] for _ in range(5)]
    grid[2][2] = 50.0
    return GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(4, 4),
        heuristic_type="manhattan",
    )


def test_astar_cost_optimality_and_efficiency():
    """A* finds the exact optimal cost while expanding fewer nodes than UCS."""
    problem = _make_weighted_grid()

    ucs = UniformCostSearch()
    res_ucs = ucs.solve(problem)

    astar = AStarSearch()
    res_astar = astar.solve(problem)

    assert res_astar.success is True
    # Cost optimality: A* cost must match UCS exactly
    assert res_astar.total_cost == res_ucs.total_cost
    # Heuristic pruning efficiency: A* expands <= UCS nodes
    assert res_astar.nodes_expanded <= res_ucs.nodes_expanded


def test_greedy_best_first_speed():
    """Greedy Best-First reaches goal rapidly with minimal node expansions."""
    problem = _make_weighted_grid()
    greedy = GreedyBestFirstSearch()
    res = greedy.solve(problem)

    assert res.success is True
    assert res.nodes_expanded <= 15
    assert res.states[-1] == Position(4, 4)


def test_weighted_astar_suboptimality_bound():
    """Weighted A* cost satisfies C* <= C_weighted <= w * C*."""
    problem = _make_weighted_grid()

    astar = AStarSearch()
    c_star = astar.solve(problem).total_cost

    for w in [1.2, 1.5, 2.0]:
        weighted_astar = WeightedAStarSearch(weight=w)
        res_w = weighted_astar.solve(problem)

        assert res_w.success is True
        # Theoretical bound check
        assert c_star <= res_w.total_cost <= w * c_star + 1e-6


def test_ida_star_optimality():
    """IDA* returns the optimal cost with linear memory depth."""
    problem = _make_weighted_grid()
    ida = IterativeDeepeningAStar(max_iterations=100)
    res = ida.solve(problem)

    astar = AStarSearch()
    res_astar = astar.solve(problem)

    assert res.success is True
    assert res.total_cost == res_astar.total_cost
    assert res.max_frontier_size <= 15  # Depth bound << full grid
