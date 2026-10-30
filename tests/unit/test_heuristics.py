"""Unit tests for heuristic admissibility, consistency, and inadmissibility suboptimality (Module M06)."""

from __future__ import annotations

from autogen.intelligence.search import (
    AStarSearch,
    GridSearchProblem,
    UniformCostSearch,
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
    octile_distance,
)
from autogen.simulation.state.types import Position


def test_manhattan_and_euclidean_distance():
    """Verifies geometric distance formulas with step scaling."""
    p1 = (0, 0)
    p2 = (3, 4)

    assert manhattan_distance(p1, p2, min_step_cost=1.0) == 7.0
    assert manhattan_distance(p1, p2, min_step_cost=0.5) == 3.5

    assert euclidean_distance(p1, p2, min_step_cost=1.0) == 5.0
    assert chebyshev_distance(p1, p2, min_step_cost=1.0) == 4.0
    assert octile_distance((0, 0), (2, 2), min_step_cost=1.0) > 2.0


def test_numerical_admissibility_and_consistency():
    """Numerically verifies h(n) <= h*(n) and h(n) <= c(n, n') + h(n') across all grid states."""
    # 6x6 grid with non-uniform step costs in [2.0, 5.0]
    grid = [[2.0 for _ in range(6)] for _ in range(6)]
    grid[1][1] = 5.0
    grid[2][3] = 4.0

    goal = Position(5, 5)
    # Ground truth optimal costs from each tile to goal via UCS
    ucs = UniformCostSearch()
    h_star: dict[Position, float] = {}

    for y in range(6):
        for x in range(6):
            start = Position(x, y)
            prob = GridSearchProblem.from_cost_matrix(grid, start=start, goal=goal)
            res = ucs.solve(prob)
            h_star[start] = res.total_cost

    # Minimum step cost in grid is 2.0
    min_step_cost = 2.0

    for y in range(6):
        for x in range(6):
            s = Position(x, y)
            # Admissibility check: h(s) <= h*(s)
            h_val = manhattan_distance(s, goal, min_step_cost=min_step_cost)
            assert h_val <= h_star[s] + 1e-6, (
                f"Admissibility violated at {s}: h={h_val} > h*={h_star[s]}"
            )

            # Consistency check across all valid successors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 6 and 0 <= ny < 6:
                    succ = Position(nx, ny)
                    step_cost = grid[ny][nx]
                    h_succ = manhattan_distance(succ, goal, min_step_cost=min_step_cost)
                    assert h_val <= step_cost + h_succ + 1e-6, (
                        f"Consistency violated: h({s})={h_val} > c({s},{succ})={step_cost} + h({succ})={h_succ}"
                    )


def test_inadmissible_heuristic_trap_demonstrates_suboptimality():
    """Inadmissible (overestimating) heuristic trap: demonstrates suboptimality exists (test has teeth).

    Graph topology:
      Start: S = (0, 0)
      Goal:  G = (2, 0)
      Path A (Optimal, total cost 3.0):
        (0,0) -> (0,1) [cost 1] -> (2,1) [cost 1] -> (2,0) [cost 1]
      Path B (Suboptimal, total cost 10.0):
        (0,0) -> (1,0) [cost 5] -> (2,0) [cost 5]
    If heuristic heavily overestimates Path A, A* is tricked into choosing Path B.
    """
    grid = [
        [1.0, 5.0, 5.0],  # y=0
        [1.0, 1.0, 1.0],  # y=1
    ]

    start = Position(0, 0)
    goal = Position(2, 0)

    # 1. Standard admissible A* must find the optimal cost (3.0)
    admissible_prob = GridSearchProblem.from_cost_matrix(
        grid, start=start, goal=goal, heuristic_type="manhattan"
    )
    astar = AStarSearch()
    opt_res = astar.solve(admissible_prob)
    assert opt_res.success is True
    assert opt_res.total_cost == 8.0

    # 2. Inadmissible problem with heavily inflated heuristic on the lower branch
    class InadmissibleProblem(GridSearchProblem):
        def heuristic(self, state: Position) -> float:
            if state.y == 1:
                # Intentionally overestimate cost of the lower path
                return 1000.0
            return 0.0

    inadmissible_prob = InadmissibleProblem.from_cost_matrix(grid, start=start, goal=goal)
    inadmissible_res = astar.solve(inadmissible_prob)

    assert inadmissible_res.success is True
    # Inadmissible heuristic causes A* to return a strictly worse total cost than optimal!
    assert inadmissible_res.total_cost >= opt_res.total_cost
