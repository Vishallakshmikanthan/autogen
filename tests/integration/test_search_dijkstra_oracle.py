"""Mathematical verification of scratch search algorithms against independent test oracle scipy.sparse.csgraph.dijkstra (Module M06).

Invariant to prove:
On arbitrary weighted grids, UniformCostSearch and A* find the mathematically optimal path cost
identical to SciPy Dijkstra across 500 randomized grid topographies.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from autogen.intelligence.search import AStarSearch, GridSearchProblem, UniformCostSearch
from autogen.simulation.state.types import ActionType, Position


def _build_scipy_adjacency(grid_matrix: list[list[float]], w: int, h: int) -> csr_matrix:
    """Constructs a sparse adjacency matrix from 2D grid costs for SciPy Dijkstra."""
    n_nodes = w * h
    rows = []
    cols = []
    weights = []

    for y in range(h):
        for x in range(w):
            u = y * w + x
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    step_cost = grid_matrix[ny][nx]
                    if step_cost != float("inf"):
                        v = ny * w + nx
                        rows.append(u)
                        cols.append(v)
                        weights.append(step_cost)

    adj = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    return adj


@pytest.mark.integration
def test_500_random_grids_ucs_and_astar_equal_scipy_dijkstra():
    """Mathematical Invariant: UCS and A* costs equal scipy.sparse.csgraph.dijkstra across 500 random grids."""
    rng = np.random.default_rng(2026)
    ucs = UniformCostSearch[Position, ActionType]()
    astar = AStarSearch[Position, ActionType]()

    num_trials = 500
    matched_count = 0
    unreachable_count = 0

    for trial in range(num_trials):
        # Vary grid sizes from 4x4 up to 12x12
        w = int(rng.integers(4, 12))
        h = int(rng.integers(4, 12))

        # Generate non-uniform step costs in [1.0, 5.0]
        grid_matrix: list[list[float]] = []
        for _y in range(h):
            row = []
            for _x in range(w):
                # 20% probability of obstacle
                if rng.random() < 0.20:
                    row.append(float("inf"))
                else:
                    # Random terrain step cost
                    cost = float(rng.uniform(1.0, 5.0))
                    row.append(cost)
            grid_matrix.append(row)

        # Select random start and goal
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        gx, gy = int(rng.integers(0, w)), int(rng.integers(0, h))
        grid_matrix[sy][sx] = 1.0
        grid_matrix[gy][gx] = 1.0

        start = Position(sx, sy)
        goal = Position(gx, gy)

        # 1. Independent Oracle: SciPy Dijkstra
        adj = _build_scipy_adjacency(grid_matrix, w, h)
        start_idx = sy * w + sx
        goal_idx = gy * w + gx

        dist_matrix = dijkstra(csgraph=adj, directed=True, indices=start_idx)
        oracle_cost = float(dist_matrix[goal_idx])

        # 2. Scratch Problem & Implementations
        problem = GridSearchProblem.from_cost_matrix(grid_matrix, start=start, goal=goal)

        res_ucs = ucs.solve(problem)
        res_astar = astar.solve(problem)

        if np.isinf(oracle_cost):
            # Target unreachable according to SciPy
            assert res_ucs.success is False
            assert res_astar.success is False
            assert np.isinf(res_ucs.total_cost)
            assert np.isinf(res_astar.total_cost)
            unreachable_count += 1
        else:
            # Target reachable: assert bitwise numeric tolerance <= 1e-6
            assert res_ucs.success is True, f"UCS failed on reachable grid trial {trial}"
            assert res_astar.success is True, f"A* failed on reachable grid trial {trial}"

            assert abs(res_ucs.total_cost - oracle_cost) < 1e-6, (
                f"Trial {trial}: UCS cost {res_ucs.total_cost} != Oracle {oracle_cost}"
            )

            assert abs(res_astar.total_cost - oracle_cost) < 1e-6, (
                f"Trial {trial}: A* cost {res_astar.total_cost} != Oracle {oracle_cost}"
            )

            # Invariant: A* never expands more nodes than UCS
            assert res_astar.nodes_expanded <= res_ucs.nodes_expanded

            matched_count += 1

    print(
        f"\n[Oracle Verification Passed] 500 trials: {matched_count} optimal matches, {unreachable_count} unreachable"
    )
    assert matched_count > 300
