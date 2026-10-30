"""Integration tests for AlgorithmDuelRunner lockstep search execution (Module M06)."""

from __future__ import annotations

import pytest

from autogen.intelligence.search import (
    AlgorithmDuelRunner,
    AStarSearch,
    BreadthFirstSearch,
    GreedyBestFirstSearch,
    GridSearchProblem,
    SearchStepEvent,
    UniformCostSearch,
)
from autogen.simulation.state.types import ActionType, Position


@pytest.mark.integration
def test_algorithm_duel_lockstep_execution():
    """Verifies that multiple algorithms execute synchronously in lockstep, yielding step events."""
    grid = [[1.0 for _ in range(8)] for _ in range(8)]
    grid[2][2] = float("inf")
    grid[2][3] = float("inf")

    problem = GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(7, 7),
    )

    algos = {
        "BFS": BreadthFirstSearch(),
        "UCS": UniformCostSearch(),
        "Greedy": GreedyBestFirstSearch(),
        "A*": AStarSearch(),
    }

    runner: AlgorithmDuelRunner[Position, ActionType] = AlgorithmDuelRunner(algos)
    step_events_log: list[dict[str, SearchStepEvent[Position, ActionType]]] = []

    for event_tick in runner.step_duel(problem):
        step_events_log.append(event_tick)
        # Every tick contains events for the active algorithms
        assert len(event_tick) > 0

    results = runner.run_duel(problem)

    assert len(results) == 4
    for _name, res in results.items():
        assert res.success is True
        assert res.states[-1] == Position(7, 7)

    # Cost optimality check: A* and UCS must produce identical costs
    assert results["A*"].total_cost == results["UCS"].total_cost
    assert results["BFS"].total_cost == results["UCS"].total_cost
