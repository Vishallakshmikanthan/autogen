"""Unit tests for benchmark CSP problems: Map Coloring, N-Queens, Sudoku, Patrol Scheduling."""

from __future__ import annotations

import pytest

from autogen.intelligence.csp.problems import (
    australia_map_coloring,
    map_coloring_problem,
    multi_agent_schedule_csp,
    n_queens_problem,
    sudoku_problem,
)
from autogen.intelligence.csp.solver import backtracking_search
from autogen.intelligence.csp.verifier import verify_csp_solution


def test_australia_map_coloring() -> None:
    """Verifies Australia 7-region map coloring with 3 colors."""
    csp = australia_map_coloring()
    solution = backtracking_search(csp, inference="forward_checking")
    assert solution.is_solved is True
    valid, err = verify_csp_solution(csp, solution.assignment)
    assert valid is True, f"Australia map coloring solution failed: {err}"
    assert len(solution.assignment) == 7


def test_n_queens_various_sizes() -> None:
    """Verifies N-Queens solver across multiple board sizes: 4, 6, 8."""
    for n in (4, 6, 8):
        csp = n_queens_problem(n)
        solution = backtracking_search(csp, inference="forward_checking")
        assert solution.is_solved is True
        valid, err = verify_csp_solution(csp, solution.assignment)
        assert valid is True, f"{n}-Queens solution invalid: {err}"


def test_sudoku_solver() -> None:
    """Verifies Sudoku CSP on a standard 9x9 puzzle."""
    # A standard valid easy puzzle
    puzzle = (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    )
    csp = sudoku_problem(puzzle)
    solution = backtracking_search(csp, inference="forward_checking")
    assert solution.is_solved is True
    valid, err = verify_csp_solution(csp, solution.assignment)
    assert valid is True, f"Sudoku solution invalid: {err}"
    assert len(solution.assignment) == 81


def test_multi_agent_patrol_schedule_assignment() -> None:
    """Verifies multi-agent schedule assignment with conflict prevention."""
    agents = ("Agent1", "Agent2")
    tasks = ("PatrolNorth", "PatrolSouth", "ForageWest")
    time_slots = (1, 2)
    # PatrolNorth and PatrolSouth cannot occur in the same time slot
    conflicts = [("PatrolNorth", "PatrolSouth")]

    csp = multi_agent_schedule_csp(agents, tasks, time_slots, conflicts)
    solution = backtracking_search(csp, inference="forward_checking")
    assert solution.is_solved is True
    valid, err = verify_csp_solution(csp, solution.assignment)
    assert valid is True, f"Schedule assignment invalid: {err}"

    # Verify no agent has two tasks in the same slot
    assignments_by_agent_slot: set[tuple[str, int]] = set()
    for task, (ag, slot) in solution.assignment.items():
        assert (ag, slot) not in assignments_by_agent_slot
        assignments_by_agent_slot.add((ag, slot))

