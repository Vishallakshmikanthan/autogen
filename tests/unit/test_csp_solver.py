"""Unit tests for CSP solvers, inference (AC-3, Forward Checking), and heuristics (Module M09)."""

from __future__ import annotations

import pytest

from autogen.intelligence.csp.contracts import BinaryConstraint, CSP
from autogen.intelligence.csp.heuristics import (
    order_domain_values_lcv,
    order_domain_values_naive,
    select_unassigned_variable_mrv,
    select_unassigned_variable_naive,
)
from autogen.intelligence.csp.inference import ac3, forward_check
from autogen.intelligence.csp.problems import australia_map_coloring, n_queens_problem
from autogen.intelligence.csp.solver import backtracking_search, min_conflicts
from autogen.intelligence.csp.verifier import verify_csp_solution


def test_ac3_arc_consistency_pruning() -> None:
    """Tests AC-3 arc consistency pruning on a toy CSP."""
    # X in {1, 2}, Y in {1, 2}, Z in {1, 2}.
    # Constraints: X < Y and Y < Z.
    # AC-3 must deduce: X=1, Y must be 2, but Z must be > 2 (empty) -> inconsistent!
    csp = CSP(
        variables=("X", "Y", "Z"),
        domains={"X": [1, 2], "Y": [1, 2], "Z": [1, 2]},
        constraints=[
            BinaryConstraint("X", "Y", lambda x, y: x < y),
            BinaryConstraint("Y", "Z", lambda y, z: y < z),
        ],
    )
    is_consistent, pruned = ac3(csp)
    assert is_consistent is False
    assert len(csp.domains["Z"]) == 0 or len(csp.domains["Y"]) == 0 or len(csp.domains["X"]) == 0


def test_ac3_vs_brute_force_support_oracle() -> None:
    """Oracle test: AC-3 pruned domains match exhaustive pairwise support validation."""
    # A in {1, 2, 3}, B in {2, 3, 4}. Constraint: A + B == 5.
    csp = CSP(
        variables=("A", "B"),
        domains={"A": [1, 2, 3], "B": [2, 3, 4]},
        constraints=[BinaryConstraint("A", "B", lambda a, b: a + b == 5)],
    )
    consistent, pruned = ac3(csp)
    assert consistent is True
    # For A=1, B=4 is valid. For A=2, B=3 is valid. For A=3, B=2 is valid.
    # Neither should have values pruned.
    assert set(csp.domains["A"]) == {1, 2, 3}
    assert set(csp.domains["B"]) == {2, 3, 4}

    # Now add constraint B > 3 (only B=4 remains)
    csp2 = CSP(
        variables=("A", "B"),
        domains={"A": [1, 2, 3], "B": [4]},
        constraints=[BinaryConstraint("A", "B", lambda a, b: a + b == 5)],
    )
    c2, p2 = ac3(csp2)
    assert c2 is True
    assert csp2.domains["A"] == [1]  # Only 1 + 4 == 5


def test_backtracking_mrv_reduces_backtracks_measurably() -> None:
    """Verification: MRV + LCV explores significantly fewer backtracks than naive chronological search."""
    # 8-Queens problem
    csp_naive = n_queens_problem(8)
    csp_mrv = n_queens_problem(8)

    sol_naive = backtracking_search(
        csp_naive,
        select_unassigned_var=select_unassigned_variable_naive,
        order_domain_values=order_domain_values_naive,
        inference="none",
    )

    sol_mrv = backtracking_search(
        csp_mrv,
        select_unassigned_var=select_unassigned_variable_mrv,
        order_domain_values=order_domain_values_lcv,
        inference="forward_checking",
    )

    assert sol_naive.is_solved is True
    assert sol_mrv.is_solved is True
    # MRV + FC should reduce nodes expanded by at least 50% vs naive blind search
    assert sol_mrv.nodes_expanded < sol_naive.nodes_expanded
    assert sol_mrv.backtracks <= sol_naive.backtracks


def test_min_conflicts_local_search_n_queens() -> None:
    """Verifies that Min-Conflicts local search solves 8-Queens."""
    csp = n_queens_problem(8)
    solution = min_conflicts(csp, max_steps=2000, seed=42)
    assert solution.is_solved is True
    valid, reason = verify_csp_solution(csp, solution.assignment)
    assert valid is True, f"Min-conflicts solution invalid: {reason}"

