"""Backtracking and Local Search CSP Solvers (SYS-08, Module M09)."""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Generator
from typing import Any, Callable, Mapping, Sequence, TypeVar

from autogen.intelligence.csp.contracts import CSP, CSPSolution, CSPStepEvent
from autogen.intelligence.csp.heuristics import (
    order_domain_values_lcv,
    order_domain_values_naive,
    select_unassigned_variable_mrv,
    select_unassigned_variable_naive,
)
from autogen.intelligence.csp.inference import ac3, forward_check

V = TypeVar("V")
D = TypeVar("D")


def backtracking_search(
    csp: CSP[V, D],
    select_unassigned_var: Callable[[CSP[V, D], Mapping[V, D]], V] | None = None,
    order_domain_values: Callable[[CSP[V, D], V, Mapping[V, D]], list[D]] | None = None,
    inference: str = "forward_checking",  # "forward_checking", "ac3", or "none"
    step_callback: Callable[[CSPStepEvent[V, D]], None] | None = None,
) -> CSPSolution[V, D]:
    """Solves a CSP using recursive backtracking search with pluggable heuristics and inference.

    Args:
        csp: The constraint satisfaction problem instance.
        select_unassigned_var: Variable selection heuristic (defaults to MRV + Degree).
        order_domain_values: Value ordering heuristic (defaults to LCV).
        inference: Constraint propagation strategy ("forward_checking", "ac3", or "none").
        step_callback: Optional observer callback emitting search events.
    """
    t0 = time.perf_counter()
    var_selector = select_unassigned_var or select_unassigned_variable_mrv
    val_orderer = order_domain_values or order_domain_values_lcv

    working_csp = csp.copy()
    nodes_expanded = 0
    backtracks = 0
    step_counter = 0

    def backtrack(assignment: dict[V, D]) -> dict[V, D] | None:
        nonlocal nodes_expanded, backtracks, step_counter

        if len(assignment) == len(working_csp.variables):
            return assignment

        var = var_selector(working_csp, assignment)
        nodes_expanded += 1

        for val in val_orderer(working_csp, var, assignment):
            if working_csp.is_consistent(var, val, assignment):
                assignment[var] = val
                step_counter += 1

                if step_callback is not None:
                    step_callback(
                        CSPStepEvent(
                            step=step_counter,
                            variable=var,
                            value=val,
                            action="assign",
                            domains_remaining={v: len(working_csp.domains[v]) for v in working_csp.variables},
                            backtracks=backtracks,
                        )
                    )

                # Inference phase
                inferences_valid = True
                pruned_domains: dict[V, list[D]] = {}

                if inference == "forward_checking":
                    fc_pruned = forward_check(working_csp, var, val, assignment)
                    if fc_pruned is None:
                        inferences_valid = False
                    else:
                        pruned_domains = fc_pruned
                elif inference == "ac3":
                    # Initialize queue with arcs from unassigned neighbors to var
                    init_queue = deque((nbr, var) for nbr in working_csp.neighbors.get(var, set()) if nbr not in assignment)
                    ac3_valid, ac3_pruned = ac3(working_csp, init_queue)
                    if not ac3_valid:
                        inferences_valid = False
                    pruned_domains = ac3_pruned

                if inferences_valid:
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                # Backtrack / Undo assignment and inferences
                del assignment[var]
                backtracks += 1
                for p_var, p_vals in pruned_domains.items():
                    working_csp.domains[p_var].extend(p_vals)

                if step_callback is not None:
                    step_callback(
                        CSPStepEvent(
                            step=step_counter,
                            variable=var,
                            value=val,
                            action="backtrack",
                            domains_remaining={v: len(working_csp.domains[v]) for v in working_csp.variables},
                            backtracks=backtracks,
                        )
                    )

        return None

    solution_dict = backtrack({})
    t1 = time.perf_counter()

    return CSPSolution(
        is_solved=solution_dict is not None,
        assignment=solution_dict or {},
        nodes_expanded=nodes_expanded,
        backtracks=backtracks,
        runtime_sec=t1 - t0,
    )


def min_conflicts(
    csp: CSP[V, D],
    max_steps: int = 1000,
    seed: int | None = None,
) -> CSPSolution[V, D]:
    """Min-Conflicts local search heuristic algorithm for large CSPs.

    Starts with a complete initial assignment and iteratively minimizes constraint violations.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)

    # 1. Generate initial random complete assignment
    current: dict[V, D] = {v: rng.choice(csp.domains[v]) for v in csp.variables}
    steps_since_improvement = 0
    best_conflict_count = len(csp.variables) * len(csp.constraints)

    for step in range(max_steps):
        # 2. Check if current assignment is conflict-free
        conflicted_vars: list[V] = []
        for var in csp.variables:
            for c in csp.constraints_by_var.get(var, []):
                if not c.is_satisfied(current):
                    conflicted_vars.append(var)
                    break

        if not conflicted_vars:
            t1 = time.perf_counter()
            return CSPSolution(
                is_solved=True,
                assignment=current,
                nodes_expanded=step,
                backtracks=0,
                runtime_sec=t1 - t0,
            )

        num_conflicts = len(conflicted_vars)
        if num_conflicts < best_conflict_count:
            best_conflict_count = num_conflicts
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        # Plateau breakout: if stuck on local minimum/plateau for 100 steps, random restart
        if steps_since_improvement > 100:
            current = {v: rng.choice(csp.domains[v]) for v in csp.variables}
            steps_since_improvement = 0
            best_conflict_count = len(csp.variables) * len(csp.constraints)
            continue

        # 3. Choose a random conflicted variable
        var_to_change = rng.choice(conflicted_vars)

        # 4. Choose value that minimizes conflicts (break ties randomly)
        val_conflicts: list[tuple[D, int]] = []
        for val in csp.domains[var_to_change]:
            current[var_to_change] = val
            v_conflicts = sum(
                1 for c in csp.constraints_by_var.get(var_to_change, []) if not c.is_satisfied(current)
            )
            val_conflicts.append((val, v_conflicts))

        min_c = min(c for _, c in val_conflicts)
        best_vals = [val for val, c in val_conflicts if c == min_c]
        current[var_to_change] = rng.choice(best_vals)

    t1 = time.perf_counter()
    return CSPSolution(
        is_solved=False,
        assignment=current,
        nodes_expanded=max_steps,
        backtracks=0,
        runtime_sec=t1 - t0,
    )
