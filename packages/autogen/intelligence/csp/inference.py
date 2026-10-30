"""Inference algorithms for CSP: AC-3 (Arc Consistency) and Forward Checking (SYS-08, Module M09)."""

from __future__ import annotations

from collections import deque
from typing import Mapping, TypeVar

from autogen.intelligence.csp.contracts import CSP

V = TypeVar("V")
D = TypeVar("D")


def ac3(
    csp: CSP[V, D],
    queue: deque[tuple[V, V]] | None = None,
) -> tuple[bool, dict[V, list[D]]]:
    """Arc Consistency Algorithm 3 (AC-3).

    Enforces arc consistency across all binary constraints in the CSP.
    Returns:
        (is_consistent, pruned_values_by_var) where pruned_values_by_var tracks
        which domain values were removed so they can be restored upon backtracking.
    """
    if queue is None:
        queue = deque()
        for xi in csp.variables:
            for xj in csp.neighbors.get(xi, set()):
                queue.append((xi, xj))

    pruned: dict[V, list[D]] = {v: [] for v in csp.variables}

    while queue:
        xi, xj = queue.popleft()
        revised, removed_vals = _revise(csp, xi, xj)
        if revised:
            pruned[xi].extend(removed_vals)
            if len(csp.domains[xi]) == 0:
                return False, pruned  # Domain wipeout -> inconsistency detected
            for xk in csp.neighbors.get(xi, set()):
                if xk != xj:
                    queue.append((xk, xi))

    return True, pruned


def _revise(csp: CSP[V, D], xi: V, xj: V) -> tuple[bool, list[D]]:
    """Revises domain of xi with respect to xj.

    Removes any value x in Domain(xi) for which there is no supporting value y in Domain(xj)
    that satisfies all constraints between xi and xj.
    """
    revised = False
    removed: list[D] = []
    constraints_between = [
        c for c in csp.constraints_by_var.get(xi, []) if xj in c.scope
    ]

    new_domain: list[D] = []
    for x in csp.domains[xi]:
        # Check if there exists at least one y in Domain(xj) satisfying all constraints
        has_support = False
        for y in csp.domains[xj]:
            assignment: dict[V, D] = {xi: x, xj: y}
            if all(c.is_satisfied(assignment) for c in constraints_between):
                has_support = True
                break
        if has_support:
            new_domain.append(x)
        else:
            removed.append(x)
            revised = True

    if revised:
        csp.domains[xi] = new_domain

    return revised, removed


def forward_check(
    csp: CSP[V, D],
    var: V,
    value: D,
    assignment: Mapping[V, D],
) -> dict[V, list[D]] | None:
    """Forward checking inference.

    Prunes incompatible values from domains of unassigned neighbors of `var`.
    Returns:
        A mapping of pruned values {neighbor: [removed_values]}, or None if a domain becomes empty.
    """
    pruned: dict[V, list[D]] = {}
    temp_assignment = dict(assignment)
    temp_assignment[var] = value

    for nbr in csp.neighbors.get(var, set()):
        if nbr not in assignment:
            removed: list[D] = []
            remaining: list[D] = []
            constraints_between = [
                c for c in csp.constraints_by_var.get(var, []) if nbr in c.scope
            ]

            for val_nbr in csp.domains[nbr]:
                temp_assignment[nbr] = val_nbr
                if all(c.is_satisfied(temp_assignment) for c in constraints_between):
                    remaining.append(val_nbr)
                else:
                    removed.append(val_nbr)

            del temp_assignment[nbr]

            if removed:
                pruned[nbr] = removed
                csp.domains[nbr] = remaining

            if len(csp.domains[nbr]) == 0:
                # Domain wipeout! Restore pruned domains and return conflict
                for p_var, p_vals in pruned.items():
                    csp.domains[p_var].extend(p_vals)
                return None

    return pruned
