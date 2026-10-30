"""Variable and value ordering heuristics for CSP (SYS-08, Module M09)."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, TypeVar

from autogen.intelligence.csp.contracts import CSP

V = TypeVar("V")
D = TypeVar("D")


def select_unassigned_variable_mrv(
    csp: CSP[V, D],
    assignment: Mapping[V, D],
) -> V:
    """Selects an unassigned variable using the Minimum Remaining Values (MRV) heuristic,

    tie-broken by the Degree Heuristic (most constraints on remaining unassigned variables).
    """
    unassigned = [v for v in csp.variables if v not in assignment]
    if not unassigned:
        raise ValueError("Cannot select unassigned variable: assignment is complete.")

    def mrv_degree_key(v: V) -> tuple[int, int]:
        # 1. MRV: fewest remaining values in domain (minimize)
        domain_size = len(csp.domains[v])
        # 2. Degree: most unassigned neighbors (maximize -> negate for min)
        unassigned_neighbors = sum(1 for nbr in csp.neighbors.get(v, set()) if nbr not in assignment)
        return (domain_size, -unassigned_neighbors)

    return min(unassigned, key=mrv_degree_key)


def select_unassigned_variable_naive(
    csp: CSP[V, D],
    assignment: Mapping[V, D],
) -> V:
    """Static chronological variable ordering (for baseline ablation experiments)."""
    for v in csp.variables:
        if v not in assignment:
            return v
    raise ValueError("Cannot select unassigned variable: assignment is complete.")


def order_domain_values_lcv(
    csp: CSP[V, D],
    var: V,
    assignment: Mapping[V, D],
) -> list[D]:
    """Orders domain values of `var` using the Least Constraining Value (LCV) heuristic.

    Prefers values that rule out the fewest choices for neighboring unassigned variables.
    """
    unassigned_neighbors = [nbr for nbr in csp.neighbors.get(var, set()) if nbr not in assignment]
    if not unassigned_neighbors:
        return list(csp.domains[var])

    constraints_with_var = csp.constraints_by_var.get(var, [])

    def count_conflicts(val: D) -> int:
        conflicts = 0
        temp_assignment = dict(assignment)
        temp_assignment[var] = val

        for nbr in unassigned_neighbors:
            constraints_between = [c for c in constraints_with_var if nbr in c.scope]
            for nbr_val in csp.domains[nbr]:
                temp_assignment[nbr] = nbr_val
                if not all(c.is_satisfied(temp_assignment) for c in constraints_between):
                    conflicts += 1
            if nbr in temp_assignment:
                del temp_assignment[nbr]

        return conflicts

    return sorted(csp.domains[var], key=count_conflicts)


def order_domain_values_naive(
    csp: CSP[V, D],
    var: V,
    assignment: Mapping[V, D],
) -> list[D]:
    """Preserves native domain ordering (for baseline ablation experiments)."""
    return list(csp.domains[var])

