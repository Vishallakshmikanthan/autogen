"""Independent solution verification for Constraint Satisfaction Problems (SYS-08, Module M09)."""

from __future__ import annotations

from typing import Mapping, TypeVar

from autogen.intelligence.csp.contracts import CSP

V = TypeVar("V")
D = TypeVar("D")


def verify_csp_solution(
    csp: CSP[V, D],
    assignment: Mapping[V, D],
    initial_domains: Mapping[V, list[D]] | None = None,
) -> tuple[bool, str | None]:
    """Independent oracle verifying that a CSP assignment satisfies all variables and constraints.

    Returns:
        (is_valid, error_reason)
    """
    # 1. Completeness: all variables must be assigned
    for v in csp.variables:
        if v not in assignment:
            return False, f"Variable '{v}' is not assigned in the solution."

    # 2. Domain validity: assigned values must belong to the variable's initial domain
    ref_domains = initial_domains or csp.domains
    for v in csp.variables:
        val = assignment[v]
        if val not in ref_domains[v]:
            return False, f"Value '{val}' for variable '{v}' is not in valid domain: {ref_domains[v]}"

    # 3. Constraint satisfaction: every constraint must be strictly satisfied
    for c in csp.constraints:
        if not c.is_satisfied(assignment):
            scope_str = ", ".join(f"{var}={assignment[var]}" for var in c.scope)
            return False, f"Constraint violation in scope ({scope_str}) on {c}"

    return True, None

