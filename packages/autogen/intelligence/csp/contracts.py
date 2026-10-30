"""Constraint Satisfaction Problem (CSP) contracts and data structures (SYS-08, Module M09)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

V = TypeVar("V")  # Variable type (hashable, immutable)
D = TypeVar("D")  # Domain value type (hashable, immutable)
D_contra = TypeVar("D_contra", contravariant=True)


@runtime_checkable
class Constraint(Protocol[V, D_contra]):
    """Pure constraint interface defining a relational restriction over a scope of variables."""

    @property
    def scope(self) -> tuple[V, ...]:
        """Variables constrained by this relation."""
        ...

    def is_satisfied(self, assignment: Mapping[V, D_contra]) -> bool:
        """Returns True if the partial or complete assignment satisfies this constraint."""
        ...


@dataclass(frozen=True, slots=True)
class BinaryConstraint(Generic[V, D]):
    """Binary constraint between two variables defined by a predicate function."""

    var1: V
    var2: V
    predicate: Callable[[D, D], bool]
    name: str = "BinaryConstraint"

    @property
    def scope(self) -> tuple[V, ...]:
        return (self.var1, self.var2)

    def is_satisfied(self, assignment: Mapping[V, D]) -> bool:
        if self.var1 not in assignment or self.var2 not in assignment:
            return True  # Vacuously satisfied for unassigned variables
        val1 = assignment[self.var1]
        val2 = assignment[self.var2]
        return self.predicate(val1, val2)


@dataclass(frozen=True, slots=True)
class AllDifferentConstraint(Generic[V, D]):
    """Global constraint requiring all variables in scope to have mutually distinct values."""

    variables: tuple[V, ...]

    @property
    def scope(self) -> tuple[V, ...]:
        return self.variables

    def is_satisfied(self, assignment: Mapping[V, D]) -> bool:
        seen: set[D] = set()
        for v in self.variables:
            if v in assignment:
                val = assignment[v]
                if val in seen:
                    return False
                seen.add(val)
        return True


@dataclass(frozen=True, slots=True)
class CSPStepEvent(Generic[V, D]):
    """Step event emitted during CSP search for visualization and algorithm duel tracing."""

    step: int
    variable: V
    value: D
    action: str  # "assign", "backtrack", "prune"
    domains_remaining: dict[V, int]
    backtracks: int


@dataclass(slots=True)
class CSP(Generic[V, D]):
    """A Constraint Satisfaction Problem defined by variables, domains, and constraints."""

    variables: tuple[V, ...]
    domains: dict[V, list[D]]
    constraints: list[Constraint[V, D]]
    neighbors: dict[V, set[V]] = field(default_factory=dict)
    constraints_by_var: dict[V, list[Constraint[V, D]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.neighbors:
            self._build_index()

    def _build_index(self) -> None:
        self.neighbors = {v: set() for v in self.variables}
        self.constraints_by_var = {v: [] for v in self.variables}

        for c in self.constraints:
            for v in c.scope:
                if v in self.constraints_by_var:
                    self.constraints_by_var[v].append(c)
                for other in c.scope:
                    if other != v and v in self.neighbors:
                        self.neighbors[v].add(other)

    def copy(self) -> CSP[V, D]:
        """Creates a deep copy of the domains while preserving immutable variables and constraints."""
        new_csp = CSP(
            variables=self.variables,
            domains={v: list(d) for v, d in self.domains.items()},
            constraints=list(self.constraints),
            neighbors={v: set(nbrs) for v, nbrs in self.neighbors.items()},
            constraints_by_var={v: list(clist) for v, clist in self.constraints_by_var.items()},
        )
        return new_csp

    def is_consistent(self, var: V, val: D, assignment: Mapping[V, D]) -> bool:
        """Tests whether assigning val to var violates any constraint with currently assigned variables."""
        temp_assignment = dict(assignment)
        temp_assignment[var] = val
        for c in self.constraints_by_var.get(var, []):
            if not c.is_satisfied(temp_assignment):
                return False
        return True


@dataclass(frozen=True, slots=True)
class CSPSolution(Generic[V, D]):
    """Immutable result of a CSP solver invocation."""

    is_solved: bool
    assignment: dict[V, D]
    nodes_expanded: int
    backtracks: int
    runtime_sec: float
