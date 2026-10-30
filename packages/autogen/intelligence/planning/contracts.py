"""Classical STRIPS planning contracts and data structures (SYS-08, Module M09)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Fact:
    """A positive relational ground predicate fact under the Closed World Assumption."""

    predicate: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(self.args)})"

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True, slots=True)
class GroundAction:
    """A deterministic ground STRIPS action with preconditions, add effects, and delete effects."""

    name: str
    preconditions: frozenset[Fact]
    add_effects: frozenset[Fact]
    delete_effects: frozenset[Fact]
    cost: float = 1.0

    def is_applicable(self, state_facts: frozenset[Fact]) -> bool:
        """Checks if all action preconditions are satisfied in the current state."""
        return self.preconditions.issubset(state_facts)

    def apply(self, state_facts: frozenset[Fact]) -> frozenset[Fact]:
        """Applies the transition: Result(S, a) = (S \\ Del(a)) U Add(a)."""
        if not self.is_applicable(state_facts):
            raise ValueError(f"Action '{self.name}' is not applicable in state: {state_facts}")
        return (state_facts - self.delete_effects) | self.add_effects

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"GroundAction({self.name}, cost={self.cost})"


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    """Classical STRIPS planning problem instance: P = (S_0, A, G)."""

    initial_facts: frozenset[Fact]
    goal_facts: frozenset[Fact]
    actions: tuple[GroundAction, ...]

    def is_goal(self, state_facts: frozenset[Fact]) -> bool:
        """Goal test: G is satisfied if G subset of S."""
        return self.goal_facts.issubset(state_facts)


@dataclass(frozen=True, slots=True)
class PlanningMetrics:
    """Instrumentation metrics for plan generation."""

    nodes_expanded: int
    nodes_generated: int
    runtime_sec: float
    max_queue_size: int = 0


@dataclass(frozen=True, slots=True)
class Plan:
    """Immutable sequence of actions solving a planning problem."""

    is_solved: bool
    actions: tuple[GroundAction, ...]
    total_cost: float
    metrics: PlanningMetrics
    is_proven_unsolvable: bool = False

    @property
    def length(self) -> int:
        return len(self.actions)
