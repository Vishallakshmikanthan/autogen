"""Search problem contracts, protocols, and data structures (SYS-06, Module M06)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

S = TypeVar("S")  # State type (hashable, immutable)
S_contra = TypeVar("S_contra", contravariant=True)
A = TypeVar("A")  # Action type


class HeuristicFn(Protocol[S_contra]):
    """Callable protocol estimating cost from state to goal h(s) -> [0, inf)."""

    def __call__(self, state: S_contra) -> float: ...


class SearchProblem(Protocol[S, A]):
    """Pure generic protocol defining a state-space search problem."""

    @property
    def initial_state(self) -> S:
        """Initial state s_0 in the search space."""
        ...

    def is_goal(self, state: S) -> bool:
        """Goal predicate testing whether state s in Goal."""
        ...

    def successors(self, state: S) -> list[tuple[A, S, float]]:
        """Transition function yielding (action, next_state, step_cost)."""
        ...

    def heuristic(self, state: S) -> float:
        """Heuristic evaluation h(s) estimating remaining cost to closest goal."""
        ...


@dataclass(slots=True)
class SearchNode(Generic[S, A]):
    """Node in the search tree tracking path lineage and cumulative costs."""

    state: S
    parent: SearchNode[S, A] | None = None
    action: A | None = None
    g_cost: float = 0.0
    h_cost: float = 0.0
    depth: int = 0

    @property
    def f_cost(self) -> float:
        """Total evaluation cost f(n) = g(n) + h(n)."""
        return self.g_cost + self.h_cost


@dataclass(slots=True)
class SearchStepEvent(Generic[S, A]):
    """Traceable step event emitted by search algorithms for animation and duels."""

    algorithm: str
    step: int
    current_state: S
    frontier_states: list[S] = field(default_factory=list)
    explored_states: list[S] = field(default_factory=list)
    f_cost: float = 0.0
    g_cost: float = 0.0
    nodes_expanded: int = 0


@dataclass(slots=True)
class SearchResult(Generic[S, A]):
    """Immutable result of a search execution with telemetry metrics."""

    success: bool
    path: list[A]
    states: list[S]
    total_cost: float
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int
    runtime_seconds: float


def reconstruct_path(goal_node: SearchNode[S, A]) -> tuple[list[A], list[S]]:
    """Reconstructs the sequence of actions and states from initial node to goal node."""
    actions: list[A] = []
    states: list[S] = []

    curr: SearchNode[S, A] | None = goal_node
    while curr is not None:
        states.append(curr.state)
        if curr.action is not None:
            actions.append(curr.action)
        curr = curr.parent

    actions.reverse()
    states.reverse()
    return actions, states
