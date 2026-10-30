"""Adversarial game search contracts, protocols, and data structures (SYS-10, Module M07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Protocol, Sequence, TypeVar

S = TypeVar("S", bound=Hashable)  # State type (immutable, hashable)
S_contra = TypeVar("S_contra", bound=Hashable, contravariant=True)
A = TypeVar("A", bound=Hashable)  # Action type


class EvaluationFunction(Protocol[S_contra]):
    """Static heuristic evaluation function: h(s, player) -> float."""

    def __call__(self, state: S_contra, player: int) -> float: ...


class MoveOrderingFunction(Protocol[S_contra, A]):
    """Move ordering heuristic prioritising candidate moves to maximise alpha-beta cutoffs."""

    def __call__(self, actions: Sequence[A], state: S_contra, player: int) -> list[A]: ...


class AdversarialGame(Protocol[S, A]):
    """Sequential turn-based game protocol for adversarial search."""

    @property
    def initial_state(self) -> S:
        """Initial state of the game."""
        ...

    def player(self, state: S) -> int:
        """Active player index: 0 for MAX, 1 for MIN (or other players), -1 for CHANCE."""
        ...

    def legal_actions(self, state: S) -> list[A]:
        """List of legal actions available from state s."""
        ...

    def successor(self, state: S, action: A) -> S:
        """Deterministic transition s' = T(s, a)."""
        ...

    def is_terminal(self, state: S) -> bool:
        """Terminal test predicate."""
        ...

    def utility(self, state: S, player: int) -> float:
        """Terminal utility value for the specified player."""
        ...

    def chance_outcomes(self, state: S, action: A) -> list[tuple[S, float]]:
        """Stochastic transition for Expectimax: list of (next_state, probability)."""
        ...


class AdversarialAlgorithm(Protocol[S, A]):
    """Deliberation algorithm contract for adversarial search."""

    def search(
        self,
        game: AdversarialGame[S, A],
        state: S,
        max_depth: int,
        eval_fn: EvaluationFunction[S],
    ) -> AdversarialSearchResult[A]: ...



@dataclass(slots=True)
class GameTreeNode:
    """Traceable lookahead search tree node for visualization, debugging, and inspection."""

    node_id: str
    player: int
    depth: int
    value: float
    best_action: Any | None = None
    alpha: float | None = None
    beta: float | None = None
    pruned: bool = False
    state_repr: str = ""
    children: list[GameTreeNode] = field(default_factory=list)


@dataclass(slots=True)
class AdversarialStepEvent(Generic[S, A]):
    """Trace event emitted during tree expansion for animation / step tracing."""

    step: int
    current_state: S
    player: int
    depth: int
    alpha: float
    beta: float
    evaluated_value: float
    nodes_expanded: int
    pruned: bool = False


@dataclass(slots=True)
class AdversarialSearchResult(Generic[A]):
    """Deliberation result produced by an adversarial search algorithm."""

    best_action: A | None
    expected_value: float
    nodes_expanded: int
    nodes_pruned: int
    max_depth_reached: int
    tree_root: GameTreeNode | None = None
    runtime_seconds: float = 0.0
