"""Contracts, protocols, and data structures for hybrid cognitive architectures (SYS-15, Module M17)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position


@dataclass
class MCTSNode:
    """A search tree node in Monte Carlo Tree Search with visit counts and cumulative values."""

    state_key: str
    parent: MCTSNode | None = None
    action_from_parent: ActionType | None = None
    children: dict[ActionType, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list[ActionType] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        """Mean expected value of this node: W / N."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        """True if all candidate actions have been expanded into child nodes."""
        return len(self.untried_actions) == 0

    def is_leaf(self) -> bool:
        """True if node has no expanded children."""
        return len(self.children) == 0


@runtime_checkable
class Option(Protocol):
    """An option omega = (I, pi, beta) in the Hierarchical Reinforcement Learning framework."""

    option_id: str
    description: str

    def can_initiate(self, state: dict[str, Any], observation: Observation) -> bool:
        """Initiation set predicate I(s) in {0, 1}."""
        ...

    def select_action(self, state: dict[str, Any], observation: Observation) -> ActionType:
        """Internal option subpolicy pi_omega(s) -> a."""
        ...

    def should_terminate(self, state: dict[str, Any], observation: Observation) -> bool:
        """Termination condition beta_omega(s) in {0, 1}."""
        ...


@dataclass
class ShieldViolationRecord:
    """Record of an unsafe action intercepted and vetoed by the symbolic safety shield."""

    step: int
    proposed_action: ActionType
    safe_action: ActionType
    violated_rule: str
    hazard_risk: float
    target_position: Position | None = None


@runtime_checkable
class SafetyShield(Protocol):
    """Runtime constraint verification shield providing non-bypassable safety overrides."""

    def verify_and_shield(
        self,
        step: int,
        proposed_action: ActionType,
        observation: Observation,
        hazard_probabilities: dict[Position, float],
    ) -> tuple[ActionType, ShieldViolationRecord | None]:
        """Validates proposed action against formal safety constraints and provides fallback if violated."""
        ...
