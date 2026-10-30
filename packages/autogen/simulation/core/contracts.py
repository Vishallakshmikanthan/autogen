"""Core simulation contracts, action models, observations, and events."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from autogen.core.types import DecisionTrace
from autogen.simulation.state.types import ActionType, Position
from autogen.simulation.state.world_state import CellState, WorldState


class Action(BaseModel):
    """Command pattern model representing an agent's intended action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    action_type: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Sensory observation granted to an agent at discrete time step $t$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    step: int = Field(ge=0)
    position: Position
    energy: float = Field(ge=0.0)
    health: float = Field(ge=0.0)
    carried_food: float = Field(ge=0.0)
    is_alive: bool = True
    visible_cells: tuple[CellState, ...]
    visible_agents: tuple[tuple[str, Position], ...]


class SimulationEvent(BaseModel):
    """Base model for typed events dispatched across the event bus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_ns: int = Field(default_factory=time.time_ns)
    step: int = Field(ge=0)
    event_type: str = "simulation_event"


class InvalidActionEvent(SimulationEvent):
    """Emitted when an agent proposes an illegal action that is rejected/substituted."""

    event_type: str = "invalid_action"
    agent_id: str
    attempted_action: Action
    substituted_action: Action
    reason: str


class StepCompletedEvent(SimulationEvent):
    """Emitted at phase 8 of each simulation tick."""

    event_type: str = "step_completed"
    state_hash: str
    active_agents_count: int


class DecisionTraceEmittedEvent(SimulationEvent):
    """Emitted when an agent produces a verified decision trace."""

    event_type: str = "decision_trace_emitted"
    agent_id: str
    algorithm_name: str
    decision_type: str
    metrics: dict[str, float | int | str]


class ConflictPolicy(str, Enum):
    """Policy for arbitrating contested movement into the same cell."""

    CANCEL_ALL = "cancel_all"  # All colliding agents remain in place
    SEEDED_PRIORITY = "seeded_priority"  # Winner selected via seeded permutation


class CollectPolicy(str, Enum):
    """Policy for arbitrating simultaneous food collection on the same cell."""

    PROPORTIONAL = "proportional"  # Available food is split evenly
    SEEDED_PRIORITY = "seeded_priority"  # Winner receives full share, others remaining


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol satisfied by any agent interacting with SimulationEngine."""

    agent_id: str

    def observe(self, observation: Observation) -> None: ...

    def update_belief(self) -> None: ...

    def decide(self) -> tuple[Action, DecisionTrace]: ...

    def learn(self, reward: float, terminal: bool) -> None: ...

    def reset(self) -> None: ...


@runtime_checkable
class BaseEnvironment(Protocol):
    """Formal protocol required of any environment pluggable into the SimulationEngine."""

    @property
    def state(self) -> WorldState: ...

    def reset(self, seed: int | None = None) -> WorldState: ...

    def restore_state(self, state: WorldState) -> None: ...

    def get_observation(self, agent_id: str) -> Observation: ...

    def validate_action(self, agent_id: str, action: ActionType) -> bool: ...

    def apply_transitions(self, actions: dict[str, ActionType]) -> WorldState: ...
