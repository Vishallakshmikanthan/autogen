"""Exact full-state snapshots for time-machine rewind and forking."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.core.contracts import Action
from autogen.simulation.state.world_state import WorldState, canonical_world_hash


class SimulationSnapshot(BaseModel):
    """Immutable snapshot of the engine state at a discrete step."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    step: int = Field(ge=0)
    world_state: WorldState
    state_hash: str
    agents_backup: dict[str, Any] = Field(default_factory=dict)
    conflict_rng_state: dict[str, Any] = Field(default_factory=dict)
    recorded_actions: tuple[tuple[int, tuple[Action, ...]], ...]

    @classmethod
    def create(
        cls,
        step: int,
        world_state: WorldState,
        agents: dict[str, Any],
        conflict_rng_state: dict[str, Any],
        action_history: list[dict[str, Action]],
    ) -> SimulationSnapshot:
        actions_tuple = tuple(
            (t, tuple(actions.values())) for t, actions in enumerate(action_history)
        )
        return cls(
            step=step,
            world_state=world_state,
            state_hash=canonical_world_hash(world_state),
            agents_backup=copy.deepcopy(agents),
            conflict_rng_state=copy.deepcopy(conflict_rng_state),
            recorded_actions=actions_tuple,
        )
