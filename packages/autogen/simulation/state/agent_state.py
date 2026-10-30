"""Immutable agent state container and operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.state.types import Position


class AgentState(BaseModel):
    """Immutable state snapshot for an individual agent $s_t^i$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    position: Position
    energy: float = Field(ge=0.0)
    health: float = Field(ge=0.0)
    carried_food: float = Field(default=0.0, ge=0.0)
    starvation_ticks: int = Field(default=0, ge=0)
    is_alive: bool = True
    internal_memory_ref: dict[str, Any] = Field(default_factory=dict)
