"""Simulation session control and status schemas (M19, SYS-18)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.experiments.config import AgentSpec


class CreateSimulationRequest(BaseModel):
    """Payload to spawn an interactive simulation session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    seed: int = Field(ge=0, default=42)
    env_id: str | None = None
    width: int = Field(ge=4, le=500, default=20)
    height: int = Field(ge=4, le=500, default=20)
    obstacle_density: float = Field(ge=0.0, le=0.8, default=0.15)
    max_steps: int = Field(ge=1, le=10000, default=200)
    agents: list[AgentSpec] = Field(default_factory=list)


class SimulationControlRequest(BaseModel):
    """Payload to dispatch execution commands to the simulation worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str = Field(pattern="^(START|PAUSE|STEP|RESET|STOP|SET_SPEED)$")
    params: dict[str, Any] = Field(default_factory=dict)


class SimulationStatusResponse(BaseModel):
    """Summary of simulation status and step progress."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    status: str
    current_step: int
    max_steps: int
    seed: int
    state_hash: str
    env_version_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SimulationStateResponse(BaseModel):
    """Full snapshot of simulation state for inspection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sim_id: str
    step: int
    state_hash: str
    snapshot: dict[str, Any]
