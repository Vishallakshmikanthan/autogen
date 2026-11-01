"""Environment REST request and response schemas (M19, SYS-18)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateEnvironmentRequest(BaseModel):
    """Payload to register a new simulation environment definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    width: int = Field(ge=4, le=1024, default=20)
    height: int = Field(ge=4, le=1024, default=20)
    obstacle_density: float = Field(ge=0.0, le=0.8, default=0.15)
    resource_density: float = Field(ge=0.0, le=0.8, default=0.08)
    energy_max: float = Field(gt=0.0, default=100.0)


class EnvironmentResponse(BaseModel):
    """Response representing a registered environment with versioning metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    width: int
    height: int
    version: int
    version_id: str
    config: dict[str, Any]
    config_hash: str

