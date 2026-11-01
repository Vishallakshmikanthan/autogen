"""Agent architecture request and response schemas (M19, SYS-18)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateAgentRequest(BaseModel):
    """Payload to register an agent policy configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    agent_type: str = Field(min_length=1, max_length=100)
    policy_params: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response representing a registered agent configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    agent_type: str
    version: int
    version_id: str
    policy_params: dict[str, Any]

