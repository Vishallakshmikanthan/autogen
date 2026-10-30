"""Contracts and configuration models for perception engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PerceptionMode(str, Enum):
    """Sensory field-of-view modes for agent perception."""

    FULL = "full"
    LOCAL_WINDOW = "local_window"
    LINE_OF_SIGHT = "line_of_sight"


class NoiseConfig(BaseModel):
    """Configuration for sensory observation noise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    resource_noise_std: float = Field(default=0.0, ge=0.0)
    agent_detection_probability: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentSensingConfig(BaseModel):
    """Configuration for detecting other agents in the vicinity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anonymous: bool = False
    sensing_radius: int = Field(default=3, ge=0)


class PerceptionConfig(BaseModel):
    """Complete sensory configuration for an individual agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: PerceptionMode = PerceptionMode.LOCAL_WINDOW
    radius: int = Field(default=3, ge=0)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    agent_sensing: AgentSensingConfig = Field(default_factory=AgentSensingConfig)
    custom_parameters: dict[str, Any] = Field(default_factory=dict)
