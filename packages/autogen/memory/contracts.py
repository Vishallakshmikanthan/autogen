"""Typed data contracts, configuration models, and schemas for the Memory Engine (SYS-05)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import Position, TerrainType


class DecayPolicy(str, Enum):
    """Temporal decay dynamics for spatial memory confidence."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"


class EvictionPolicy(str, Enum):
    """Eviction policy when memory reaches maximum capacity."""

    FIFO = "fifo"
    LRU = "lru"
    LEAST_CONFIDENT = "least_confident"


class MemoryConfig(BaseModel):
    """Hyperparameter specification for an agent's memory subsystems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    working_memory_capacity: int = Field(default=10, ge=1, le=100)
    episodic_memory_capacity: int = Field(default=500, ge=1, le=100_000)
    spatial_memory_capacity: int = Field(default=1000, ge=1, le=100_000)
    decay_policy: DecayPolicy = DecayPolicy.EXPONENTIAL
    decay_rate: float = Field(default=0.02, ge=0.0)  # lambda per tick
    confidence_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    eviction_policy: EvictionPolicy = EvictionPolicy.LEAST_CONFIDENT


class EpisodicTransition(BaseModel):
    """Immutable transition record $e_t = (t, o_t, a_t, r_t, o_{t+1}, done_t, metadata)$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    state_observation: Observation
    action: Action
    reward: float
    next_observation: Observation | None = None
    terminal: bool = False
    feature_vector: tuple[float, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpatialMemoryCell(BaseModel):
    """Internal spatial representation of a remembered grid cell with decaying confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Position
    terrain: TerrainType
    walkable: bool
    last_seen_food: float = Field(ge=0.0)
    last_observed_tick: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class MemorySnapshot(BaseModel):
    """Immutable export schema for Memory Explorer API, UI inspection, and Algorithm Lab."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    current_step: int
    working_memory_size: int
    episodic_memory_size: int
    spatial_cells_count: int
    mean_spatial_confidence: float
    working_memory_steps: list[int]
    spatial_cells: list[SpatialMemoryCell]
