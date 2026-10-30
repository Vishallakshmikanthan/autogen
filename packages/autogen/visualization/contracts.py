"""Data Transfer Object (DTO) contracts for simulation visualization and UI consumers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.core.types import DecisionTrace


class ViewerCellState(BaseModel):
    """Serializable cell snapshot for renderer consumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    terrain: str
    food: float = Field(ge=0.0)
    movement_cost: float = Field(ge=0.0)


class ViewerAgentState(BaseModel):
    """Serializable agent snapshot for renderer consumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    energy: float = Field(ge=0.0)
    health: float = Field(ge=0.0)
    carried_food: float = Field(ge=0.0)
    is_alive: bool = True


class ViewerOverlayData(BaseModel):
    """Dynamic overlay layers (fog of war, trails, collisions, heatmaps)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fog_of_war_visible: list[tuple[int, int]] = Field(default_factory=list)
    movement_trails: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    collision_markers: list[tuple[int, int]] = Field(default_factory=list)
    resource_intensity: dict[str, float] = Field(default_factory=dict)


class ViewerTimeMachineState(BaseModel):
    """State of the playback controller and time-machine scrubber."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_step: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    available_snapshots: list[int] = Field(default_factory=list)
    is_paused: bool = True
    playback_speed: float = Field(default=1.0, gt=0.0)
    can_fork: bool = True


class ViewerAgentMindCell(BaseModel):
    """Cell known to the selected agent in its internal spatial memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    terrain: str
    food: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    last_observed_tick: int = Field(ge=0)


class ViewerState(BaseModel):
    """Universal, JSON-serializable DTO consumed by Pygame (M05) and Web Studio (M20)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    seed: int
    state_hash: str
    grid_cells: list[ViewerCellState]
    agents: list[ViewerAgentState]
    selected_agent_id: str | None = None
    decision_trace: DecisionTrace | None = None
    agent_mind: list[ViewerAgentMindCell] | None = None
    lookahead_tree: dict[str, Any] | None = None
    overlays: ViewerOverlayData
    time_machine: ViewerTimeMachineState
