"""Immutable WorldState representation and canonical state hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.state.agent_state import AgentState
from autogen.simulation.state.types import Position, TerrainType


class CellState(BaseModel):
    """Immutable state snapshot for a single grid cell $C_{xy}$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Position
    terrain: TerrainType
    food: float = Field(ge=0.0)
    danger: float = Field(default=0.0, ge=0.0, le=1.0)
    movement_cost: float = Field(ge=0.0)


class WorldState(BaseModel):
    """Immutable snapshot of the entire world at discrete time step $t$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(default=0, ge=0)
    width: int = Field(ge=4)
    height: int = Field(ge=4)
    grid: tuple[tuple[CellState, ...], ...]
    agents: dict[str, AgentState] = Field(default_factory=dict)
    weather: dict[str, float] = Field(default_factory=dict)

    def get_cell(self, pos: Position) -> CellState:
        """Retrieves cell at coordinate pos."""
        if not (0 <= pos.x < self.width and 0 <= pos.y < self.height):
            raise IndexError(f"Position {pos} out of bounds ({self.width}x{self.height})")
        return self.grid[pos.y][pos.x]


def canonical_world_hash(state: WorldState) -> str:
    """Computes a deterministic, platform-independent SHA-256 hash of WorldState.

    Used for reproducible verification, determinism checks, and time-machine replay.
    """
    serialized_cells: list[dict[str, Any]] = []
    for y in range(state.height):
        for x in range(state.width):
            cell = state.grid[y][x]
            serialized_cells.append(
                {
                    "x": x,
                    "y": y,
                    "terrain": cell.terrain.value,
                    "food": round(float(cell.food), 4),
                    "danger": round(float(cell.danger), 4),
                }
            )

    serialized_agents: list[dict[str, Any]] = []
    for agent_id in sorted(state.agents.keys()):
        agent = state.agents[agent_id]
        serialized_agents.append(
            {
                "id": agent.agent_id,
                "x": agent.position.x,
                "y": agent.position.y,
                "energy": round(float(agent.energy), 4),
                "health": round(float(agent.health), 4),
                "carried_food": round(float(agent.carried_food), 4),
                "starvation_ticks": agent.starvation_ticks,
                "alive": agent.is_alive,
            }
        )

    canonical_obj = {
        "step": state.step,
        "width": state.width,
        "height": state.height,
        "cells": serialized_cells,
        "agents": serialized_agents,
        "weather": {k: round(float(v), 4) for k, v in sorted(state.weather.items())},
    }

    raw_json = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
