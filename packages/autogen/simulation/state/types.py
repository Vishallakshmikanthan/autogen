"""Spatial coordinates, directions, actions, and terrain definitions."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class Position(NamedTuple):
    """Immutable discrete 2D spatial coordinate (x, y)."""

    x: int
    y: int

    def offset(self, dx: int, dy: int) -> Position:
        return Position(self.x + dx, self.y + dy)

    def manhattan_distance(self, other: Position) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Direction(str, Enum):
    """Cardinal movement directions."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class ActionType(str, Enum):
    """Primitive actions available to agents in the world."""

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"
    COLLECT = "COLLECT"


DIRECTION_DISPLACEMENTS: dict[ActionType, tuple[int, int]] = {
    ActionType.UP: (0, -1),
    ActionType.DOWN: (0, 1),
    ActionType.LEFT: (-1, 0),
    ActionType.RIGHT: (1, 0),
    ActionType.STAY: (0, 0),
}


class TerrainType(str, Enum):
    """Discrete terrain classifications."""

    GRASSLAND = "grassland"
    FOREST = "forest"
    DESERT = "desert"
    WATER = "water"
    MOUNTAIN = "mountain"
    OBSTACLE = "obstacle"


class TerrainProperties(BaseModel):
    """Physical and biological characteristics of a terrain type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    walkable: bool
    move_cost: float = Field(ge=0.0)
    food_capacity: float = Field(ge=0.0)
    regen_rate: float = Field(ge=0.0)
    danger: float = Field(ge=0.0, le=1.0)


TERRAIN_PROPERTIES: dict[TerrainType, TerrainProperties] = {
    TerrainType.GRASSLAND: TerrainProperties(
        walkable=True, move_cost=1.0, food_capacity=10.0, regen_rate=0.2, danger=0.0
    ),
    TerrainType.FOREST: TerrainProperties(
        walkable=True, move_cost=1.5, food_capacity=25.0, regen_rate=0.5, danger=0.0
    ),
    TerrainType.DESERT: TerrainProperties(
        walkable=True, move_cost=2.0, food_capacity=2.0, regen_rate=0.05, danger=0.05
    ),
    TerrainType.WATER: TerrainProperties(
        walkable=False, move_cost=float("inf"), food_capacity=0.0, regen_rate=0.0, danger=0.0
    ),
    TerrainType.MOUNTAIN: TerrainProperties(
        walkable=False, move_cost=float("inf"), food_capacity=0.0, regen_rate=0.0, danger=0.0
    ),
    TerrainType.OBSTACLE: TerrainProperties(
        walkable=False, move_cost=float("inf"), food_capacity=0.0, regen_rate=0.0, danger=0.0
    ),
}
