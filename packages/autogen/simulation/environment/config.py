"""Configuration models for the simulated environment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.state.types import TerrainType


class EnvironmentConfig(BaseModel):
    """Configuration parameters for GridWorld and procedural generation."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    width: int = Field(default=32, ge=4, le=1024)
    height: int = Field(default=32, ge=4, le=1024)
    seed: int = Field(default=42, ge=0)

    # Biological and physical bounds
    energy_max: float = Field(default=100.0, gt=0.0)
    health_max: float = Field(default=100.0, gt=0.0)
    starvation_damage: float = Field(default=5.0, ge=0.0)
    satiated_healing: float = Field(default=1.0, ge=0.0)

    # Action energy costs & recovery
    cost_base_move: float = Field(default=1.0, ge=0.0)
    cost_stay: float = Field(default=0.2, ge=0.0)
    cost_collect: float = Field(default=0.8, ge=0.0)
    food_energy_ratio: float = Field(default=10.0, gt=0.0)
    collect_rate: float = Field(default=5.0, gt=0.0)

    # Procedural generation weights
    terrain_weights: dict[TerrainType, float] = Field(
        default_factory=lambda: {
            TerrainType.GRASSLAND: 0.55,
            TerrainType.FOREST: 0.20,
            TerrainType.DESERT: 0.10,
            TerrainType.WATER: 0.05,
            TerrainType.MOUNTAIN: 0.05,
            TerrainType.OBSTACLE: 0.05,
        }
    )
    min_walkable_ratio: float = Field(default=0.65, ge=0.1, le=1.0)

    # Environmental weather / seasonal hook
    weather_enabled: bool = Field(default=False)
