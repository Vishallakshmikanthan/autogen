"""Village municipal zones, private huts, Granary Commons, Well, and landmarks."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.environment.village.commodities import CommodityInventory, CommodityType
from autogen.simulation.state.types import Position


class ZoneType(str, Enum):
    """Functional municipal zones and architectural landmarks in AUTOGEN Village."""

    WILD_FOREST = "wild_forest"
    PRIVATE_HUTS = "private_huts"
    CENTRAL_MARKET = "central_market"
    TOWN_WELL = "town_well"
    VILLAGE_COMMONS = "village_commons"
    AGRICULTURAL_FIELDS = "agricultural_fields"
    WINDMILL = "windmill"
    TAVERN = "tavern"
    SMITHY = "smithy"
    CHAPEL = "chapel"
    WATCHTOWER = "watchtower"
    RIVER_BRIDGE = "river_bridge"


class VillageZone(BaseModel):
    """Spatial municipal zone or landmark on the discrete village coordinate grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zone_id: str
    zone_type: ZoneType
    label: str
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)
    center: Position
    radius_3d: float = Field(default=8.0, gt=0.0)
    world_3d_coords: tuple[float, float] = (0.0, 0.0)

    def contains(self, pos: Position) -> bool:
        """Returns True if discrete coordinate pos lies inside this zone's bounding box."""
        return self.x_min <= pos.x <= self.x_max and self.y_min <= pos.y <= self.y_max


class PrivateHut(BaseModel):
    """Residential hut containing an agent's bed and private locked storage chest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hut_id: str
    owner_id: str
    position: Position
    world_3d_coords: tuple[float, float] = (0.0, 0.0)
    chest_inventory: CommodityInventory = Field(default_factory=CommodityInventory)
    chest_coins: float = Field(default=10.0, ge=0.0)
    lock_difficulty: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Base probability boost to detection/alarm when a burglar picks this chest",
    )
    rest_energy_recovery: float = Field(default=38.0, gt=0.0)


class GranaryCommons(BaseModel):
    """Community storage granary supporting public tithing, famine relief, and theft risk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Position
    world_3d_coords: tuple[float, float] = (-30.0, -16.0)
    storage: CommodityInventory = Field(
        default_factory=lambda: CommodityInventory(
            items={
                CommodityType.WHEAT: 60.0,
                CommodityType.WATER: 40.0,
                CommodityType.BERRIES: 20.0,
            }
        )
    )
    tithes_by_agent: dict[str, float] = Field(default_factory=dict)
    famine_relief_threshold: float = Field(
        default=28.0,
        description="Hunger level below which an agent is eligible for legal granary relief",
    )
    guard_security_bonus: float = Field(default=0.25, ge=0.0, le=1.0)

    def total_food_stored(self) -> float:
        """Returns total edible rations stored in the Granary Commons."""
        return self.storage.total_edible_units()


class TownWell(BaseModel):
    """Clean water reservoir at the heart of the Town Square."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Position
    world_3d_coords: tuple[float, float] = (0.0, 0.0)
    water_level: float = Field(default=300.0, ge=0.0)
    max_capacity: float = Field(default=300.0, gt=0.0)
    regen_per_step: float = Field(default=10.0, ge=0.0)


class AgriculturalField(BaseModel):
    """Cultivated farm plot producing wheat or vegetables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_id: str
    position: Position
    world_3d_coords: tuple[float, float] = (0.0, -58.0)
    crop_type: CommodityType = CommodityType.WHEAT
    crop_available: float = Field(default=90.0, ge=0.0)
    max_capacity: float = Field(default=120.0, gt=0.0)
    regrowth_rate: float = Field(default=3.0, ge=0.0)
