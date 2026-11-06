"""VillageWorld spatial environment, municipal zones, and commodity/ethics physics."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.core.rng import SeedManager
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.environment.village.amenities import (
    AgriculturalField,
    GranaryCommons,
    PrivateHut,
    TownWell,
    VillageZone,
    ZoneType,
)
from autogen.simulation.environment.village.commodities import (
    COMMODITY_CATALOG,
    CommodityInventory,
    CommodityType,
    DoubleAuctionMarket,
    MarketAsk,
    MarketBid,
    TradeExecution,
)
from autogen.simulation.state.agent_state import AgentState
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.simulation.state.world_state import CellState, WorldState


class VillageRole(str, Enum):
    """Occupational roles of autonomous villagers."""

    FARMER = "Farmer"
    WATER_CARRIER = "Water-Carrier"
    MERCHANT = "Merchant"
    GUARD = "Guard"
    WOODCUTTER = "Woodcutter"
    WANDERER = "Wanderer"
    MAGISTRATE = "Magistrate"
    BLACKSMITH = "Blacksmith"
    MILLER = "Miller"
    HERBALIST = "Herbalist"
    OUTCAST = "Outcast"
    BARD = "Bard"


class VillageActionType(str, Enum):
    """High-level and primitive actions in AUTOGEN Village."""

    # Spatial movement
    MOVE_NORTH = "MOVE_NORTH"
    MOVE_SOUTH = "MOVE_SOUTH"
    MOVE_EAST = "MOVE_EAST"
    MOVE_WEST = "MOVE_WEST"
    STAY = "STAY"

    # Legal survival, craft, & civic pathways
    HARVEST_FIELD = "HARVEST_FIELD"
    FORAGE_FOREST = "FORAGE_FOREST"
    DRAW_WATER = "DRAW_WATER"
    REST_HUT = "REST_HUT"
    CONSUME_RATION = "CONSUME_RATION"
    TRADE_MARKET = "TRADE_MARKET"
    TITHE_GRANARY = "TITHE_GRANARY"
    CLAIM_FAMINE_RELIEF = "CLAIM_FAMINE_RELIEF"
    PATROL_GUARD = "PATROL_GUARD"
    MILL_WHEAT = "MILL_WHEAT"
    FORGE_TOOLS = "FORGE_TOOLS"
    BREW_HEALING_BROTH = "BREW_HEALING_BROTH"
    PRESIDE_SABHA = "PRESIDE_SABHA"
    ATTEND_SABHA = "ATTEND_SABHA"
    SING_BALLAD = "SING_BALLAD"
    INVESTIGATE_CRIME = "INVESTIGATE_CRIME"
    COMMUNITY_SERVICE = "COMMUNITY_SERVICE"
    COURTSHIP_BOND = "COURTSHIP_BOND"
    MOURN_AT_CHAPEL = "MOURN_AT_CHAPEL"

    # Unethical / Criminal pathways
    STEAL_GRANARY = "STEAL_GRANARY"
    STEAL_HUT_CHEST = "STEAL_HUT_CHEST"
    HOARD_AND_GOUGE = "HOARD_AND_GOUGE"
    FREE_RIDE_COMMONS = "FREE_RIDE_COMMONS"
    SABOTAGE_GRANARY = "SABOTAGE_GRANARY"
    OFFER_BRIBE = "OFFER_BRIBE"


LEGAL_VILLAGE_ACTIONS: frozenset[VillageActionType] = frozenset(
    {
        VillageActionType.MOVE_NORTH,
        VillageActionType.MOVE_SOUTH,
        VillageActionType.MOVE_EAST,
        VillageActionType.MOVE_WEST,
        VillageActionType.STAY,
        VillageActionType.HARVEST_FIELD,
        VillageActionType.FORAGE_FOREST,
        VillageActionType.DRAW_WATER,
        VillageActionType.REST_HUT,
        VillageActionType.CONSUME_RATION,
        VillageActionType.TRADE_MARKET,
        VillageActionType.TITHE_GRANARY,
        VillageActionType.CLAIM_FAMINE_RELIEF,
        VillageActionType.PATROL_GUARD,
        VillageActionType.MILL_WHEAT,
        VillageActionType.FORGE_TOOLS,
        VillageActionType.BREW_HEALING_BROTH,
        VillageActionType.PRESIDE_SABHA,
        VillageActionType.ATTEND_SABHA,
        VillageActionType.SING_BALLAD,
        VillageActionType.INVESTIGATE_CRIME,
        VillageActionType.COMMUNITY_SERVICE,
        VillageActionType.COURTSHIP_BOND,
        VillageActionType.MOURN_AT_CHAPEL,
    }
)

UNETHICAL_VILLAGE_ACTIONS: frozenset[VillageActionType] = frozenset(
    {
        VillageActionType.STEAL_GRANARY,
        VillageActionType.STEAL_HUT_CHEST,
        VillageActionType.HOARD_AND_GOUGE,
        VillageActionType.FREE_RIDE_COMMONS,
        VillageActionType.SABOTAGE_GRANARY,
        VillageActionType.OFFER_BRIBE,
    }
)


CANONICAL_VILLAGERS: tuple[tuple[str, str, VillageRole], ...] = (
    ("villager_arun", "Arun", VillageRole.FARMER),
    ("villager_meera", "Meera", VillageRole.FARMER),
    ("villager_ravi", "Ravi", VillageRole.FARMER),
    ("villager_kavya", "Kavya", VillageRole.WATER_CARRIER),
    ("villager_imran", "Imran", VillageRole.WATER_CARRIER),
    ("villager_lata", "Lata", VillageRole.MERCHANT),
    ("villager_bheem", "Bheem", VillageRole.MERCHANT),
    ("villager_tara", "Tara", VillageRole.GUARD),
    ("villager_dev", "Dev", VillageRole.GUARD),
    ("villager_nila", "Nila", VillageRole.WANDERER),
)

CANONICAL_ARCHETYPES: dict[str, str] = {
    "villager_arun": "Master Farmer & Harvester",
    "villager_meera": "Grain Miller & Baker",
    "villager_ravi": "Blacksmith & Toolmaker",
    "villager_kavya": "Water-Carrier & Well Keeper",
    "villager_imran": "Herbalist & Apothecary",
    "villager_lata": "Village Magistrate & Elder",
    "villager_bheem": "Double-Auction Merchant",
    "villager_tara": "Captain of the Watch",
    "villager_dev": "Town Guard & Investigator",
    "villager_nila": "Wandering Bard & Chronicler",
}


class VillagerPhysicalState(BaseModel):
    """Ground-truth physical and economic state of a villager owned by VillageWorld."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    name: str
    role: VillageRole
    archetype_title: str = "Villager"
    position: Position
    home_hut_id: str
    hunger: float = Field(default=80.0, ge=0.0, le=100.0, description="H_i in [0, 100]")
    hydration: float = Field(default=80.0, ge=0.0, le=100.0, description="T_i in [0, 100]")
    energy: float = Field(default=85.0, ge=0.0, le=100.0, description="E_i in [0, 100]")
    max_energy: float = Field(default=100.0, ge=25.0, le=100.0, description="Age-adjusted E_max")
    health: float = Field(default=100.0, ge=0.0, le=100.0)
    coins: float = Field(default=25.0, ge=0.0, description="Liquid coin balance C_i")
    inventory: CommodityInventory = Field(default_factory=CommodityInventory)
    societal_status: float = Field(default=0.1, ge=0.0, le=1.0, description="PageRank S_i(t)")
    is_alive: bool = True
    age_ticks: int = Field(default=160, ge=0)
    generation: int = Field(default=1, ge=1)
    life_stage: str = "Adult"
    dominant_emotion: str = "Calm / Composed"
    pad_vector: tuple[float, float, float] = (0.1, 0.05, 0.1)
    starvation_streak: int = Field(default=0, ge=0)
    in_pillory_ticks: int = Field(default=0, ge=0)
    crimes_committed: int = Field(default=0, ge=0)
    crimes_detected: int = Field(default=0, ge=0)
    tithes_contributed: float = Field(default=0.0, ge=0.0)
    last_action: VillageActionType = VillageActionType.STAY
    current_goal_zone: str = "town_well"


class VillageInteractionOutcome(BaseModel):
    """Ground-truth record of a cooperative, defective, or criminal event at step $t$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    actor_id: str
    target_id: str | None = None
    action_type: VillageActionType
    category: str = Field(
        description="One of: 'coop', 'defect', 'crime_caught', 'crime_unseen', 'trade', 'work'"
    )
    commodity: CommodityType | None = None
    quantity: float = 0.0
    witness_ids: tuple[str, ...] = ()
    detection_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    detected: bool = False
    coin_penalty: float = Field(default=0.0, ge=0.0)
    description: str = ""


class VisibleVillagerSummary(BaseModel):
    """Observable attributes of a nearby villager (never leaks private hut chest)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    name: str
    role: VillageRole
    position: Position
    societal_status: float
    in_pillory: bool = False
    last_action: VillageActionType = VillageActionType.STAY


class VillageObservation(BaseModel):
    """Sensory observation granted to a villager at discrete step $t$.

    Maintains strict separation from ground-truth world state:
    agents only observe their own physiological state, local witnesses within
    vision radius, public market prices, and public Granary/Well levels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    step: int = Field(ge=0)
    time_of_day: float = Field(ge=0.0, lt=24.0)
    ambient_light: float = Field(ge=0.0, le=1.0)
    is_drought: bool = False
    position: Position
    home_hut_position: Position
    hunger: float
    hydration: float
    energy: float
    health: float
    coins: float
    inventory: CommodityInventory
    home_chest_inventory: CommodityInventory
    societal_status: float
    in_pillory_ticks: int
    current_zone: ZoneType | None = None
    zone_centers: dict[str, Position] = Field(default_factory=dict)
    visible_villagers: tuple[VisibleVillagerSummary, ...] = ()
    visible_guards_count: int = 0
    near_watchtower: bool = False
    market_clearing_prices: dict[CommodityType, float] = Field(default_factory=dict)
    granary_wheat_available: float = 0.0
    well_water_available: float = 0.0
    witnessed_outcomes: tuple[VillageInteractionOutcome, ...] = ()


class VillageWorldConfig(BaseModel):
    """Configuration for instantiating the AUTOGEN Village Society environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(default=40, ge=20, le=80)
    height: int = Field(default=40, ge=20, le=80)
    seed: int = Field(default=42, ge=0)
    vision_radius: int = Field(default=10, ge=2, le=30)
    hours_per_step: float = Field(default=0.5, gt=0.0, le=6.0)
    initial_time_of_day: float = Field(default=8.0, ge=0.0, lt=24.0)
    hunger_decay_per_step: float = Field(default=1.4, ge=0.0)
    hydration_decay_per_step: float = Field(default=1.8, ge=0.0)
    energy_decay_per_step: float = Field(default=1.1, ge=0.0)
    crime_coin_penalty: float = Field(default=12.0, ge=0.0)
    pillory_sentence_ticks: int = Field(default=3, ge=0)
    guard_sight_multiplier: float = Field(default=1.45, ge=1.0)
    watchtower_sight_bonus: float = Field(default=0.22, ge=0.0, le=0.8)
    drought_active: bool = False


class VillageWorld:
    """Ground-truth physical and economic environment for AUTOGEN Village.

    Enforces physical laws, movement, harvesting, private chests, double-auction
    market clearing, and witness line-of-sight detection. Agents NEVER directly
    mutate world state.
    """

    def __init__(self, config: VillageWorldConfig | None = None) -> None:
        self.config = config or VillageWorldConfig()
        self.seed_manager = SeedManager(self.config.seed)
        self._detection_rng = self.seed_manager.spawn_stream("village_detection")
        self._forage_rng = self.seed_manager.spawn_stream("village_forage")

        self.step_count: int = 0
        self.time_of_day: float = self.config.initial_time_of_day
        self.drought_active: bool = self.config.drought_active

        self.zones: dict[str, VillageZone] = {}
        self.huts: dict[str, PrivateHut] = {}
        self.fields: dict[str, AgriculturalField] = {}
        self.watchtowers: list[Position] = []
        self.granary = GranaryCommons(position=Position(16, 28))
        self.well = TownWell(position=Position(20, 20))
        self.market = DoubleAuctionMarket()

        self.villagers: dict[str, VillagerPhysicalState] = {}
        self.step_outcomes: list[VillageInteractionOutcome] = []
        self.all_outcomes: list[VillageInteractionOutcome] = []
        self._world_state: WorldState | None = None

        self.reset(self.config.seed)

    @property
    def state(self) -> WorldState:
        """Returns the canonical immutable WorldState snapshot (BaseEnvironment contract)."""
        if self._world_state is None:
            raise RuntimeError("VillageWorld not initialized.")
        return self._world_state

    def reset(self, seed: int | None = None) -> WorldState:
        """Resets the village grid, zones, amenities, and villagers deterministically."""
        if seed is not None:
            self.config = self.config.model_copy(update={"seed": seed})
            self.seed_manager = SeedManager(seed)
            self._detection_rng = self.seed_manager.spawn_stream("village_detection")
            self._forage_rng = self.seed_manager.spawn_stream("village_forage")

        self.step_count = 0
        self.time_of_day = self.config.initial_time_of_day
        self.drought_active = self.config.drought_active
        self.market = DoubleAuctionMarket()
        self.step_outcomes = []
        self.all_outcomes = []

        self._build_layout()
        self._spawn_default_villagers()
        self._world_state = self._build_world_state()
        return self._world_state

    def restore_state(self, state: WorldState) -> None:
        """Restores underlying WorldState snapshot."""
        self._world_state = state
        self.step_count = state.step

    def _build_layout(self) -> None:
        """Constructs the municipal zones and 3D reference coordinates."""
        w, h = self.config.width, self.config.height
        cx, cy = w // 2, h // 2

        self.zones = {
            "wild_forest": VillageZone(
                zone_id="wild_forest",
                zone_type=ZoneType.WILD_FOREST,
                label="Wild Forest",
                x_min=0,
                y_min=0,
                x_max=w - 1,
                y_max=max(5, h // 5),
                center=Position(cx, max(2, h // 10)),
                radius_3d=55.0,
                world_3d_coords=(-115.0, -20.0),
            ),
            "private_huts_west": VillageZone(
                zone_id="private_huts_west",
                zone_type=ZoneType.PRIVATE_HUTS,
                label="West Residential Huts",
                x_min=1,
                y_min=h // 5 + 1,
                x_max=w // 4,
                y_max=(3 * h) // 5,
                center=Position(max(2, w // 8), (2 * h) // 5),
                radius_3d=18.0,
                world_3d_coords=(-34.0, 18.0),
            ),
            "central_market": VillageZone(
                zone_id="central_market",
                zone_type=ZoneType.CENTRAL_MARKET,
                label="Central Market",
                x_min=w // 4 + 1,
                y_min=h // 5 + 1,
                x_max=(3 * w) // 4 - 1,
                y_max=(2 * h) // 5,
                center=Position(cx, (3 * h) // 10),
                radius_3d=12.0,
                world_3d_coords=(0.0, 31.0),
            ),
            "private_huts_east": VillageZone(
                zone_id="private_huts_east",
                zone_type=ZoneType.PRIVATE_HUTS,
                label="East Residential Huts",
                x_min=(3 * w) // 4,
                y_min=h // 5 + 1,
                x_max=w - 2,
                y_max=(3 * h) // 5,
                center=Position(min(w - 3, (7 * w) // 8), (2 * h) // 5),
                radius_3d=18.0,
                world_3d_coords=(33.0, 36.0),
            ),
            "town_well": VillageZone(
                zone_id="town_well",
                zone_type=ZoneType.TOWN_WELL,
                label="Town Square & Well",
                x_min=cx - 3,
                y_min=cy - 2,
                x_max=cx + 3,
                y_max=cy + 3,
                center=Position(cx, cy),
                radius_3d=14.0,
                world_3d_coords=(0.0, 0.0),
            ),
            "village_commons": VillageZone(
                zone_id="village_commons",
                zone_type=ZoneType.VILLAGE_COMMONS,
                label="Village Commons & Granary",
                x_min=w // 4,
                y_min=(3 * h) // 5 + 1,
                x_max=(2 * w) // 3 - 1,
                y_max=h - 2,
                center=Position((2 * w) // 5, (3 * h) // 4),
                radius_3d=10.0,
                world_3d_coords=(-30.0, -16.0),
            ),
            "agricultural_fields": VillageZone(
                zone_id="agricultural_fields",
                zone_type=ZoneType.AGRICULTURAL_FIELDS,
                label="Agricultural Fields",
                x_min=(2 * w) // 3,
                y_min=(3 * h) // 5 + 1,
                x_max=w - 2,
                y_max=h - 2,
                center=Position((4 * w) // 5, (3 * h) // 4),
                radius_3d=16.0,
                world_3d_coords=(0.0, -58.0),
            ),
            "windmill": VillageZone(
                zone_id="windmill",
                zone_type=ZoneType.WINDMILL,
                label="Old Windmill",
                x_min=2,
                y_min=(3 * h) // 5 + 2,
                x_max=max(4, w // 5),
                y_max=h - 3,
                center=Position(max(3, w // 8), (4 * h) // 5),
                radius_3d=7.0,
                world_3d_coords=(-52.0, -34.0),
            ),
            "tavern": VillageZone(
                zone_id="tavern",
                zone_type=ZoneType.TAVERN,
                label="Golden Hearth Tavern",
                x_min=cx + 4,
                y_min=cy - 3,
                x_max=min(w - 2, cx + 8),
                y_max=cy + 2,
                center=Position(min(w - 3, cx + 6), cy),
                radius_3d=9.0,
                world_3d_coords=(24.0, -13.0),
            ),
            "smithy": VillageZone(
                zone_id="smithy",
                zone_type=ZoneType.SMITHY,
                label="Blacksmith Forge",
                x_min=max(1, cx - 8),
                y_min=cy - 3,
                x_max=cx - 4,
                y_max=cy + 2,
                center=Position(max(2, cx - 6), cy),
                radius_3d=8.0,
                world_3d_coords=(-25.0, 14.0),
            ),
            "chapel": VillageZone(
                zone_id="chapel",
                zone_type=ZoneType.CHAPEL,
                label="St. Aurelia Chapel & Memorials",
                x_min=min(w - 5, cx + 6),
                y_min=max(2, cy - 8),
                x_max=min(w - 2, cx + 10),
                y_max=max(4, cy - 4),
                center=Position(min(w - 3, cx + 8), max(3, cy - 6)),
                radius_3d=9.0,
                world_3d_coords=(42.0, -22.0),
            ),
        }

        self.well = TownWell(
            position=self.zones["town_well"].center,
            world_3d_coords=(0.0, 0.0),
            water_level=160.0 if self.drought_active else 300.0,
            max_capacity=300.0,
            regen_per_step=2.5 if self.drought_active else 10.0,
        )
        self.granary = GranaryCommons(
            position=self.zones["village_commons"].center,
            world_3d_coords=(-30.0, -16.0),
            storage=CommodityInventory(
                items={
                    CommodityType.WHEAT: 35.0 if self.drought_active else 75.0,
                    CommodityType.WATER: 25.0 if self.drought_active else 50.0,
                    CommodityType.BERRIES: 18.0,
                }
            ),
        )

        f_center = self.zones["agricultural_fields"].center
        self.fields = {
            "farm_0": AgriculturalField(
                field_id="farm_0",
                position=f_center,
                world_3d_coords=(-30.0, -58.0),
                crop_type=CommodityType.WHEAT,
                crop_available=45.0 if self.drought_active else 100.0,
                regrowth_rate=1.0 if self.drought_active else 3.5,
            ),
            "farm_1": AgriculturalField(
                field_id="farm_1",
                position=Position(min(w - 2, f_center.x + 2), f_center.y),
                world_3d_coords=(0.0, -62.0),
                crop_type=CommodityType.WHEAT,
                crop_available=45.0 if self.drought_active else 100.0,
                regrowth_rate=1.0 if self.drought_active else 3.5,
            ),
        }

        self.watchtowers = [
            Position(cx - 2, (3 * h) // 5),
            Position(cx + 2, (2 * h) // 5),
        ]

    def _spawn_default_villagers(self) -> None:
        """Instantiates the 10 canonical villagers and their private huts."""
        self.huts = {}
        self.villagers = {}

        west_c = self.zones["private_huts_west"].center
        east_c = self.zones["private_huts_east"].center

        for idx, (agent_id, name, role) in enumerate(CANONICAL_VILLAGERS):
            is_west = idx < 5
            base = west_c if is_west else east_c
            offset_y = (idx % 5) - 2
            hut_pos = Position(
                max(1, min(self.config.width - 2, base.x)),
                max(1, min(self.config.height - 2, base.y + offset_y)),
            )
            ang = idx * 2.399 + 0.4
            r3d = 34.0 + (idx % 3) * 4.0
            hut_3d = (round(math.cos(ang) * r3d, 2), round(math.sin(ang) * r3d, 2))
            hut_id = f"hut_{idx}"

            initial_chest = CommodityInventory(
                items={
                    CommodityType.WHEAT: 6.0 if role == VillageRole.FARMER else 3.0,
                    CommodityType.WATER: 6.0 if role == VillageRole.WATER_CARRIER else 3.0,
                    CommodityType.TOOLS: 2.0 if role == VillageRole.MERCHANT else 1.0,
                }
            )
            self.huts[hut_id] = PrivateHut(
                hut_id=hut_id,
                owner_id=agent_id,
                position=hut_pos,
                world_3d_coords=hut_3d,
                chest_inventory=initial_chest,
                chest_coins=20.0 if role == VillageRole.MERCHANT else 10.0,
            )

            initial_inv = CommodityInventory(
                items={
                    CommodityType.WHEAT: 3.0 if role == VillageRole.FARMER else 1.5,
                    CommodityType.WATER: 3.0 if role == VillageRole.WATER_CARRIER else 1.5,
                    CommodityType.BERRIES: 2.0 if role == VillageRole.WANDERER else 0.5,
                    CommodityType.TOOLS: 1.0 if agent_id in ("villager_arun", "villager_ravi") else 0.0,
                    CommodityType.HERBS: 2.0 if agent_id == "villager_imran" else 0.0,
                    CommodityType.IRON_ORE: 2.0 if agent_id == "villager_ravi" else 0.0,
                }
            )
            self.villagers[agent_id] = VillagerPhysicalState(
                agent_id=agent_id,
                name=name,
                role=role,
                archetype_title=CANONICAL_ARCHETYPES.get(agent_id, f"{role.value} Specialist"),
                position=hut_pos,
                home_hut_id=hut_id,
                hunger=82.0,
                hydration=84.0,
                energy=88.0,
                max_energy=92.0 if agent_id == "villager_lata" else 100.0,
                health=100.0,
                coins=45.0 if role == VillageRole.MERCHANT else 22.0,
                inventory=initial_inv,
                societal_status=round(1.0 / len(CANONICAL_VILLAGERS), 4),
                age_ticks=250 if agent_id == "villager_lata" else 140 + idx * 8,
                life_stage="Elder" if agent_id == "villager_lata" else "Adult",
            )

    def spawn_custom_villager(
        self,
        agent_id: str,
        name: str | None = None,
        role: VillageRole = VillageRole.FARMER,
        position: Position | None = None,
        hunger: float = 80.0,
        hydration: float = 80.0,
        energy: float = 85.0,
        coins: float = 20.0,
        inventory: CommodityInventory | None = None,
    ) -> VillagerPhysicalState:
        """Adds or replaces a villager in the village environment."""
        pos = position or self.zones["town_well"].center
        hut_id = f"hut_{agent_id}"
        if hut_id not in self.huts:
            self.huts[hut_id] = PrivateHut(
                hut_id=hut_id,
                owner_id=agent_id,
                position=pos,
                world_3d_coords=(0.0, 0.0),
                chest_inventory=CommodityInventory(items={CommodityType.WHEAT: 2.0}),
                chest_coins=5.0,
            )
        v_state = VillagerPhysicalState(
            agent_id=agent_id,
            name=name or agent_id,
            role=role,
            position=pos,
            home_hut_id=hut_id,
            hunger=hunger,
            hydration=hydration,
            energy=energy,
            coins=coins,
            inventory=inventory or CommodityInventory(items={CommodityType.WHEAT: 1.0}),
            societal_status=round(1.0 / max(1, len(self.villagers) + 1), 4),
        )
        self.villagers[agent_id] = v_state
        self._world_state = self._build_world_state()
        return v_state

    def get_ambient_light(self) -> float:
        """Computes ambient daylight factor in $[0.15, 1.0]$ from time_of_day $\\in [0, 24)$."""
        angle = ((self.time_of_day - 6.0) / 12.0) * math.pi
        sun_elev = math.sin(angle)
        if sun_elev >= 0.0:
            return round(0.25 + 0.75 * min(1.0, sun_elev / 0.6), 4)
        return round(max(0.15, 0.25 + 0.10 * sun_elev), 4)

    def get_zone_at(self, pos: Position) -> VillageZone | None:
        """Returns the primary VillageZone containing pos, if any."""
        for z in self.zones.values():
            if z.contains(pos):
                return z
        return None

    def has_line_of_sight(self, origin: Position, target: Position) -> bool:
        """Bresenham raycast checking if line-of-sight between origin and target is unoccluded.

        Forest canopy cells (other than endpoints) and hut walls along the interior
        of the ray occlude sight.
        """
        x0, y0 = origin.x, origin.y
        x1, y1 = target.x, target.y
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        hut_positions = {h.position for h in self.huts.values()}
        forest_zone = self.zones.get("wild_forest")

        while True:
            cur = Position(x0, y0)
            if cur != origin and cur != target:
                if cur in hut_positions:
                    return False
                if forest_zone is not None and forest_zone.contains(cur):
                    # Dense forest canopy occludes long-range sight (> 3 tiles)
                    if origin.manhattan_distance(target) > 3:
                        return False
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return True

    def compute_witness_sight_probability(
        self,
        actor_pos: Position,
        witness_pos: Position,
        witness_role: VillageRole = VillageRole.FARMER,
        ambient_light: float | None = None,
    ) -> float:
        """Computes individual witness sight probability $P_{j, \\text{sight}}(\\mathbf{x}_i, \\mathbf{x}_j)$.

        Strictly satisfies the Detection Monotonicity invariant:
        $\\frac{\\partial P_{j, \\text{sight}}}{\\partial \\|\\mathbf{x}_i - \\mathbf{x}_j\\|} \\le 0$
        along unoccluded lines of sight.
        """
        if not self.has_line_of_sight(witness_pos, actor_pos):
            return 0.0

        dist = math.hypot(actor_pos.x - witness_pos.x, actor_pos.y - witness_pos.y)
        max_range = float(self.config.vision_radius) * (
            1.25 if witness_role == VillageRole.GUARD else 1.0
        )
        if dist > max_range:
            return 0.0

        light = ambient_light if ambient_light is not None else self.get_ambient_light()
        role_mult = (
            self.config.guard_sight_multiplier
            if witness_role == VillageRole.GUARD
            else 1.0
        )

        # Monotonically decreasing exponential-linear decay with distance
        distance_factor = math.exp(-0.18 * dist) * max(0.0, 1.0 - dist / (max_range + 1.0))
        base_prob = 0.88 * light * role_mult * distance_factor
        return float(max(0.0, min(0.98, round(base_prob, 6))))

    def compute_detection_probability(
        self,
        actor_id: str,
        actor_pos: Position | None = None,
        action_type: VillageActionType = VillageActionType.STEAL_GRANARY,
    ) -> tuple[float, list[str]]:
        """Computes compound witness detection probability (§2.3.1):

        $$P_{\\text{detect}} = 1 - \\prod_{j \\in \\text{Witnesses}} (1 - P_{j, \\text{sight}})$$

        Includes watchtower proximity and lock/granary security bonuses.
        """
        actor = self.villagers.get(actor_id)
        pos = actor_pos or (actor.position if actor else self.zones["town_well"].center)
        light = self.get_ambient_light()

        not_detected_product = 1.0
        witness_ids: list[str] = []

        for other_id, other in sorted(self.villagers.items()):
            if other_id == actor_id or not other.is_alive:
                continue
            p_sight = self.compute_witness_sight_probability(
                actor_pos=pos,
                witness_pos=other.position,
                witness_role=other.role,
                ambient_light=light,
            )
            if p_sight > 0.0:
                witness_ids.append(other_id)
                not_detected_product *= 1.0 - p_sight

        # Environmental security modifiers (Watchtowers, Granary lock, Hut lock)
        for wt_pos in self.watchtowers:
            wt_dist = math.hypot(pos.x - wt_pos.x, pos.y - wt_pos.y)
            if wt_dist <= self.config.vision_radius and self.has_line_of_sight(wt_pos, pos):
                wt_sight = self.config.watchtower_sight_bonus * math.exp(-0.15 * wt_dist) * light
                not_detected_product *= 1.0 - min(0.85, wt_sight)

        if action_type == VillageActionType.STEAL_GRANARY:
            not_detected_product *= 1.0 - (self.granary.guard_security_bonus * light)
        elif action_type == VillageActionType.STEAL_HUT_CHEST:
            not_detected_product *= 1.0 - (0.20 * light)
        elif action_type == VillageActionType.HOARD_AND_GOUGE:
            # Market price gouging is public at the market stalls if witnesses are present
            not_detected_product *= 0.65

        p_detect = 1.0 - not_detected_product
        return float(max(0.0, min(0.99, round(p_detect, 6)))), witness_ids

    def update_societal_statuses(self, status_map: dict[str, float]) -> None:
        """Updates villagers' global PageRank societal status $\\mathcal{S}_i(t)$."""
        updated: dict[str, VillagerPhysicalState] = {}
        for aid, v in self.villagers.items():
            if aid in status_map:
                s_val = float(max(0.0, min(1.0, status_map[aid])))
                updated[aid] = v.model_copy(update={"societal_status": round(s_val, 6)})
            else:
                updated[aid] = v
        self.villagers = updated
        self._world_state = self._build_world_state()

    def get_village_observation(self, agent_id: str) -> VillageObservation:
        """Constructs a villager's rich sensory observation without leaking hidden state."""
        v = self.villagers.get(agent_id)
        if v is None:
            raise ValueError(f"Villager '{agent_id}' not found in VillageWorld.")

        light = self.get_ambient_light()
        eff_radius = max(3, int(round(self.config.vision_radius * (0.5 + 0.5 * light))))

        visible_villagers: list[VisibleVillagerSummary] = []
        guards_seen = 0
        for other_id, other in sorted(self.villagers.items()):
            if other_id == agent_id or not other.is_alive:
                continue
            dist = math.hypot(v.position.x - other.position.x, v.position.y - other.position.y)
            if dist <= eff_radius and self.has_line_of_sight(v.position, other.position):
                visible_villagers.append(
                    VisibleVillagerSummary(
                        agent_id=other.agent_id,
                        name=other.name,
                        role=other.role,
                        position=other.position,
                        societal_status=other.societal_status,
                        in_pillory=(other.in_pillory_ticks > 0),
                        last_action=other.last_action,
                    )
                )
                if other.role == VillageRole.GUARD:
                    guards_seen += 1

        near_wt = any(
            math.hypot(v.position.x - wt.x, v.position.y - wt.y) <= self.config.vision_radius
            for wt in self.watchtowers
        )
        cur_zone = self.get_zone_at(v.position)
        home_hut = self.huts.get(v.home_hut_id)
        home_pos = home_hut.position if home_hut else v.position
        home_chest = home_hut.chest_inventory if home_hut else CommodityInventory()

        # Filter outcomes that this agent either performed, was targeted by, or witnessed
        witnessed = tuple(
            out
            for out in self.step_outcomes
            if out.actor_id == agent_id
            or out.target_id == agent_id
            or agent_id in out.witness_ids
            or out.category in ("coop", "crime_caught", "trade")
        )

        return VillageObservation(
            agent_id=agent_id,
            step=self.step_count,
            time_of_day=self.time_of_day,
            ambient_light=light,
            is_drought=self.drought_active,
            position=v.position,
            home_hut_position=home_pos,
            hunger=v.hunger,
            hydration=v.hydration,
            energy=v.energy,
            health=v.health,
            coins=v.coins,
            inventory=v.inventory,
            home_chest_inventory=home_chest,
            societal_status=v.societal_status,
            in_pillory_ticks=v.in_pillory_ticks,
            current_zone=cur_zone.zone_type if cur_zone else None,
            zone_centers={zid: z.center for zid, z in self.zones.items()},
            visible_villagers=tuple(visible_villagers),
            visible_guards_count=guards_seen,
            near_watchtower=near_wt,
            market_clearing_prices=dict(self.market.clearing_prices),
            granary_wheat_available=self.granary.storage.get(CommodityType.WHEAT),
            well_water_available=self.well.water_level,
            witnessed_outcomes=witnessed,
        )

    def get_observation(self, agent_id: str) -> Observation:
        """Satisfies BaseEnvironment.get_observation for standard SimulationEngine."""
        ws = self.state
        ag = ws.agents[agent_id]
        v_obs = self.get_village_observation(agent_id)
        vis_cells: list[CellState] = []
        r = 3
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = ag.position.x + dx, ag.position.y + dy
                if 0 <= nx < ws.width and 0 <= ny < ws.height:
                    vis_cells.append(ws.grid[ny][nx])
        vis_agents = tuple((vv.agent_id, vv.position) for vv in v_obs.visible_villagers)
        return Observation(
            agent_id=agent_id,
            step=self.step_count,
            position=ag.position,
            energy=ag.energy,
            health=ag.health,
            carried_food=ag.carried_food,
            is_alive=ag.is_alive,
            visible_cells=tuple(vis_cells),
            visible_agents=vis_agents,
        )

    def validate_action(self, agent_id: str, action: ActionType) -> bool:
        """Validates standard ActionType on the village grid (BaseEnvironment contract)."""
        v = self.villagers.get(agent_id)
        if v is None or not v.is_alive:
            return False
        if v.in_pillory_ticks > 0 and action != ActionType.STAY:
            return False
        dx, dy = {
            ActionType.UP: (0, -1),
            ActionType.DOWN: (0, 1),
            ActionType.LEFT: (-1, 0),
            ActionType.RIGHT: (1, 0),
            ActionType.STAY: (0, 0),
            ActionType.COLLECT: (0, 0),
        }.get(action, (0, 0))
        nx, ny = v.position.x + dx, v.position.y + dy
        return 0 <= nx < self.config.width and 0 <= ny < self.config.height

    def apply_transitions(self, actions: dict[str, ActionType]) -> WorldState:
        """Adapts standard ActionType dictionary into village actions and steps the world."""
        rich_actions: dict[str, Action] = {}
        for aid, act_type in actions.items():
            v = self.villagers.get(aid)
            if v is None:
                continue
            v_act = self._map_primitive_to_village_action(v, act_type)
            rich_actions[aid] = Action(
                agent_id=aid,
                action_type=act_type,
                parameters={"village_action": v_act.value},
            )
        return self.step_village(rich_actions)

    def _map_primitive_to_village_action(
        self, v: VillagerPhysicalState, act_type: ActionType
    ) -> VillageActionType:
        if act_type == ActionType.UP:
            return VillageActionType.MOVE_NORTH
        if act_type == ActionType.DOWN:
            return VillageActionType.MOVE_SOUTH
        if act_type == ActionType.LEFT:
            return VillageActionType.MOVE_WEST
        if act_type == ActionType.RIGHT:
            return VillageActionType.MOVE_EAST
        if act_type == ActionType.COLLECT:
            zone = self.get_zone_at(v.position)
            if zone is None:
                return VillageActionType.CONSUME_RATION
            if zone.zone_type == ZoneType.TOWN_WELL:
                return VillageActionType.DRAW_WATER
            if zone.zone_type == ZoneType.AGRICULTURAL_FIELDS:
                return VillageActionType.HARVEST_FIELD
            if zone.zone_type == ZoneType.WILD_FOREST:
                return VillageActionType.FORAGE_FOREST
            if zone.zone_type == ZoneType.CENTRAL_MARKET:
                return VillageActionType.TRADE_MARKET
            if zone.zone_type == ZoneType.VILLAGE_COMMONS:
                if v.hunger < self.granary.famine_relief_threshold:
                    return VillageActionType.CLAIM_FAMINE_RELIEF
                return VillageActionType.TITHE_GRANARY
            if zone.zone_type == ZoneType.PRIVATE_HUTS:
                return VillageActionType.REST_HUT
        return VillageActionType.STAY

    def step_village(self, actions: dict[str, Action]) -> WorldState:
        """Executes one complete physical, economic, and ethics detection tick."""
        self.step_count += 1
        self.time_of_day = round((self.time_of_day + self.config.hours_per_step) % 24.0, 4)
        self.step_outcomes = []

        bids: list[MarketBid] = []
        asks: list[MarketAsk] = []
        updated_villagers: dict[str, VillagerPhysicalState] = {}

        # Phase 1: Regenerate natural resources
        new_water = min(
            self.well.max_capacity,
            self.well.water_level + self.well.regen_per_step,
        )
        self.well = self.well.model_copy(update={"water_level": round(new_water, 4)})

        for fid, field in self.fields.items():
            regrown = min(field.max_capacity, field.crop_available + field.regrowth_rate)
            self.fields[fid] = field.model_copy(update={"crop_available": round(regrown, 4)})

        # Phase 2: Resolve each villager's action deterministically in sorted agent_id order
        for aid in sorted(self.villagers.keys()):
            v = self.villagers[aid]
            if not v.is_alive:
                updated_villagers[aid] = v
                continue

            # Apply physiological metabolism and inventory spoilage
            decay_mult = 1.35 if self.drought_active else 1.0
            hunger = max(0.0, v.hunger - self.config.hunger_decay_per_step)
            hydration = max(0.0, v.hydration - self.config.hydration_decay_per_step * decay_mult)
            energy = max(0.0, v.energy - self.config.energy_decay_per_step)
            inv = v.inventory.apply_spoilage()
            coins = v.coins
            pos = v.position
            pillory = v.in_pillory_ticks
            crimes_comm = v.crimes_committed
            crimes_det = v.crimes_detected
            tithes_tot = v.tithes_contributed
            goal_zone = v.current_goal_zone

            # Auto-consume emergency water/food from personal inventory if critically low
            if hydration < 35.0 and inv.get(CommodityType.WATER) >= 0.5:
                use_w = min(1.0, inv.get(CommodityType.WATER))
                inv = inv.with_delta(CommodityType.WATER, -use_w)
                hydration = min(100.0, hydration + use_w * COMMODITY_CATALOG[CommodityType.WATER].hydration_value)

            if hunger < 35.0:
                for food_type in (CommodityType.WHEAT, CommodityType.FISH, CommodityType.BERRIES):
                    if inv.get(food_type) >= 0.5:
                        use_f = min(1.0, inv.get(food_type))
                        inv = inv.with_delta(food_type, -use_f)
                        spec = COMMODITY_CATALOG[food_type]
                        hunger = min(100.0, hunger + use_f * spec.nutrition_value)
                        hydration = min(100.0, hydration + use_f * spec.hydration_value)
                        energy = min(100.0, energy + use_f * spec.energy_boost)
                        break

            # If serving pillory sentence, agent cannot act this tick
            if pillory > 0:
                updated_villagers[aid] = v.model_copy(
                    update={
                        "hunger": round(hunger, 4),
                        "hydration": round(hydration, 4),
                        "energy": round(energy, 4),
                        "inventory": inv,
                        "in_pillory_ticks": pillory - 1,
                        "last_action": VillageActionType.STAY,
                    }
                )
                continue

            act_obj = actions.get(aid)
            raw_v_act = (
                act_obj.parameters.get("village_action")
                if act_obj and "village_action" in act_obj.parameters
                else None
            )
            if raw_v_act is not None:
                try:
                    v_act = VillageActionType(raw_v_act)
                except ValueError:
                    v_act = self._map_primitive_to_village_action(
                        v, act_obj.action_type if act_obj else ActionType.STAY
                    )
            elif act_obj is not None:
                v_act = self._map_primitive_to_village_action(v, act_obj.action_type)
            else:
                v_act = VillageActionType.STAY

            if act_obj and "goal_zone" in act_obj.parameters:
                goal_zone = str(act_obj.parameters["goal_zone"])

            # Movement handling
            if v_act in (
                VillageActionType.MOVE_NORTH,
                VillageActionType.MOVE_SOUTH,
                VillageActionType.MOVE_EAST,
                VillageActionType.MOVE_WEST,
            ):
                dx, dy = {
                    VillageActionType.MOVE_NORTH: (0, -1),
                    VillageActionType.MOVE_SOUTH: (0, 1),
                    VillageActionType.MOVE_WEST: (-1, 0),
                    VillageActionType.MOVE_EAST: (1, 0),
                }[v_act]
                nx = max(0, min(self.config.width - 1, pos.x + dx))
                ny = max(0, min(self.config.height - 1, pos.y + dy))
                pos = Position(nx, ny)

            elif v_act == VillageActionType.DRAW_WATER:
                drawn = min(3.0, self.well.water_level)
                if drawn > 0.0:
                    self.well = self.well.model_copy(
                        update={"water_level": round(self.well.water_level - drawn, 4)}
                    )
                    inv = inv.with_delta(CommodityType.WATER, drawn)
                    hydration = min(100.0, hydration + 28.0)
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category="work",
                            commodity=CommodityType.WATER,
                            quantity=drawn,
                            description=f"{v.name} drew {drawn:.1f} units of water from the Town Well.",
                        )
                    )

            elif v_act == VillageActionType.HARVEST_FIELD:
                field = self.fields["farm_0"]
                has_tools = inv.get(CommodityType.TOOLS) >= 1.0
                tool_bonus = 0.6 if has_tools else 0.0
                energy_cost = 1.5 if has_tools else 2.5
                harvested = min(3.2 + tool_bonus, field.crop_available)
                if harvested > 0.0:
                    self.fields["farm_0"] = field.model_copy(
                        update={"crop_available": round(field.crop_available - harvested, 4)}
                    )
                    inv = inv.with_delta(CommodityType.WHEAT, harvested)
                    energy = max(0.0, energy - energy_cost)
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category="work",
                            commodity=CommodityType.WHEAT,
                            quantity=harvested,
                            description=(
                                f"{v.name} harvested {harvested:.1f} wheat from the fields"
                                + (" using iron tools (-40% energy cost)." if has_tools else ".")
                            ),
                        )
                    )

            elif v_act == VillageActionType.MILL_WHEAT:
                wheat_in = min(2.0, inv.get(CommodityType.WHEAT))
                if wheat_in < 0.8 and self.granary.storage.get(CommodityType.WHEAT) >= 2.0:
                    # Miller draws grain allotment from Granary to mill into village flour & bread
                    wheat_in = 1.5
                    self.granary = self.granary.model_copy(
                        update={"storage": self.granary.storage.with_delta(CommodityType.WHEAT, -wheat_in)}
                    )
                else:
                    inv = inv.with_delta(CommodityType.WHEAT, -wheat_in)
                if wheat_in > 0.0:
                    flour_out = round(wheat_in * 1.1, 3)
                    bread_out = round(wheat_in * 1.25, 3)
                    inv = inv.with_delta(CommodityType.FLOUR, flour_out * 0.4)
                    inv = inv.with_delta(CommodityType.BREAD, bread_out)
                    self.granary = self.granary.model_copy(
                        update={"storage": self.granary.storage.with_delta(CommodityType.BREAD, round(bread_out * 0.5, 3))}
                    )
                    energy = max(0.0, energy - 1.8)
                    coins = round(coins + 1.8, 4)
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category="coop",
                            commodity=CommodityType.BREAD,
                            quantity=bread_out,
                            description=f"{v.name} milled {wheat_in:.1f} wheat into {flour_out:.1f} flour and baked {bread_out:.1f} loaves of bread.",
                        )
                    )

            elif v_act == VillageActionType.FORGE_TOOLS:
                timber_used = min(1.0, inv.get(CommodityType.TIMBER))
                if timber_used > 0.0:
                    inv = inv.with_delta(CommodityType.TIMBER, -timber_used)
                tools_forged = 1.5
                inv = inv.with_delta(CommodityType.TOOLS, tools_forged)
                energy = max(0.0, energy - 2.2)
                coins = round(coins + 2.2, 4)
                # Supply a forged tool to any farmer lacking one
                for f_id, f_v in updated_villagers.items():
                    if f_v.role == VillageRole.FARMER and f_v.inventory.get(CommodityType.TOOLS) < 1.0:
                        updated_villagers[f_id] = f_v.model_copy(
                            update={"inventory": f_v.inventory.with_delta(CommodityType.TOOLS, 1.0)}
                        )
                        break
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        commodity=CommodityType.TOOLS,
                        quantity=tools_forged,
                        description=f"Blacksmith {v.name} forged {tools_forged:.1f} iron tools at the Smithy anvil.",
                    )
                )

            elif v_act == VillageActionType.BREW_HEALING_BROTH:
                broth_qty = 1.5
                inv = inv.with_delta(CommodityType.HEALING_BROTH, broth_qty)
                energy = max(0.0, energy - 1.5)
                coins = round(coins + 1.6, 4)
                # Heal the most injured or fatigued villager
                healed_target: str | None = None
                for o_id, o_v in updated_villagers.items():
                    if o_v.is_alive and (o_v.health < 90.0 or o_v.hunger < 50.0):
                        updated_villagers[o_id] = o_v.model_copy(
                            update={
                                "health": min(100.0, round(o_v.health + 18.0, 4)),
                                "hunger": min(100.0, round(o_v.hunger + 16.0, 4)),
                            }
                        )
                        healed_target = o_id
                        break
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        target_id=healed_target,
                        action_type=v_act,
                        category="coop",
                        commodity=CommodityType.HEALING_BROTH,
                        quantity=broth_qty,
                        description=f"Herbalist {v.name} brewed medicinal healing broth from forest herbs.",
                    )
                )

            elif v_act in (VillageActionType.PRESIDE_SABHA, VillageActionType.ATTEND_SABHA):
                _, witnesses = self.compute_detection_probability(aid, pos, v_act)
                coins = round(coins + (1.2 if v_act == VillageActionType.PRESIDE_SABHA else 0.4), 4)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        quantity=1.0,
                        witness_ids=tuple(witnesses),
                        description=(
                            f"Magistrate {v.name} presided over the Gram Sabha Town Assembly."
                            if v_act == VillageActionType.PRESIDE_SABHA
                            else f"{v.name} attended the Gram Sabha council circle at Town Square."
                        ),
                    )
                )

            elif v_act == VillageActionType.SING_BALLAD:
                _, witnesses = self.compute_detection_probability(aid, pos, v_act)
                coins = round(coins + 1.5, 4)
                energy = min(v.max_energy, energy + 6.0)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        quantity=1.0,
                        witness_ids=tuple(witnesses),
                        description=f"Bard {v.name} performed a lute ballad chronicling village deeds.",
                    )
                )

            elif v_act == VillageActionType.INVESTIGATE_CRIME:
                coins = round(coins + 1.8, 4)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        quantity=1.0,
                        description=f"Guard {v.name} inspected forensic footprints and interviewed witnesses.",
                    )
                )

            elif v_act == VillageActionType.COMMUNITY_SERVICE:
                _, witnesses = self.compute_detection_probability(aid, pos, v_act)
                contrib = 1.5
                self.granary = self.granary.model_copy(
                    update={"storage": self.granary.storage.with_delta(CommodityType.WHEAT, contrib)}
                )
                tithes_tot = round(tithes_tot + contrib, 4)
                energy = max(0.0, energy - 2.0)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        commodity=CommodityType.WHEAT,
                        quantity=contrib,
                        witness_ids=tuple(witnesses),
                        description=f"{v.name} performed public community service, restoring {contrib:.1f} grain to the Commons.",
                    )
                )

            elif v_act == VillageActionType.COURTSHIP_BOND:
                _, witnesses = self.compute_detection_probability(aid, pos, v_act)
                energy = min(v.max_energy, energy + 5.0)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        quantity=1.0,
                        witness_ids=tuple(witnesses),
                        description=f"{v.name} shared courtship gifts and strengthened household kinship.",
                    )
                )

            elif v_act == VillageActionType.MOURN_AT_CHAPEL:
                energy = min(v.max_energy, energy + 8.0)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        quantity=1.0,
                        description=f"{v.name} lit a memorial candle at St. Aurelia Chapel.",
                    )
                )

            elif v_act == VillageActionType.FORAGE_FOREST:
                roll = float(self._forage_rng.uniform(1.2, 2.8))
                comm = CommodityType.BERRIES if roll < 2.2 else CommodityType.TIMBER
                inv = inv.with_delta(comm, round(roll, 3))
                energy = max(0.0, energy - 2.0)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="work",
                        commodity=comm,
                        quantity=round(roll, 3),
                        description=f"{v.name} foraged {roll:.1f} {comm.value} in the Wild Forest.",
                    )
                )

            elif v_act == VillageActionType.REST_HUT:
                hut = self.huts.get(v.home_hut_id)
                rec = hut.rest_energy_recovery if hut else 30.0
                energy = min(100.0, energy + rec)
                # Deposit surplus carried items into home chest
                if hut is not None:
                    chest_inv = hut.chest_inventory
                    for comm_t in (CommodityType.WHEAT, CommodityType.WATER, CommodityType.TIMBER):
                        surplus = max(0.0, inv.get(comm_t) - 4.0)
                        if surplus >= 1.0:
                            inv = inv.with_delta(comm_t, -surplus)
                            chest_inv = chest_inv.with_delta(comm_t, surplus)
                    self.huts[v.home_hut_id] = hut.model_copy(update={"chest_inventory": chest_inv})

            elif v_act == VillageActionType.CONSUME_RATION:
                consumed = False
                for comm_t in (CommodityType.WHEAT, CommodityType.BERRIES, CommodityType.FISH):
                    avail = inv.get(comm_t)
                    if avail > 0.2:
                        qty = min(1.5, avail)
                        inv = inv.with_delta(comm_t, -qty)
                        spec = COMMODITY_CATALOG[comm_t]
                        hunger = min(100.0, hunger + qty * spec.nutrition_value)
                        energy = min(100.0, energy + qty * spec.energy_boost)
                        consumed = True
                        break
                if not consumed and inv.get(CommodityType.WATER) > 0.2:
                    qty = min(1.5, inv.get(CommodityType.WATER))
                    inv = inv.with_delta(CommodityType.WATER, -qty)
                    hydration = min(100.0, hydration + qty * 35.0)

            elif v_act == VillageActionType.TITHE_GRANARY:
                tithe_qty = min(2.0, inv.get(CommodityType.WHEAT))
                if tithe_qty >= 0.5:
                    inv = inv.with_delta(CommodityType.WHEAT, -tithe_qty)
                    new_store = self.granary.storage.with_delta(CommodityType.WHEAT, tithe_qty)
                    new_tithes = dict(self.granary.tithes_by_agent)
                    new_tithes[aid] = round(new_tithes.get(aid, 0.0) + tithe_qty, 4)
                    self.granary = self.granary.model_copy(
                        update={"storage": new_store, "tithes_by_agent": new_tithes}
                    )
                    tithes_tot = round(tithes_tot + tithe_qty, 4)
                    _, witnesses = self.compute_detection_probability(aid, pos, v_act)
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category="coop",
                            commodity=CommodityType.WHEAT,
                            quantity=tithe_qty,
                            witness_ids=tuple(witnesses),
                            description=f"{v.name} donated {tithe_qty:.1f} wheat to the Granary Commons.",
                        )
                    )

            elif v_act == VillageActionType.CLAIM_FAMINE_RELIEF:
                avail_wheat = self.granary.storage.get(CommodityType.WHEAT)
                if hunger <= self.granary.famine_relief_threshold and avail_wheat >= 1.0:
                    relief = min(2.0, avail_wheat)
                    self.granary = self.granary.model_copy(
                        update={"storage": self.granary.storage.with_delta(CommodityType.WHEAT, -relief)}
                    )
                    hunger = min(100.0, hunger + relief * COMMODITY_CATALOG[CommodityType.WHEAT].nutrition_value)
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category="coop",
                            commodity=CommodityType.WHEAT,
                            quantity=relief,
                            description=f"{v.name} received {relief:.1f} famine relief wheat from the Granary.",
                        )
                    )

            elif v_act == VillageActionType.PATROL_GUARD:
                coins = round(coins + 1.5, 4)
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=aid,
                        action_type=v_act,
                        category="coop",
                        quantity=1.0,
                        description=f"Guard {v.name} patrolled the village perimeter.",
                    )
                )

            elif v_act == VillageActionType.TRADE_MARKET:
                # Submit fair bid or ask based on physiological need vs surplus
                wheat_p = self.market.get_price(CommodityType.WHEAT)
                water_p = self.market.get_price(CommodityType.WATER)
                if inv.get(CommodityType.WHEAT) > 3.0:
                    sell_q = min(2.5, inv.get(CommodityType.WHEAT) - 2.0)
                    asks.append(
                        MarketAsk(
                            seller_id=aid,
                            commodity=CommodityType.WHEAT,
                            quantity=sell_q,
                            min_unit_price=round(wheat_p * 0.92, 4),
                        )
                    )
                elif coins >= wheat_p and hunger < 75.0:
                    bids.append(
                        MarketBid(
                            buyer_id=aid,
                            commodity=CommodityType.WHEAT,
                            quantity=1.5,
                            max_unit_price=round(min(coins / 1.5, wheat_p * 1.15), 4),
                            buyer_status=v.societal_status,
                        )
                    )
                if inv.get(CommodityType.WATER) > 3.0:
                    sell_w = min(2.5, inv.get(CommodityType.WATER) - 2.0)
                    asks.append(
                        MarketAsk(
                            seller_id=aid,
                            commodity=CommodityType.WATER,
                            quantity=sell_w,
                            min_unit_price=round(water_p * 0.92, 4),
                        )
                    )
                elif coins >= water_p and hydration < 75.0:
                    bids.append(
                        MarketBid(
                            buyer_id=aid,
                            commodity=CommodityType.WATER,
                            quantity=1.5,
                            max_unit_price=round(min(coins / 1.5, water_p * 1.15), 4),
                            buyer_status=v.societal_status,
                        )
                    )

            # ==================================================================
            # UNETHICAL / CRIMINAL PATHWAYS (§2.3)
            # ==================================================================
            elif v_act in UNETHICAL_VILLAGE_ACTIONS:
                crimes_comm += 1
                p_detect, witnesses = self.compute_detection_probability(aid, pos, v_act)
                roll_det = float(self._detection_rng.uniform(0.0, 1.0))
                caught = roll_det < p_detect

                if v_act == VillageActionType.STEAL_GRANARY:
                    avail = self.granary.storage.get(CommodityType.WHEAT)
                    stolen = min(3.5, avail)
                    if caught:
                        crimes_det += 1
                        fine = min(coins, self.config.crime_coin_penalty)
                        coins = round(coins - fine, 4)
                        pillory = self.config.pillory_sentence_ticks
                        self.step_outcomes.append(
                            VillageInteractionOutcome(
                                step=self.step_count,
                                actor_id=aid,
                                action_type=v_act,
                                category="crime_caught",
                                commodity=CommodityType.WHEAT,
                                quantity=stolen,
                                witness_ids=tuple(witnesses),
                                detection_probability=p_detect,
                                detected=True,
                                coin_penalty=fine,
                                description=f"{v.name} was CAUGHT stealing wheat from the Granary (P_detect={p_detect:.2f})!",
                            )
                        )
                    else:
                        if stolen > 0.0:
                            self.granary = self.granary.model_copy(
                                update={"storage": self.granary.storage.with_delta(CommodityType.WHEAT, -stolen)}
                            )
                            inv = inv.with_delta(CommodityType.WHEAT, stolen)
                            hunger = min(100.0, hunger + 25.0)
                        self.step_outcomes.append(
                            VillageInteractionOutcome(
                                step=self.step_count,
                                actor_id=aid,
                                action_type=v_act,
                                category="crime_unseen",
                                commodity=CommodityType.WHEAT,
                                quantity=stolen,
                                witness_ids=tuple(witnesses),
                                detection_probability=p_detect,
                                detected=False,
                                description=f"{v.name} covertly pilfered {stolen:.1f} wheat from the Granary.",
                            )
                        )

                elif v_act == VillageActionType.STEAL_HUT_CHEST:
                    target_hut = next(
                        (
                            h
                            for h in self.huts.values()
                            if h.owner_id != aid and h.chest_inventory.get(CommodityType.WHEAT) > 0.5
                        ),
                        None,
                    )
                    target_owner = target_hut.owner_id if target_hut else None
                    stolen_q = (
                        min(2.5, target_hut.chest_inventory.get(CommodityType.WHEAT))
                        if target_hut
                        else 0.0
                    )
                    if caught:
                        crimes_det += 1
                        fine = min(coins, self.config.crime_coin_penalty)
                        coins = round(coins - fine, 4)
                        pillory = self.config.pillory_sentence_ticks
                        self.step_outcomes.append(
                            VillageInteractionOutcome(
                                step=self.step_count,
                                actor_id=aid,
                                target_id=target_owner,
                                action_type=v_act,
                                category="crime_caught",
                                commodity=CommodityType.WHEAT,
                                quantity=stolen_q,
                                witness_ids=tuple(witnesses),
                                detection_probability=p_detect,
                                detected=True,
                                coin_penalty=fine,
                                description=f"{v.name} was CAUGHT burglarizing a private hut chest!",
                            )
                        )
                    else:
                        if target_hut and stolen_q > 0.0:
                            self.huts[target_hut.hut_id] = target_hut.model_copy(
                                update={
                                    "chest_inventory": target_hut.chest_inventory.with_delta(
                                        CommodityType.WHEAT, -stolen_q
                                    )
                                }
                            )
                            inv = inv.with_delta(CommodityType.WHEAT, stolen_q)
                        self.step_outcomes.append(
                            VillageInteractionOutcome(
                                step=self.step_count,
                                actor_id=aid,
                                target_id=target_owner,
                                action_type=v_act,
                                category="crime_unseen",
                                commodity=CommodityType.WHEAT,
                                quantity=stolen_q,
                                witness_ids=tuple(witnesses),
                                detection_probability=p_detect,
                                detected=False,
                                description=f"{v.name} quietly stole {stolen_q:.1f} wheat from a neighbor's chest.",
                            )
                        )

                elif v_act in (VillageActionType.HOARD_AND_GOUGE, VillageActionType.FREE_RIDE_COMMONS):
                    wheat_p = self.market.get_price(CommodityType.WHEAT)
                    if inv.get(CommodityType.WHEAT) >= 1.0:
                        asks.append(
                            MarketAsk(
                                seller_id=aid,
                                commodity=CommodityType.WHEAT,
                                quantity=min(3.0, inv.get(CommodityType.WHEAT)),
                                min_unit_price=round(wheat_p * 1.65, 4),
                                is_price_gouging=True,
                            )
                        )
                    # Free-riding or price gouging is a social defection observed by witnesses
                    cat = "defect" if caught or witnesses else "crime_unseen"
                    if caught:
                        crimes_det += 1
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category=cat,
                            commodity=CommodityType.WHEAT,
                            quantity=1.5,
                            witness_ids=tuple(witnesses),
                            detection_probability=p_detect,
                            detected=caught,
                            description=f"{v.name} hoarded grain / price-gouged at 165% market rate.",
                        )
                    )

                elif v_act == VillageActionType.SABOTAGE_GRANARY:
                    avail = self.granary.storage.get(CommodityType.WHEAT)
                    spoiled = min(3.0, avail)
                    if caught:
                        crimes_det += 1
                        fine = min(coins, self.config.crime_coin_penalty * 1.2)
                        coins = round(coins - fine, 4)
                        pillory = self.config.pillory_sentence_ticks + 1
                        self.step_outcomes.append(
                            VillageInteractionOutcome(
                                step=self.step_count,
                                actor_id=aid,
                                action_type=v_act,
                                category="crime_caught",
                                commodity=CommodityType.WHEAT,
                                quantity=spoiled,
                                witness_ids=tuple(witnesses),
                                detection_probability=p_detect,
                                detected=True,
                                coin_penalty=fine,
                                description=f"{v.name} was CAUGHT attempting to sabotage the Granary stores!",
                            )
                        )
                    else:
                        if spoiled > 0.0:
                            self.granary = self.granary.model_copy(
                                update={"storage": self.granary.storage.with_delta(CommodityType.WHEAT, -spoiled)}
                            )
                        self.step_outcomes.append(
                            VillageInteractionOutcome(
                                step=self.step_count,
                                actor_id=aid,
                                action_type=v_act,
                                category="crime_unseen",
                                commodity=CommodityType.WHEAT,
                                quantity=spoiled,
                                witness_ids=tuple(witnesses),
                                detection_probability=p_detect,
                                detected=False,
                                description=f"{v.name} covertly sabotaged {spoiled:.1f} wheat in the Granary.",
                            )
                        )

                elif v_act == VillageActionType.OFFER_BRIBE:
                    bribe_amt = min(6.0, coins)
                    coins = round(max(0.0, coins - bribe_amt), 4)
                    if caught:
                        crimes_det += 1
                    self.step_outcomes.append(
                        VillageInteractionOutcome(
                            step=self.step_count,
                            actor_id=aid,
                            action_type=v_act,
                            category="crime_caught" if caught else "defect",
                            quantity=bribe_amt,
                            witness_ids=tuple(witnesses),
                            detection_probability=p_detect,
                            detected=caught,
                            coin_penalty=bribe_amt,
                            description=f"{v.name} attempted to bribe a watch guard with {bribe_amt:.1f} coins.",
                        )
                    )

            # Survival health check
            health = v.health
            starv_streak = v.starvation_streak
            if hunger <= 1.0 or hydration <= 1.0:
                health = max(0.0, health - 8.0)
                starv_streak += 1
            else:
                health = min(100.0, health + 1.0)
                starv_streak = 0
            alive = health > 0.0

            updated_villagers[aid] = v.model_copy(
                update={
                    "position": pos,
                    "hunger": round(hunger, 4),
                    "hydration": round(hydration, 4),
                    "energy": round(min(v.max_energy, energy), 4),
                    "health": round(health, 4),
                    "coins": round(coins, 4),
                    "inventory": inv,
                    "is_alive": alive,
                    "starvation_streak": starv_streak,
                    "in_pillory_ticks": pillory,
                    "crimes_committed": crimes_comm,
                    "crimes_detected": crimes_det,
                    "tithes_contributed": tithes_tot,
                    "last_action": v_act,
                    "current_goal_zone": goal_zone,
                }
            )

        self.villagers = updated_villagers

        # Phase 3: Clear Double Auction Market
        scarcity = {
            CommodityType.WHEAT: max(
                0.7,
                min(2.5, 80.0 / max(10.0, self.granary.storage.get(CommodityType.WHEAT))),
            ),
            CommodityType.WATER: max(
                0.7,
                min(2.5, 250.0 / max(20.0, self.well.water_level)),
            ),
        }
        trades: list[TradeExecution] = self.market.clear_market(
            step=self.step_count,
            bids=bids,
            asks=asks,
            scarcity_index=scarcity,
        )

        for tr in trades:
            buyer = self.villagers.get(tr.buyer_id)
            seller = self.villagers.get(tr.seller_id)
            if buyer is None or seller is None:
                continue
            if buyer.coins >= tr.total_coins and seller.inventory.get(tr.commodity) >= tr.quantity:
                self.villagers[tr.buyer_id] = buyer.model_copy(
                    update={
                        "coins": round(buyer.coins - tr.total_coins, 4),
                        "inventory": buyer.inventory.with_delta(tr.commodity, tr.quantity),
                    }
                )
                self.villagers[tr.seller_id] = seller.model_copy(
                    update={
                        "coins": round(seller.coins + tr.total_coins, 4),
                        "inventory": seller.inventory.with_delta(tr.commodity, -tr.quantity),
                    }
                )
                self.step_outcomes.append(
                    VillageInteractionOutcome(
                        step=self.step_count,
                        actor_id=tr.seller_id,
                        target_id=tr.buyer_id,
                        action_type=VillageActionType.TRADE_MARKET,
                        category="coop",
                        commodity=tr.commodity,
                        quantity=tr.quantity,
                        description=(
                            f"{seller.name} traded {tr.quantity:.1f} {tr.commodity.value} "
                            f"to {buyer.name} at {tr.clearing_unit_price:.2f} coins/unit."
                        ),
                    )
                )

        self.all_outcomes.extend(self.step_outcomes)
        self._world_state = self._build_world_state()
        return self._world_state

    def _build_world_state(self) -> WorldState:
        """Projects the VillageWorld into the canonical immutable WorldState contract."""
        w, h = self.config.width, self.config.height
        rows: list[tuple[CellState, ...]] = []
        for y in range(h):
            row_cells: list[CellState] = []
            for x in range(w):
                pos = Position(x, y)
                zone = self.get_zone_at(pos)
                terrain = TerrainType.GRASSLAND
                food = 0.0
                if zone is not None:
                    if zone.zone_type == ZoneType.WILD_FOREST:
                        terrain = TerrainType.FOREST
                        food = 5.0
                    elif zone.zone_type == ZoneType.AGRICULTURAL_FIELDS:
                        terrain = TerrainType.GRASSLAND
                        food = round(self.fields["farm_0"].crop_available / 10.0, 2)
                    elif zone.zone_type == ZoneType.TOWN_WELL:
                        terrain = TerrainType.GRASSLAND
                        food = round(self.well.water_level / 30.0, 2)
                row_cells.append(
                    CellState(
                        position=pos,
                        terrain=terrain,
                        food=food,
                        danger=0.0,
                        movement_cost=1.5 if terrain == TerrainType.FOREST else 1.0,
                    )
                )
            rows.append(tuple(row_cells))

        agents_map: dict[str, AgentState] = {}
        for aid, v in self.villagers.items():
            agents_map[aid] = AgentState(
                agent_id=aid,
                position=v.position,
                energy=v.energy,
                health=v.health,
                carried_food=round(v.inventory.total_edible_units(), 4),
                starvation_ticks=0 if v.hunger > 10.0 else 1,
                is_alive=v.is_alive,
            )

        return WorldState(
            step=self.step_count,
            width=w,
            height=h,
            grid=tuple(rows),
            agents=agents_map,
            weather={
                "time_of_day": self.time_of_day,
                "ambient_light": self.get_ambient_light(),
                "drought": 1.0 if self.drought_active else 0.0,
            },
        )

    def update_affective_and_lifecycle_states(
        self,
        pad_states: dict[str, tuple[str, tuple[float, float, float]]],
        lifecycle_updates: dict[str, tuple[int, float, str, int]],
    ) -> None:
        """Synchronizes affective PAD emotions and demographic lifecycle state onto villagers."""
        updated: dict[str, VillagerPhysicalState] = {}
        for aid, v in self.villagers.items():
            patch: dict[str, Any] = {}
            if aid in pad_states:
                dom_emotion, pad_vec = pad_states[aid]
                patch["dominant_emotion"] = dom_emotion
                patch["pad_vector"] = pad_vec
            if aid in lifecycle_updates:
                age_t, max_e, l_stage, gen = lifecycle_updates[aid]
                patch["age_ticks"] = age_t
                patch["max_energy"] = max_e
                patch["life_stage"] = l_stage
                patch["generation"] = gen
                if v.energy > max_e:
                    patch["energy"] = round(max_e, 4)
            updated[aid] = v.model_copy(update=patch) if patch else v
        self.villagers = updated
        self._world_state = self._build_world_state()

    def grid_to_3d_coords(self, pos: Position) -> tuple[float, float]:
        """Maps discrete $(x, y)$ village grid coordinates to 3D Cinematic scene $(X, Z)$."""
        zone = self.get_zone_at(pos)
        if zone is not None and pos == zone.center:
            return zone.world_3d_coords
        scale_x = (pos.x - self.config.width / 2.0) * 3.2
        scale_z = (pos.y - self.config.height / 2.0) * 3.2
        return (round(scale_x, 2), round(scale_z, 2))

    def export_village_world_json(self) -> dict[str, Any]:
        """Exports the village specification matching the Cinematic Edition schema."""
        return {
            "name": "village_world",
            "units": "metres",
            "size": 640,
            "ground_y": 0.3,
            "grid_dimensions": {"width": self.config.width, "height": self.config.height},
            "step": self.step_count,
            "time_of_day": self.time_of_day,
            "river": {"x_of_z": "70+10*sin(0.03*z)"},
            "zones": [
                {
                    "id": z.zone_id,
                    "type": z.zone_type.value,
                    "label": z.label,
                    "x": z.world_3d_coords[0],
                    "z": z.world_3d_coords[1],
                    "r": z.radius_3d,
                    "grid_center": [z.center.x, z.center.y],
                }
                for z in self.zones.values()
            ],
            "agents": [
                {
                    "agent_id": v.agent_id,
                    "name": v.name,
                    "role": v.role.value,
                    "archetype_title": v.archetype_title,
                    "grid_pos": [v.position.x, v.position.y],
                    "pos_3d": list(self.grid_to_3d_coords(v.position)),
                    "home": list(self.huts[v.home_hut_id].world_3d_coords)
                    if v.home_hut_id in self.huts
                    else [0.0, 0.0],
                    "hunger": round(v.hunger, 2),
                    "thirst": round(v.hydration, 2),
                    "energy": round(v.energy, 2),
                    "max_energy": round(v.max_energy, 2),
                    "status": round(v.societal_status * 100.0, 2),
                    "societal_status": round(v.societal_status, 4),
                    "coins": round(v.coins, 2),
                    "inventory": v.inventory.to_plain_dict(),
                    "last_action": v.last_action.value,
                    "age_ticks": v.age_ticks,
                    "generation": v.generation,
                    "life_stage": v.life_stage,
                    "dominant_emotion": v.dominant_emotion,
                    "pad_vector": list(v.pad_vector),
                }
                for v in self.villagers.values()
            ],
            "utility_needs": ["hunger", "thirst", "energy", "status"],
        }
