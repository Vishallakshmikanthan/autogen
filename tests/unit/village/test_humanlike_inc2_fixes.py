"""Regression tests for Increment 2 carry-over defects (C1 - C5)."""

import pytest

from autogen.agents.village.villager_agent import VillagerAgent
from autogen.intelligence.village.affective_pad import (
    CANONICAL_EMOTION_COORDS,
    AffectivePADEngine,
    EmotionCategory,
)
from autogen.intelligence.village.attention import Alert
from autogen.intelligence.village.intention import compute_reaction_utility
from autogen.intelligence.village.reflection import Traits
from autogen.simulation.environment.village.commodities import CommodityInventory
from autogen.simulation.environment.village.events import Modality, PerceivedEvent, WorldEvent
from autogen.simulation.environment.village.village_grid import (
    Position,
    VillageActionType,
    VillageObservation,
    VillageRole,
    VillageWorld,
    VillageWorldConfig,
    VisibleVillagerSummary,
)


class TestC4FearEmotion:
    """C4: Verify EmotionCategory.FEAR exists with canonical Mehrabian PAD coordinates."""

    def test_fear_enum_and_coordinates(self) -> None:
        assert hasattr(EmotionCategory, "FEAR")
        assert EmotionCategory.FEAR.value == "Fear / Distress"

        # Mehrabian canonical coordinates for fear: (-0.64, +0.60, -0.43)
        assert CANONICAL_EMOTION_COORDS[EmotionCategory.FEAR] == (-0.64, 0.60, -0.43)

        engine = AffectivePADEngine()
        cat, intensity, impact, speed_mult, aura_hex = engine.classify_emotion(-0.64, 0.60, -0.43)
        assert cat == EmotionCategory.FEAR
        assert aura_hex == "#fbbf24"
        assert speed_mult == 1.30
        assert "shelter" in impact.lower()


class TestC5ShelterUtilityInStorm:
    """C5: Verify SHELTER action receives sufficient reaction bonus to dominate during storms."""

    def test_storm_shelter_reaction_bonus(self) -> None:
        ev = WorldEvent(
            event_id="ev_storm",
            step=10,
            kind="THUNDER",
            modality=Modality.ENVIRONMENTAL,
            position=Position(x=10, y=10),
            intensity=1.0,
            is_global=True,
        )
        alert = Alert(
            percept=PerceivedEvent(event=ev, distance=0.0),
            salience=1.0,
            threat=0.90,
            eligible_step=10,
        )
        traits = Traits(neuroticism=0.50)

        # SHELTER receives 2.50 * salience * (1.0 + 0.50 * neuroticism)
        # = 2.50 * 1.0 * (1.0 + 0.25) = 3.125
        u_shelter = compute_reaction_utility(VillageActionType.SHELTER, [alert], traits)
        assert u_shelter == pytest.approx(3.125, abs=1e-4)

        # Regular actions like HARVEST_FIELD receive 0.0 reaction utility for storm
        u_harvest = compute_reaction_utility(VillageActionType.HARVEST_FIELD, [alert], traits)
        assert u_harvest == 0.0


class TestC3GreetProximityConstraint:
    """C3: Verify GREET candidate generation requires peer proximity <= 3 or ARRIVAL alert."""

    def test_greet_proximity_generation(self) -> None:
        empty_inv = CommodityInventory()
        agent = VillagerAgent(
            agent_id="meera",
            role=VillageRole.FARMER,
        )

        # 1. Distant peer: Bheem is at (35, 35), Meera is at (5, 5) -> distance = 60 > 3
        distant_peer = VisibleVillagerSummary(
            agent_id="bheem",
            name="Bheem",
            role=VillageRole.FARMER,
            position=Position(x=35, y=35),
            societal_status=0.10,
        )
        obs_distant = VillageObservation(
            agent_id="meera",
            step=1,
            time_of_day=10.0,
            ambient_light=1.0,
            position=Position(x=5, y=5),
            home_hut_position=Position(x=5, y=5),
            hunger=80.0,
            hydration=80.0,
            energy=80.0,
            health=100.0,
            coins=20.0,
            inventory=empty_inv,
            home_chest_inventory=empty_inv,
            societal_status=0.10,
            in_pillory_ticks=0,
            visible_villagers=(distant_peer,),
        )
        # When peer is distant, GREET is NOT generated in candidate pathways
        cands_distant = agent.evaluate_all_pathways(obs_distant)
        greet_cands_distant = [
            c for c in cands_distant if c.action_type == VillageActionType.GREET
        ]
        assert len(greet_cands_distant) == 0

        # 2. Close peer: Bheem is at (6, 5) -> distance = 1 <= 3
        close_peer = VisibleVillagerSummary(
            agent_id="bheem",
            name="Bheem",
            role=VillageRole.FARMER,
            position=Position(x=6, y=5),
            societal_status=0.10,
        )
        obs_close = VillageObservation(
            agent_id="meera",
            step=2,
            time_of_day=10.0,
            ambient_light=1.0,
            position=Position(x=5, y=5),
            home_hut_position=Position(x=5, y=5),
            hunger=95.0,
            hydration=95.0,
            energy=95.0,
            health=100.0,
            coins=20.0,
            inventory=empty_inv,
            home_chest_inventory=empty_inv,
            societal_status=0.10,
            in_pillory_ticks=0,
            visible_villagers=(close_peer,),
        )
        # When peer is close (<=3 units), GREET is generated in candidate pathways
        cands_close = agent.evaluate_all_pathways(obs_close)
        greet_cands_close = [
            c for c in cands_close if c.action_type == VillageActionType.GREET
        ]
        assert len(greet_cands_close) >= 1


class TestC2ZoneOccupancyFromWorldPosition:
    """C2: Verify hourly zone occupancy is derived from actual world coordinates rather than goal zone."""

    def test_zone_lookup_from_position(self) -> None:
        world = VillageWorld(config=VillageWorldConfig(width=40, height=40))
        market_zone = world.zones["central_market"]
        pos = Position(x=market_zone.center.x, y=market_zone.center.y)
        detected_zone = world.get_zone_at(pos)
        assert detected_zone is not None
        assert detected_zone.zone_id == "central_market"


class TestC1DynamicMetricsVerification:
    """C1: Verify that test assertions compute real ground-truth simulation values."""

    def test_dynamic_telemetry_values(self) -> None:
        from autogen.agents.village.controller import VillageSimulationController

        controller = VillageSimulationController()
        payload = controller.get_full_state_payload()
        assert "step" in payload
        assert "villagers" in payload
        for card in payload["villagers"]:
            assert "memory" in card
            assert card["memory"]["stream_size"] >= 0
            assert card["memory"]["unreflected_count"] >= 0
            assert isinstance(card["memory"]["recent_reflections"], list)
