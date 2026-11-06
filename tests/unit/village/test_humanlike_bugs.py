"""Regression tests verifying fixes for Bug 1 and Bug 2 in VillagerAgent (§0, §7)."""

from __future__ import annotations

from autogen.agents.village.villager_agent import VillagerAgent
from autogen.simulation.environment.village.commodities import CommodityInventory
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageInteractionOutcome,
    VillageObservation,
    VillageRole,
)
from autogen.simulation.state.types import Position


def test_bug1_affective_state_preserves_agent_id_and_not_overwritten() -> None:
    """Verifies Bug 1 Fix: self.affective_state is NOT overwritten and retains agent_id."""
    agent = VillagerAgent(agent_id="villager_arun", role=VillageRole.FARMER)
    assert agent.affective_state.agent_id == "villager_arun"

    # Verify reset() preserves agent_id as well
    agent.reset()
    assert agent.affective_state.agent_id == "villager_arun"


def test_bug2_memory_deduplication_prevents_duplicate_entries() -> None:
    """Verifies Bug 2 Fix: Duplicate outcome descriptions do not accumulate in episodic_memories."""
    agent = VillagerAgent(agent_id="villager_arun", role=VillageRole.FARMER)

    outcome = VillageInteractionOutcome(
        step=5,
        actor_id="villager_bheem",
        action_type=VillageActionType.TRADE_MARKET,
        category="trade",
        description="Bheem traded wheat for water.",
    )

    empty_inv = CommodityInventory()
    obs = VillageObservation(
        agent_id="villager_arun",
        step=5,
        time_of_day=10.0,
        ambient_light=1.0,
        position=Position(x=10, y=10),
        home_hut_position=Position(x=10, y=10),
        hunger=80.0,
        hydration=80.0,
        energy=80.0,
        health=100.0,
        coins=20.0,
        inventory=empty_inv,
        home_chest_inventory=empty_inv,
        societal_status=0.10,
        in_pillory_ticks=0,
        witnessed_outcomes=(outcome,),
    )

    # Ingest observation step 5
    agent.observe_village(obs)
    agent.update_belief()
    assert len(agent.belief_state.episodic_memories) == 1
    assert "Bheem traded wheat for water." in agent.belief_state.episodic_memories[0]

    # Ingest again at step 6 with the same description
    obs2 = obs.model_copy(update={"step": 6})
    agent.observe_village(obs2)
    agent.update_belief()

    # Must NOT create a duplicate entry for the exact same event description
    assert len(agent.belief_state.episodic_memories) == 1
