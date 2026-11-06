"""Unit and mathematical oracle tests for Utility-Guided Intention Behavior Tree (§5.4)."""

from __future__ import annotations

from autogen.intelligence.village.bt import (
    arbitrate_intentions,
)
from autogen.intelligence.village.intention import (
    Intention,
    action_serves_intention,
)
from autogen.intelligence.village.needs import NeedState
from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights
from autogen.simulation.environment.village.village_grid import VillageActionType, VillageRole


def test_hysteresis_barrier_oracle() -> None:
    """Verifies §5 Oracle: A rival intention cannot trigger a switch unless Score(rival) > Score(cur) + H."""
    weights = VillagerUtilityWeights()
    need_st = NeedState()

    # Current intention is WORK, started at step 0
    cur_intention = Intention(
        kind="WORK",
        started_step=0,
        commitment=0.90,
        status="ACTIVE",
        max_duration=28,
    )

    # 1. At 12:00h, MARKET is rising, but if its advantage is less than H, WORK should be maintained
    # Evaluate arbitration
    selected_intention, scores = arbitrate_intentions(
        current_intention=cur_intention,
        current_step=5,
        need_state=need_st,
        weights=weights,
        role=VillageRole.FARMER,
        time_of_day=11.5,
        candidate_evals=[],
        switching_cost_scale=1.0,
        has_active_emergency=False,
    )

    # At 11:30h, Farmer's morning WORK score plus commitment resists premature switching
    assert selected_intention.kind == "WORK"

    # 2. At 13:00h, midday MARKET prior peaks strongly for Merchant; if rival exceeds by > H, switch occurs
    selected_merchant_intent, merchant_scores = arbitrate_intentions(
        current_intention=cur_intention,
        current_step=10,
        need_state=need_st,
        weights=weights,
        role=VillageRole.MERCHANT,
        time_of_day=13.0,
        candidate_evals=[],
        switching_cost_scale=1.0,
        has_active_emergency=False,
    )
    assert selected_merchant_intent.kind == "MARKET"


def test_intention_lifecycle_and_expiration_oracle() -> None:
    """Verifies §5 Oracle: Intentions terminate upon completion or exceeding max_duration."""
    intent = Intention(kind="SHELTER", started_step=10, max_duration=12, status="ACTIVE")

    # Not expired at step 20 (delta = 10 < 12)
    assert not intent.is_expired(20)
    assert not intent.is_terminal()

    # Expired at step 22 (delta = 12 >= 12)
    assert intent.is_expired(22)

    # Progress completion transitions status to DONE
    done_intent = intent.with_progress(1.0)
    assert done_intent.status == "DONE"
    assert done_intent.is_terminal()


def test_action_serves_intention_oracle() -> None:
    """Verifies §5 Oracle: Candidate action filtering aligns with active hierarchical intention."""
    food_intent = Intention(kind="SURVIVE_FOOD")
    work_intent = Intention(kind="WORK")
    social_intent = Intention(kind="SOCIALIZE")
    shelter_intent = Intention(kind="SHELTER")

    # Food intention allows consumption but not farming
    assert action_serves_intention(VillageActionType.CONSUME_RATION, food_intent)
    assert not action_serves_intention(VillageActionType.HARVEST_FIELD, food_intent)

    # Work intention allows role actions
    assert action_serves_intention(VillageActionType.HARVEST_FIELD, work_intent)
    assert not action_serves_intention(VillageActionType.SING_BALLAD, work_intent)

    # Social intention allows greetings and ballads
    assert action_serves_intention(VillageActionType.GREET, social_intent)
    assert action_serves_intention(VillageActionType.SING_BALLAD, social_intent)
    assert not action_serves_intention(VillageActionType.TRADE_MARKET, social_intent)

    # Shelter intention allows fleeing and resting
    assert action_serves_intention(VillageActionType.SHELTER, shelter_intent)
    assert action_serves_intention(VillageActionType.REST_HUT, shelter_intent)


def test_emergency_respond_preemption_oracle() -> None:
    """Verifies §5 Oracle: An active hazard alert preempts background routines into RESPOND."""
    cur_intent = Intention(kind="WORK", commitment=0.99)
    intent, _ = arbitrate_intentions(
        current_intention=cur_intent,
        current_step=4,
        need_state=NeedState(),
        weights=VillagerUtilityWeights(),
        role=VillageRole.FARMER,
        time_of_day=9.0,
        candidate_evals=[],
        switching_cost_scale=1.0,
        has_active_emergency=True,
    )
    assert intent.kind == "RESPOND"
    assert intent.commitment >= 0.95
