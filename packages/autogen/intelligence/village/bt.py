"""Utility-Guided Behavior Tree for Hierarchical Intentions (§5.4).

Provides multi-step sequential intention pipelines:
- SURVIVE_FOOD / SURVIVE_WATER: Immediate physiological relief sequences
- WORK: Role-specific occupational routines (farming, milling, guarding, carrying)
- MARKET: Midday commerce and double auctions
- SOCIALIZE: Neighbor approach, greetings, companionship
- REST: Return to private hut for homeostatic recovery and reflection
- RESPOND: High-urgency alert mitigation (shelter, crime/noise investigation)

Arbitrates intentions using need pressures, candidate utilities, circadian priors,
and hysteresis switching barriers to prevent erratic single-tick thrashing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.intention import (
    Intention,
)
from autogen.intelligence.village.needs import NeedState
from autogen.intelligence.village.routines import compute_circadian_priors
from autogen.intelligence.village.utility_ethics import (
    ActionUtilityEvaluation,
    VillagerUtilityWeights,
)
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageRole,
)


class IntentionStep(BaseModel):
    """A discrete sequential milestone within a hierarchical intention pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    target_zone: str
    allowed_actions: tuple[VillageActionType, ...]
    progress_increment: float = 0.50


class IntentionDescriptor(BaseModel):
    """Specification defining expected deficit reduction, steps, and parameters for an intention kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    steps: tuple[IntentionStep, ...]
    deficit_deltas: tuple[float, float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0),
        description="(Delta d_phys, Delta d_safety, Delta d_belonging, Delta d_esteem, Delta d_growth)",
    )
    max_duration: int = 24
    base_priority: float = 0.0


# Catalog of Hierarchical Intentions (§5.4)
INTENTION_SPECS: dict[str, IntentionDescriptor] = {
    "SURVIVE_FOOD": IntentionDescriptor(
        kind="SURVIVE_FOOD",
        deficit_deltas=(0.70, 0.20, 0.0, 0.0, 0.0),
        max_duration=16,
        base_priority=1.5,
        steps=(
            IntentionStep(
                name="approach_food",
                target_zone="self",
                allowed_actions=(
                    VillageActionType.CONSUME_RATION,
                    VillageActionType.CLAIM_FAMINE_RELIEF,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
            IntentionStep(
                name="consume_food",
                target_zone="self",
                allowed_actions=(
                    VillageActionType.CONSUME_RATION,
                    VillageActionType.CLAIM_FAMINE_RELIEF,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
        ),
    ),
    "SURVIVE_WATER": IntentionDescriptor(
        kind="SURVIVE_WATER",
        deficit_deltas=(0.75, 0.10, 0.0, 0.0, 0.0),
        max_duration=16,
        base_priority=1.8,
        steps=(
            IntentionStep(
                name="approach_well",
                target_zone="town_well",
                allowed_actions=(
                    VillageActionType.DRAW_WATER,
                    VillageActionType.CONSUME_RATION,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
            IntentionStep(
                name="draw_and_drink",
                target_zone="town_well",
                allowed_actions=(
                    VillageActionType.DRAW_WATER,
                    VillageActionType.CONSUME_RATION,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
        ),
    ),
    "WORK": IntentionDescriptor(
        kind="WORK",
        deficit_deltas=(0.0, 0.15, 0.0, 0.25, 0.60),
        max_duration=28,
        base_priority=1.0,
        steps=(
            IntentionStep(
                name="perform_occupation",
                target_zone="work_zone",
                allowed_actions=(
                    VillageActionType.HARVEST_FIELD,
                    VillageActionType.FORAGE_FOREST,
                    VillageActionType.DRAW_WATER,
                    VillageActionType.MILL_WHEAT,
                    VillageActionType.FORGE_TOOLS,
                    VillageActionType.BREW_HEALING_BROTH,
                    VillageActionType.PATROL_GUARD,
                    VillageActionType.TITHE_GRANARY,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.34,
            ),
        ),
    ),
    "MARKET": IntentionDescriptor(
        kind="MARKET",
        deficit_deltas=(0.0, 0.20, 0.15, 0.35, 0.30),
        max_duration=20,
        base_priority=0.8,
        steps=(
            IntentionStep(
                name="trade_in_market",
                target_zone="market_square",
                allowed_actions=(
                    VillageActionType.TRADE_MARKET,
                    VillageActionType.HOARD_AND_GOUGE,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
        ),
    ),
    "SOCIALIZE": IntentionDescriptor(
        kind="SOCIALIZE",
        deficit_deltas=(0.0, 0.0, 0.70, 0.20, 0.10),
        max_duration=18,
        base_priority=0.9,
        steps=(
            IntentionStep(
                name="greet_and_converse",
                target_zone="commons",
                allowed_actions=(
                    VillageActionType.GREET,
                    VillageActionType.SING_BALLAD,
                    VillageActionType.COURTSHIP_BOND,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
        ),
    ),
    "REST": IntentionDescriptor(
        kind="REST",
        deficit_deltas=(0.35, 0.40, 0.0, 0.0, 0.0),
        max_duration=32,
        base_priority=1.1,
        steps=(
            IntentionStep(
                name="sleep_in_hut",
                target_zone="private_huts",
                allowed_actions=(
                    VillageActionType.REST_HUT,
                    VillageActionType.MOURN_AT_CHAPEL,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.34,
            ),
        ),
    ),
    "SHELTER": IntentionDescriptor(
        kind="SHELTER",
        deficit_deltas=(0.0, 0.90, 0.0, 0.0, 0.0),
        max_duration=14,
        base_priority=2.5,
        steps=(
            IntentionStep(
                name="take_shelter",
                target_zone="shelter",
                allowed_actions=(
                    VillageActionType.SHELTER,
                    VillageActionType.REST_HUT,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
        ),
    ),
    "RESPOND": IntentionDescriptor(
        kind="RESPOND",
        deficit_deltas=(0.10, 0.85, 0.0, 0.0, 0.0),
        max_duration=12,
        base_priority=2.0,
        steps=(
            IntentionStep(
                name="mitigate_hazard",
                target_zone="shelter",
                allowed_actions=(
                    VillageActionType.SHELTER,
                    VillageActionType.INVESTIGATE_NOISE,
                    VillageActionType.INVESTIGATE_CRIME,
                    VillageActionType.MOVE_NORTH,
                    VillageActionType.MOVE_SOUTH,
                    VillageActionType.MOVE_EAST,
                    VillageActionType.MOVE_WEST,
                    VillageActionType.STAY,
                ),
                progress_increment=0.50,
            ),
        ),
    ),
}


def score_intention(
    spec: IntentionDescriptor,
    need_state: NeedState,
    weights: VillagerUtilityWeights,
    circadian_prior: float,
    candidate_evals: list[ActionUtilityEvaluation],
    is_current: bool,
    switching_cost_scale: float,
) -> float:
    """Computes deliberative score for an intention candidate (§5.4):

    $$\\text{Score}(I) = \\sum_k w_k(t) \\cdot \\Delta d_k(I) + U_{\\text{first}}(I) + \\pi_I(\\tau) - C_{\\text{switch}} \\cdot \\mathbb{1}[I \\neq I_{\\text{cur}}]$$
    """
    # 1. Need deficit reduction score
    # Deficit reduction value weighted by current dynamic need pressures
    w_vec = (
        weights.w_phys,
        3.0 * need_state.g_safety * (1.0 + 2.0 * need_state.p_safety),
        weights.w_soc * 0.50,
        weights.w_soc * 0.50,
        2.0 * need_state.g_growth * (1.0 + 2.0 * need_state.p_growth),
    )
    need_gain = sum(w * delta for w, delta in zip(w_vec, spec.deficit_deltas, strict=True))

    # 2. Candidate action expected utility
    allowed_acts: set[VillageActionType] = set()
    for st in spec.steps:
        allowed_acts.update(st.allowed_actions)

    valid_cands = [c for c in candidate_evals if c.action_type in allowed_acts]
    u_first = valid_cands[0].expected_total_utility if valid_cands else 0.0

    # 3. Switching penalty if changing intention
    c_switch = 0.0 if is_current else (1.50 * switching_cost_scale)

    total_score = spec.base_priority + need_gain + (0.35 * u_first) + circadian_prior - c_switch
    return float(round(total_score, 4))


def arbitrate_intentions(
    current_intention: Intention | None,
    current_step: int,
    need_state: NeedState,
    weights: VillagerUtilityWeights,
    role: VillageRole,
    time_of_day: float,
    candidate_evals: list[ActionUtilityEvaluation],
    switching_cost_scale: float,
    has_active_emergency: bool = False,
) -> tuple[Intention, dict[str, float]]:
    """Selects the active intention using utility evaluation and hysteresis barriers (§5.4)."""
    circadian_map = compute_circadian_priors(role, time_of_day)

    # If currently actively sheltering during emergency, preserve shelter
    if (
        current_intention is not None
        and current_intention.kind == "SHELTER"
        and not current_intention.is_terminal()
        and not current_intention.is_expired(current_step)
    ):
        return current_intention, {"SHELTER": 99.0}

    # If severe crisis or alert, emergency RESPOND overrides background routines
    if has_active_emergency:
        spec = INTENTION_SPECS["RESPOND"]
        return (
            Intention(
                kind="RESPOND",
                started_step=current_step,
                commitment=0.95,
                max_duration=spec.max_duration,
                status="ACTIVE",
            ),
            {"RESPOND": 99.0},
        )

    # Score all candidate intentions
    scores: dict[str, float] = {}
    for kind, spec in INTENTION_SPECS.items():
        if kind == "RESPOND":
            continue
        is_cur = current_intention is not None and current_intention.kind == kind
        c_prior = circadian_map.get(kind, 0.0)
        scores[kind] = score_intention(
            spec=spec,
            need_state=need_state,
            weights=weights,
            circadian_prior=c_prior,
            candidate_evals=candidate_evals,
            is_current=is_cur,
            switching_cost_scale=switching_cost_scale,
        )

    # Determine best ranked intention
    best_kind = max(scores.keys(), key=lambda k: scores[k])
    best_score = scores[best_kind]

    # Evaluate whether to maintain current intention or switch
    if (
        current_intention is not None
        and not current_intention.is_terminal()
        and not current_intention.is_expired(current_step)
        and current_intention.kind in scores
    ):
        cur_score = scores[current_intention.kind]
        # Hysteresis switching check (§5.4): rival must exceed current + H
        h_threshold = 0.10 * abs(cur_score) + 0.05
        if best_score <= (cur_score + h_threshold):
            # Retain current intention with decayed commitment
            maintained = current_intention.with_decayed_commitment(0.99)
            return maintained.model_copy(update={"score": cur_score}), scores

    # Adopt new intention
    spec = INTENTION_SPECS.get(best_kind, INTENTION_SPECS["WORK"])
    new_intention = Intention(
        kind=best_kind,
        started_step=current_step,
        commitment=0.90,
        max_duration=spec.max_duration,
        status="ACTIVE",
        score=best_score,
    )
    return new_intention, scores
