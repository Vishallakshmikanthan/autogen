"""Dynamic Needs and Prepotency Hierarchy (Maslow-Style Soft Gating) (§5.3).

Implements the 5-level hierarchical drive/needs model:
- L1: Physiological (hunger, hydration, energy, health)
- L2: Safety (shelter, stored food, threat absence, legal standing)
- L3: Belonging / Loneliness Drive (social contact, conversation quality)
- L4: Esteem (societal status, respect, standing)
- L5: Growth / Mastery (craft productivity, exploration)

Computes deficits $d_k \\in [0, 1]$, convex pressures $P_k = d_k^p$, soft prepotency
gates $g_k = \\prod_{j<k} \\text{sat}_j^\\beta$, and dynamic utility weights $w_k(t)$.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.personality import Traits
from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights
from autogen.simulation.environment.village.commodities import CommodityType
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageObservation,
    ZoneType,
)

P_EXPONENT: float = 2.0
BETA_GATE: float = 0.70
LAMBDA_WEIGHT: float = 2.0
SATIATION_DECAY: float = 0.85
SATIATION_BOOST: float = 0.25


class NeedState(BaseModel):
    """Internal drive and deficiency state for villager $i$ (§5.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    d_phys: float = Field(default=0.0, ge=0.0, le=1.0, description="L1 Physiological deficit")
    d_safety: float = Field(default=0.0, ge=0.0, le=1.0, description="L2 Safety deficit")
    d_belonging: float = Field(default=0.0, ge=0.0, le=1.0, description="L3 Belonging / Loneliness deficit")
    d_esteem: float = Field(default=0.0, ge=0.0, le=1.0, description="L4 Esteem deficit")
    d_growth: float = Field(default=0.0, ge=0.0, le=1.0, description="L5 Growth / Mastery deficit")

    p_phys: float = Field(default=0.0, ge=0.0, description="L1 Pressure P1 = d1^p")
    p_safety: float = Field(default=0.0, ge=0.0, description="L2 Pressure P2 = d2^p")
    p_belonging: float = Field(default=0.0, ge=0.0, description="L3 Pressure P3 = d3^p")
    p_esteem: float = Field(default=0.0, ge=0.0, description="L4 Pressure P4 = d4^p")
    p_growth: float = Field(default=0.0, ge=0.0, description="L5 Pressure P5 = d5^p")

    g_phys: float = Field(default=1.0, ge=0.0, le=1.0, description="L1 Prepotency gate g1 = 1.0")
    g_safety: float = Field(default=1.0, ge=0.0, le=1.0, description="L2 Prepotency gate g2")
    g_belonging: float = Field(default=1.0, ge=0.0, le=1.0, description="L3 Prepotency gate g3")
    g_esteem: float = Field(default=1.0, ge=0.0, le=1.0, description="L4 Prepotency gate g4")
    g_growth: float = Field(default=1.0, ge=0.0, le=1.0, description="L5 Prepotency gate g5")

    contact_score: float = Field(default=1.0, ge=0.0, description="Decayed contact accumulator c")
    mastery_score: float = Field(default=1.0, ge=0.0, description="Decayed role-mastery accumulator m")
    satiation: dict[str, float] = Field(default_factory=dict, description="Per-action satiation s_a in [0, 1]")

    @property
    def deficits_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.d_phys, self.d_safety, self.d_belonging, self.d_esteem, self.d_growth)

    @property
    def gates_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.g_phys, self.g_safety, self.g_belonging, self.g_esteem, self.g_growth)

    @property
    def pressures_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.p_phys, self.p_safety, self.p_belonging, self.p_esteem, self.p_growth)


def compute_deficits(
    obs: VillageObservation,
    contact_score: float,
    mastery_score: float,
    traits: Traits,
    max_village_status: float = 1.0,
    max_threat: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Computes deficits $d_1 \\dots d_5 \\in [0, 1]$ from observation and accumulated drives (§5.3)."""
    # L1: Physiological bottleneck: min of hunger, hydration, energy, health (all in [0, 100])
    bottleneck = min(obs.hunger, obs.hydration, obs.energy, obs.health)
    d1 = float(np.clip(1.0 - bottleneck / 100.0, 0.0, 1.0))

    # L2: Safety deficit: exposed to storm + lack of reserve food stock + external threat
    cur_zone = obs.current_zone
    is_indoor = cur_zone is not None and cur_zone in (ZoneType.PRIVATE_HUTS, ZoneType.CHAPEL)
    is_raining = any(p.event.kind in ("THUNDER", "RAIN_ONSET") for p in obs.percepts)
    exposed_in_storm = 1.0 if (is_raining and not is_indoor) else 0.0
    food_stock = (
        obs.inventory.get(CommodityType.WHEAT)
        + obs.home_chest_inventory.get(CommodityType.WHEAT)
        + obs.granary_wheat_available
    )
    food_reserve_factor = 1.0 - min(1.0, food_stock / 6.0)
    d2 = float(
        np.clip(
            0.50 * exposed_in_storm + 0.30 * food_reserve_factor + 0.20 * min(1.0, max_threat),
            0.0,
            1.0,
        )
    )

    # L3: Belonging / Loneliness deficit: c* = 2.0 * (0.5 + Extraversion)
    c_target = max(0.1, 2.0 * (0.50 + traits.extraversion))
    d3 = float(np.clip(1.0 - (contact_score / c_target), 0.0, 1.0))

    # L4: Esteem deficit: relative status deficit compared to top peer
    status_denom = max(0.01, max_village_status)
    d4 = float(np.clip(1.0 - (obs.societal_status / status_denom), 0.0, 1.0))

    # L5: Growth deficit: m* = 3.0 * (0.5 + Openness)
    m_target = max(0.1, 3.0 * (0.50 + traits.openness))
    d5 = float(np.clip(1.0 - (mastery_score / m_target), 0.0, 1.0))

    return (
        round(d1, 4),
        round(d2, 4),
        round(d3, 4),
        round(d4, 4),
        round(d5, 4),
    )


def compute_pressures(
    deficits: tuple[float, float, float, float, float],
    p: float = P_EXPONENT,
) -> tuple[float, float, float, float, float]:
    """Computes convex drive pressures $P_k = d_k^p$ (§5.3)."""
    return tuple(round(float(d**p), 4) for d in deficits)  # type: ignore[return-value]


def compute_prepotency_gates(
    deficits: tuple[float, float, float, float, float],
    beta: float = BETA_GATE,
) -> tuple[float, float, float, float, float]:
    """Computes soft prepotency gates $g_1=1, g_k = \\prod_{j < k} \\text{sat}_j^\\beta$ with $\\text{sat}_j = 1 - d_j$ (§5.3)."""
    gates = [1.0]
    running_product = 1.0
    for j in range(len(deficits) - 1):
        sat_j = max(0.0, min(1.0, 1.0 - deficits[j]))
        running_product *= sat_j**beta
        gates.append(round(running_product, 4))
    return tuple(gates)  # type: ignore[return-value]


def compute_dynamic_weights(
    base_weights: VillagerUtilityWeights,
    deficits: tuple[float, float, float, float, float],
    gates: tuple[float, float, float, float, float],
    pressures: tuple[float, float, float, float, float],
    lambda_param: float = LAMBDA_WEIGHT,
) -> VillagerUtilityWeights:
    """Updates baseline utility weights with dynamic drive pressures and prepotency gates (§5.3).

    $$w_{phys}(t) = \\text{base}_{phys} \\cdot (1 + \\lambda P_1)$$
    $$w_{soc}(t) = \\text{base}_{soc} \\cdot \\frac{1}{2} \\left[(1 + \\lambda P_3) g_3 + (1 + \\lambda P_4) g_4\\right]$$
    """
    p1, _p2, p3, p4, _p5 = pressures
    _g1, _g2, g3, g4, _g5 = gates

    # L1 modulates w_phys
    new_w_phys = base_weights.w_phys * (1.0 + lambda_param * p1)

    # L3 & L4 modulate w_soc through prepotency gates g3 & g4
    social_factor = 0.50 * ((1.0 + lambda_param * p3) * g3 + (1.0 + lambda_param * p4) * g4)
    new_w_soc = base_weights.w_soc * social_factor

    # Catastrophic crisis penalty increases with physiological pressure
    new_kappa = base_weights.crisis_penalty_kappa * (1.0 + 0.50 * p1)

    return base_weights.model_copy(
        update={
            "w_phys": round(new_w_phys, 4),
            "w_soc": round(new_w_soc, 4),
            "crisis_penalty_kappa": round(new_kappa, 4),
        }
    )


def update_satiation(
    current_satiation: dict[str, float],
    executed_action: VillageActionType | None,
    decay: float = SATIATION_DECAY,
    boost: float = SATIATION_BOOST,
) -> dict[str, float]:
    """Updates per-action satiation tracker $s_a$ (§5.3):
    $s_a \\leftarrow s_a + boost \\cdot (1 - s_a)$ on use, else $s_a \\leftarrow decay \\cdot s_a$.
    """
    updated: dict[str, float] = {}
    act_name = executed_action.value if executed_action else None

    # Decay all known actions
    for name, val in current_satiation.items():
        if name != act_name:
            new_val = val * decay
            if new_val >= 0.01:
                updated[name] = round(new_val, 4)

    # Boost executed action if present
    if act_name:
        prev = current_satiation.get(act_name, 0.0)
        boosted = prev + boost * (1.0 - prev)
        updated[act_name] = round(boosted, 4)

    return updated


class NeedsEngine:
    """Stateful coordinator maintaining a villager's need state, dynamic weights, and satiation."""

    def __init__(self, seed: int = 42) -> None:
        self.state = NeedState()

    def step(
        self,
        obs: VillageObservation,
        traits: Traits,
        contact_quality: float = 0.0,
        executed_action: VillageActionType | None = None,
        max_village_status: float = 1.0,
        max_threat: float = 0.0,
    ) -> NeedState:
        """Executes one simulation step update of the needs and drives model."""
        # 1. Update contact score c <- 0.90 * c + q_t
        new_contact = max(0.0, self.state.contact_score * 0.90 + contact_quality)
        if new_contact < 0.05:
            new_contact = 0.0

        # 2. Update mastery score m <- 0.98 * m + 1[role productive]
        is_productive = executed_action in (
            VillageActionType.HARVEST_FIELD,
            VillageActionType.FORAGE_FOREST,
            VillageActionType.DRAW_WATER,
            VillageActionType.TRADE_MARKET,
            VillageActionType.TITHE_GRANARY,
            VillageActionType.PATROL_GUARD,
            VillageActionType.MILL_WHEAT,
            VillageActionType.FORGE_TOOLS,
            VillageActionType.BREW_HEALING_BROTH,
        )
        new_mastery = max(0.0, self.state.mastery_score * 0.98 + (1.0 if is_productive else 0.0))

        # 3. Compute deficits
        d1, d2, d3, d4, d5 = compute_deficits(
            obs=obs,
            contact_score=new_contact,
            mastery_score=new_mastery,
            traits=traits,
            max_village_status=max_village_status,
            max_threat=max_threat,
        )
        deficits = (d1, d2, d3, d4, d5)

        # 4. Compute pressures and prepotency gates
        pressures = compute_pressures(deficits)
        gates = compute_prepotency_gates(deficits)

        # 5. Update action satiation tracker
        satiation = update_satiation(self.state.satiation, executed_action)

        self.state = NeedState(
            d_phys=d1,
            d_safety=d2,
            d_belonging=d3,
            d_esteem=d4,
            d_growth=d5,
            p_phys=pressures[0],
            p_safety=pressures[1],
            p_belonging=pressures[2],
            p_esteem=pressures[3],
            p_growth=pressures[4],
            g_phys=gates[0],
            g_safety=gates[1],
            g_belonging=gates[2],
            g_esteem=gates[3],
            g_growth=gates[4],
            contact_score=round(new_contact, 4),
            mastery_score=round(new_mastery, 4),
            satiation=satiation,
        )
        return self.state

    def get_dynamic_weights(
        self,
        base_weights: VillagerUtilityWeights,
        lambda_param: float = LAMBDA_WEIGHT,
    ) -> VillagerUtilityWeights:
        """Derives modulated utility weights based on current need state."""
        return compute_dynamic_weights(
            base_weights=base_weights,
            deficits=self.state.deficits_tuple,
            gates=self.state.gates_tuple,
            pressures=self.state.pressures_tuple,
            lambda_param=lambda_param,
        )
