"""Multi-Objective Social Utility & Rational Risk-Reward Ethics Engine (§2.1, §2.3)."""

from __future__ import annotations

import math
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.environment.village.commodities import CommodityInventory, CommodityType
from autogen.simulation.environment.village.village_grid import (
    UNETHICAL_VILLAGE_ACTIONS,
    VillageActionType,
)


class VillagerUtilityWeights(BaseModel):
    """Individual preference and risk-tolerance parameters for villager $i$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    w_phys: float = Field(default=2.2, gt=0.0, description="Weight on physiological survival")
    w_econ: float = Field(default=1.0, ge=0.0, description="Weight on wealth accumulation")
    w_soc: float = Field(default=14.0, ge=0.0, description="Weight on societal status S_i(t)")
    theta_risk: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Individual risk tolerance in [0, 1] (higher = less deterred by sanction risk)",
    )
    crisis_threshold_tau: float = Field(
        default=20.0,
        ge=1.0,
        le=60.0,
        description="Critical physiological depletion threshold tau",
    )
    crisis_penalty_kappa: float = Field(
        default=18.0,
        ge=0.0,
        description="Catastrophic starvation/dehydration penalty kappa",
    )
    epsilon: float = Field(default=1.0, gt=0.0, description="Logarithmic smoothing constant")


class ActionUtilityEvaluation(BaseModel):
    """Detailed mathematical breakdown of expected utility for a candidate action $a$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_type: VillageActionType
    is_unethical: bool
    u_phys_current: float
    u_phys_projected: float
    u_econ_projected: float
    u_soc_projected: float
    u_gain: float
    detection_probability: float = Field(ge=0.0, le=1.0)
    expected_risk_penalty: float = Field(ge=0.0)
    expected_total_utility: float
    target_zone: str = ""
    rationale_note: str = ""


class UtilityEngine:
    """From-scratch mathematical engine evaluating physiological, economic, social, and ethical utility."""

    def __init__(self, weights: VillagerUtilityWeights | None = None) -> None:
        self.weights = weights or VillagerUtilityWeights()

    def compute_physiological_utility(
        self,
        hunger: float,
        hydration: float,
        energy: float,
    ) -> float:
        """Computes concave physiological survival utility (§2.1):

        $$U_{\\text{phys}}(H_i, T_i, E_i) = \\ln(H_i + \\epsilon) + \\ln(T_i + \\epsilon) + \\ln(E_i + \\epsilon) - \\kappa \\cdot \\mathbb{I}_{[\\min(H_i, T_i, E_i) < \\tau]}$$
        """
        h = max(0.0, float(hunger))
        t = max(0.0, float(hydration))
        e = max(0.0, float(energy))
        eps = self.weights.epsilon

        log_sum = math.log(h + eps) + math.log(t + eps) + math.log(e + eps)
        min_vital = min(h, t, e)
        crisis_indicator = 1.0 if min_vital < self.weights.crisis_threshold_tau else 0.0
        return float(log_sum - self.weights.crisis_penalty_kappa * crisis_indicator)

    def compute_economic_utility(
        self,
        coins: float,
        inventory: CommodityInventory,
        clearing_prices: dict[CommodityType, float],
    ) -> float:
        """Computes concave wealth accumulation utility (§2.1):

        $$U_{\\text{econ}}(W_i) = \\sqrt{C_i + \\sum_{k} p_k \\cdot I_{i, k}}$$
        """
        total_wealth = max(0.0, float(coins)) + inventory.total_valuation(clearing_prices)
        return float(math.sqrt(total_wealth))

    def compute_risk_penalty(
        self,
        action_type: VillageActionType,
        detection_probability: float,
        coin_penalty: float = 12.0,
        status_loss: float = 0.08,
    ) -> float:
        """Computes expected sanction and status loss penalty $\\mathcal{R}_i(a)$ (§2.3):

        $$\\mathcal{R}_i(a) = P_{\\text{detect}}(a, \\mathbf{x}_i, \\text{env}) \\cdot (1 - 0.65 \\, \\theta_{\\text{risk}}) \\cdot \\left[ C_{\\text{penalty}} + \\Delta \\mathcal{S}_{\\text{loss}} \\cdot w_{\\text{soc}} \\right]$$
        For legal actions, $\\mathcal{R}_i(a) = 0$.
        """
        if action_type not in UNETHICAL_VILLAGE_ACTIONS:
            return 0.0

        p_det = max(0.0, min(1.0, float(detection_probability)))
        risk_attitude = max(0.15, 1.0 - 0.65 * self.weights.theta_risk)
        sanction_magnitude = float(coin_penalty) + float(status_loss) * self.weights.w_soc
        return float(p_det * risk_attitude * sanction_magnitude)

    def compute_state_utility(
        self,
        hunger: float,
        hydration: float,
        energy: float,
        coins: float,
        inventory: CommodityInventory,
        clearing_prices: dict[CommodityType, float],
        societal_status: float,
        risk_penalty: float = 0.0,
    ) -> float:
        """Computes composite multi-objective utility (§2.1):

        $$U_i(t) = w_{\\text{phys}} \\cdot U_{\\text{phys}}(H_i, T_i, E_i) + w_{\\text{econ}} \\cdot U_{\\text{econ}}(W_i) + w_{\\text{soc}} \\cdot \\mathcal{S}_i(t) - \\mathcal{R}_i(a)$$
        """
        u_phys = self.compute_physiological_utility(hunger, hydration, energy)
        u_econ = self.compute_economic_utility(coins, inventory, clearing_prices)
        return float(
            self.weights.w_phys * u_phys
            + self.weights.w_econ * u_econ
            + self.weights.w_soc * float(societal_status)
            - float(risk_penalty)
        )

    def evaluate_action(
        self,
        action_type: VillageActionType,
        current_hunger: float,
        current_hydration: float,
        current_energy: float,
        current_coins: float,
        current_inventory: CommodityInventory,
        clearing_prices: dict[CommodityType, float],
        current_status: float,
        projected_hunger: float,
        projected_hydration: float,
        projected_energy: float,
        projected_coins: float,
        projected_inventory: CommodityInventory,
        projected_status: float,
        detection_probability: float = 0.0,
        distance_cost: float = 0.0,
        coin_penalty: float = 12.0,
        status_loss: float = 0.08,
        target_zone: str = "",
        rationale_note: str = "",
    ) -> ActionUtilityEvaluation:
        """Evaluates expected utility $\\mathbb{E}[U_i(a, s_t)]$ for a candidate action."""
        u_phys_cur = self.compute_physiological_utility(
            current_hunger, current_hydration, current_energy
        )
        u_phys_proj = self.compute_physiological_utility(
            projected_hunger, projected_hydration, projected_energy
        )
        u_econ_proj = self.compute_economic_utility(
            projected_coins, projected_inventory, clearing_prices
        )
        u_soc_proj = float(projected_status)

        u_gain = (
            self.weights.w_phys * u_phys_proj
            + self.weights.w_econ * u_econ_proj
            + self.weights.w_soc * u_soc_proj
            - 0.18 * max(0.0, float(distance_cost))
        )
        r_pen = self.compute_risk_penalty(
            action_type=action_type,
            detection_probability=detection_probability,
            coin_penalty=coin_penalty,
            status_loss=status_loss,
        )
        expected_total = u_gain - r_pen

        return ActionUtilityEvaluation(
            action_type=action_type,
            is_unethical=(action_type in UNETHICAL_VILLAGE_ACTIONS),
            u_phys_current=round(u_phys_cur, 5),
            u_phys_projected=round(u_phys_proj, 5),
            u_econ_projected=round(u_econ_proj, 5),
            u_soc_projected=round(u_soc_proj, 5),
            u_gain=round(u_gain, 5),
            detection_probability=round(detection_probability, 5),
            expected_risk_penalty=round(r_pen, 5),
            expected_total_utility=round(expected_total, 5),
            target_zone=target_zone,
            rationale_note=rationale_note,
        )

    def rank_candidate_actions(
        self,
        candidates: list[ActionUtilityEvaluation],
        step_callback: Callable[[ActionUtilityEvaluation], None] | None = None,
    ) -> list[ActionUtilityEvaluation]:
        """Ranks candidate actions by expected utility descending (deterministic tie-break)."""
        for c in candidates:
            if step_callback is not None:
                step_callback(c)
        return sorted(
            candidates,
            key=lambda c: (-c.expected_total_utility, c.is_unethical, c.action_type.value),
        )
