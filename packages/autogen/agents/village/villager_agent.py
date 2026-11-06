"""VillagerAgent: Multi-Objective Utility, Belief State, and Rational Ethics Agent."""

from __future__ import annotations

import math
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.core.types import DecisionTrace
from autogen.intelligence.village.affective_pad import (
    AffectivePADEngine,
    AffectiveStateVector,
    EmotionCategory,
)
from autogen.intelligence.village.utility_ethics import (
    ActionUtilityEvaluation,
    UtilityEngine,
    VillagerUtilityWeights,
)
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.environment.village.commodities import (
    COMMODITY_CATALOG,
    CommodityType,
)
from autogen.simulation.environment.village.village_grid import (
    CANONICAL_ARCHETYPES,
    VillageActionType,
    VillageObservation,
    VillageRole,
)
from autogen.simulation.state.types import ActionType, Position


class VillagerBeliefState(BaseModel):
    """Subjective cognitive map, price beliefs, and episodic memory of a villager.

    Strictly separated from actual ground-truth world state (AGENTS.md rule 6).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    believed_prices: dict[CommodityType, float] = Field(
        default_factory=lambda: {c: s.base_price for c, s in COMMODITY_CATALOG.items()}
    )
    believed_zone_centers: dict[str, Position] = Field(default_factory=dict)
    subjective_trust: dict[str, float] = Field(default_factory=dict)
    known_offenders: tuple[str, ...] = ()
    episodic_memories: tuple[str, ...] = ()
    believed_detection_risk: float = Field(default=0.25, ge=0.0, le=1.0)


class VillagerAgent(BaseAgent):
    """Autonomous villager optimizing physiological survival, wealth, status, and ethics risk."""

    def __init__(
        self,
        agent_id: str,
        name: str | None = None,
        role: VillageRole = VillageRole.FARMER,
        utility_weights: VillagerUtilityWeights | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        caps = capabilities or AgentCapabilities(
            memory=True,
            belief=True,
            planning=True,
            learning=True,
        )
        super().__init__(agent_id=agent_id, capabilities=caps)
        self.name = name or agent_id
        self.role = role
        self.archetype_title = CANONICAL_ARCHETYPES.get(agent_id, f"{role.value} Specialist")
        self.utility_engine = UtilityEngine(weights=utility_weights)
        self.affective_engine = AffectivePADEngine()
        self.affective_state = AffectiveStateVector(agent_id=agent_id)
        self.affective_state = AffectiveStateVector()
        self.sabha_summoned: bool = False
        self._last_village_obs: VillageObservation | None = None
        self.belief_state = VillagerBeliefState()
        self.cumulative_reward: float = 0.0
        self.last_trace: DecisionTrace | None = None
        self.last_evaluations: list[ActionUtilityEvaluation] = []

    @property
    def utility_weights(self) -> VillagerUtilityWeights:
        return self.utility_engine.weights

    def observe(self, observation: Observation) -> None:
        """Ingests standard Observation (BaseAgent contract)."""
        super().observe(observation)

    def observe_village(self, village_obs: VillageObservation) -> None:
        """Ingests rich sensory VillageObservation from VillageWorld."""
        self._last_village_obs = village_obs

    def update_belief(self) -> None:
        """Updates subjective beliefs, cognitive map, and episodic memory from observation."""
        v_obs = self._last_village_obs
        if v_obs is None:
            return

        new_prices = dict(self.belief_state.believed_prices)
        new_prices.update(v_obs.market_clearing_prices)

        new_zones = dict(self.belief_state.believed_zone_centers)
        new_zones.update(v_obs.zone_centers)

        trust_map = dict(self.belief_state.subjective_trust)
        offenders = list(self.belief_state.known_offenders)
        memories = list(self.belief_state.episodic_memories)

        for out in v_obs.witnessed_outcomes:
            if out.description and out.description not in memories:
                memories.append(f"[Step {out.step}] {out.description}")
            if out.actor_id != self.agent_id:
                cur_t = trust_map.get(out.actor_id, 0.5)
                if out.category == "coop":
                    trust_map[out.actor_id] = min(1.0, round(cur_t + 0.12 * (1.0 - cur_t), 4))
                elif out.category == "defect":
                    trust_map[out.actor_id] = max(0.0, round(cur_t * 0.68, 4))
                    if out.actor_id not in offenders:
                        offenders.append(out.actor_id)
                elif out.category == "crime_caught":
                    trust_map[out.actor_id] = max(0.0, round(cur_t - 0.50, 4))
                    if out.actor_id not in offenders:
                        offenders.append(out.actor_id)

        # Estimate subjective detection probability from visible witnesses and ambient light
        est_det = self.estimate_subjective_detection_probability(v_obs)

        # Appraise OCC affective stimuli and update continuous 3D PAD emotional state
        witnessed_tuples = [
            (out.category, out.actor_id, out.target_id, out.detected)
            for out in v_obs.witnessed_outcomes
        ]
        max_vis_status = max(
            (vv.societal_status for vv in v_obs.visible_villagers),
            default=v_obs.societal_status,
        )
        stimuli = self.affective_engine.appraise_observation(
            agent_id=self.agent_id,
            hunger=v_obs.hunger,
            hydration=v_obs.hydration,
            energy=v_obs.energy,
            coins=v_obs.coins,
            societal_status=v_obs.societal_status,
            in_pillory=(v_obs.in_pillory_ticks > 0),
            witnessed_categories=witnessed_tuples,
            max_visible_peer_status=max_vis_status,
        )
        self.affective_state = self.affective_engine.step_pad_state(
            current=self.affective_state,
            stimuli=stimuli,
            step=v_obs.step,
        )

        self.belief_state = VillagerBeliefState(
            believed_prices=new_prices,
            believed_zone_centers=new_zones,
            subjective_trust=trust_map,
            known_offenders=tuple(offenders[-10:]),
            episodic_memories=tuple(memories[-15:]),
            believed_detection_risk=est_det,
        )

    def estimate_subjective_detection_probability(self, v_obs: VillageObservation) -> float:
        """Computes the agent's subjective estimate of $P_{\\text{detect}}$ (§2.3.1)
        based strictly on visible witnesses, ambient light, and watchtowers.
        """
        light = v_obs.ambient_light
        not_detected = 1.0

        for vis in v_obs.visible_villagers:
            dist = math.hypot(
                v_obs.position.x - vis.position.x,
                v_obs.position.y - vis.position.y,
            )
            role_mult = 1.45 if vis.role == VillageRole.GUARD else 1.0
            dist_factor = math.exp(-0.18 * dist) * max(0.0, 1.0 - dist / 14.0)
            p_sight = min(0.98, 0.88 * light * role_mult * dist_factor)
            not_detected *= 1.0 - p_sight

        if v_obs.near_watchtower:
            not_detected *= 1.0 - (0.20 * light)

        # Base granary/chest lock security perception
        not_detected *= 1.0 - (0.22 * light)
        return float(max(0.02, min(0.99, round(1.0 - not_detected, 5))))

    def evaluate_all_pathways(
        self, v_obs: VillageObservation
    ) -> list[ActionUtilityEvaluation]:
        """Evaluates legal and unethical candidate actions under $\\mathbb{E}[U_i(a, s_t)]$."""
        h = v_obs.hunger
        t = v_obs.hydration
        e = v_obs.energy
        c = v_obs.coins
        inv = v_obs.inventory
        s = v_obs.societal_status
        prices = self.belief_state.believed_prices
        p_det = self.belief_state.believed_detection_risk

        pos = v_obs.position
        zones = v_obs.zone_centers

        def dist_to(z_key: str, fallback: Position = pos) -> float:
            target = zones.get(z_key, fallback)
            return float(pos.manhattan_distance(target))

        candidates: list[ActionUtilityEvaluation] = []

        # 1. CONSUME_RATION (Immediate physiological recovery if hunger/hydration < 65)
        if (h < 65.0 and inv.total_edible_units() >= 0.5) or (
            t < 65.0 and inv.get(CommodityType.WATER) >= 0.5
        ):
            dh = 30.0 if inv.total_edible_units() >= 0.5 else 0.0
            dt = 35.0 if inv.get(CommodityType.WATER) >= 0.5 else 0.0
            proj_inv = inv.with_delta(
                CommodityType.WHEAT if inv.get(CommodityType.WHEAT) >= 0.5 else CommodityType.WATER,
                -1.0,
            )
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.CONSUME_RATION,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=min(100.0, h + dh),
                    projected_hydration=min(100.0, t + dt),
                    projected_energy=min(100.0, e + 8.0),
                    projected_coins=c,
                    projected_inventory=proj_inv,
                    projected_status=s,
                    distance_cost=0.0,
                    target_zone="self",
                    rationale_note="Consume carried rations to restore vital saturation.",
                )
            )

        # 2. DRAW_WATER at Town Well
        if v_obs.well_water_available > 2.0 and (t < 75.0 or self.role == VillageRole.WATER_CARRIER):
            role_bonus = 1.6 if self.role == VillageRole.WATER_CARRIER else 1.0
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.DRAW_WATER,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=min(100.0, t + 32.0),
                    projected_energy=max(0.0, e - 1.0),
                    projected_coins=c,
                    projected_inventory=inv.with_delta(CommodityType.WATER, 3.0 * role_bonus),
                    projected_status=min(1.0, s + 0.015 * role_bonus),
                    distance_cost=dist_to("town_well") * 0.08,
                    target_zone="town_well",
                    rationale_note="Draw clean water from the Town Well.",
                )
            )

        # 3. HARVEST_FIELD at Agricultural Fields
        farm_mult = 1.6 if self.role == VillageRole.FARMER else 0.9
        candidates.append(
            self.utility_engine.evaluate_action(
                action_type=VillageActionType.HARVEST_FIELD,
                current_hunger=h,
                current_hydration=t,
                current_energy=e,
                current_coins=c,
                current_inventory=inv,
                clearing_prices=prices,
                current_status=s,
                projected_hunger=min(100.0, h + (15.0 if h < 50.0 else 4.0)),
                projected_hydration=t,
                projected_energy=max(0.0, e - 2.0),
                projected_coins=c,
                projected_inventory=inv.with_delta(CommodityType.WHEAT, 3.2 * farm_mult),
                projected_status=min(1.0, s + 0.022 * farm_mult),
                distance_cost=dist_to("agricultural_fields") * 0.08,
                target_zone="agricultural_fields",
                rationale_note="Work agricultural fields to harvest wheat.",
            )
        )

        # 4. FORAGE_FOREST at Wild Forest
        forage_mult = 1.65 if self.role in (VillageRole.WANDERER, VillageRole.WOODCUTTER) else 0.85
        candidates.append(
            self.utility_engine.evaluate_action(
                action_type=VillageActionType.FORAGE_FOREST,
                current_hunger=h,
                current_hydration=t,
                current_energy=e,
                current_coins=c,
                current_inventory=inv,
                clearing_prices=prices,
                current_status=s,
                projected_hunger=min(100.0, h + 14.0 * forage_mult),
                projected_hydration=min(100.0, t + 6.0),
                projected_energy=max(0.0, e - 1.5),
                projected_coins=c,
                projected_inventory=inv.with_delta(CommodityType.BERRIES, 2.8 * forage_mult),
                projected_status=min(1.0, s + 0.018 * forage_mult),
                distance_cost=dist_to("wild_forest") * 0.08,
                target_zone="wild_forest",
                rationale_note="Forage berries and timber in the Wild Forest.",
            )
        )

        # 5. REST_HUT at Private Hut (Evaluated when energy is depleted < 65%)
        if e < 65.0:
            hut_dist = float(pos.manhattan_distance(v_obs.home_hut_position))
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.REST_HUT,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=min(100.0, e + 38.0),
                    projected_coins=c,
                    projected_inventory=inv,
                    projected_status=s,
                    distance_cost=hut_dist * 0.08,
                    target_zone="private_huts_west",
                    rationale_note="Rest inside private hut to restore depleted energy.",
                )
            )

        # 6. TRADE_MARKET at Central Market
        merch_mult = 1.65 if self.role == VillageRole.MERCHANT else 1.0
        wheat_p = prices.get(CommodityType.WHEAT, 3.0)
        if self.role == VillageRole.MERCHANT or inv.get(CommodityType.WHEAT) > 3.0 or (c >= wheat_p and h < 65.0):
            proj_c = c + 2.8 * wheat_p * merch_mult if inv.get(CommodityType.WHEAT) >= 1.5 else max(0.0, c - wheat_p)
            proj_h = h if inv.get(CommodityType.WHEAT) >= 1.5 else min(100.0, h + 28.0)
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.TRADE_MARKET,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=proj_h,
                    projected_hydration=t,
                    projected_energy=e,
                    projected_coins=proj_c,
                    projected_inventory=inv,
                    projected_status=min(1.0, s + 0.025 * merch_mult),
                    distance_cost=dist_to("central_market") * 0.08,
                    target_zone="central_market",
                    rationale_note="Exchange surplus commodities at the Central Market.",
                )
            )

        # 7. TITHE_GRANARY at Village Commons (builds Societal Status S_i)
        if inv.get(CommodityType.WHEAT) >= 2.0 and h > 45.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.TITHE_GRANARY,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=e,
                    projected_coins=c,
                    projected_inventory=inv.with_delta(CommodityType.WHEAT, -1.5),
                    projected_status=min(1.0, s + 0.065),
                    distance_cost=dist_to("village_commons") * 0.08,
                    target_zone="village_commons",
                    rationale_note="Tithe surplus wheat to the Granary to build social capital.",
                )
            )

        # 8. PATROL_GUARD (for Guards)
        if self.role == VillageRole.GUARD:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.PATROL_GUARD,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=max(0.0, e - 1.0),
                    projected_coins=c + 3.5,
                    projected_inventory=inv,
                    projected_status=min(1.0, s + 0.045),
                    distance_cost=dist_to("village_commons") * 0.06,
                    target_zone="village_commons",
                    rationale_note="Guard the Granary Commons and Town Square.",
                )
            )
            if self.agent_id == "villager_dev" or len(self.belief_state.known_offenders) > 0:
                candidates.append(
                    self.utility_engine.evaluate_action(
                        action_type=VillageActionType.INVESTIGATE_CRIME,
                        current_hunger=h,
                        current_hydration=t,
                        current_energy=e,
                        current_coins=c,
                        current_inventory=inv,
                        clearing_prices=prices,
                        current_status=s,
                        projected_hunger=h,
                        projected_hydration=t,
                        projected_energy=max(0.0, e - 1.0),
                        projected_coins=c + 3.8,
                        projected_inventory=inv,
                        projected_status=min(1.0, s + 0.055),
                        distance_cost=dist_to("village_commons") * 0.05,
                        target_zone="village_commons",
                        rationale_note="Inspect forensic evidence and interview eyewitnesses.",
                    )
                )

        # 8b. Specialized Guild & Institutional Pathways (§4 & §5)
        if self.sabha_summoned and h > 30.0 and t > 30.0:
            is_magistrate = self.agent_id == "villager_lata" or self.role == VillageRole.MAGISTRATE
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.PRESIDE_SABHA if is_magistrate else VillageActionType.ATTEND_SABHA,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=e,
                    projected_coins=c + (3.5 if is_magistrate else 1.5),
                    projected_inventory=inv,
                    projected_status=min(1.0, s + (0.095 if is_magistrate else 0.065)),
                    distance_cost=dist_to("town_well") * 0.03,
                    target_zone="town_well",
                    rationale_note="Answer the Town Hall bell to deliberate in the Gram Sabha assembly.",
                )
            )

        if (self.agent_id == "villager_meera" or self.role == VillageRole.MILLER) and h > 40.0 and t > 40.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.MILL_WHEAT,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=min(100.0, h + 18.0),
                    projected_hydration=t,
                    projected_energy=max(0.0, e - 1.5),
                    projected_coins=c + 4.2,
                    projected_inventory=inv.with_delta(CommodityType.BREAD, 1.8),
                    projected_status=min(1.0, s + 0.048),
                    distance_cost=dist_to("windmill") * 0.06,
                    target_zone="windmill",
                    rationale_note="Mill raw wheat at the Old Windmill into flour and baked bread.",
                )
            )

        if (self.agent_id == "villager_ravi" or self.role == VillageRole.BLACKSMITH) and h > 40.0 and t > 40.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.FORGE_TOOLS,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=max(0.0, e - 1.8),
                    projected_coins=c + 4.8,
                    projected_inventory=inv.with_delta(CommodityType.TOOLS, 1.5),
                    projected_status=min(1.0, s + 0.052),
                    distance_cost=dist_to("smithy") * 0.06,
                    target_zone="smithy",
                    rationale_note="Forge iron agricultural tools at the Blacksmith anvil.",
                )
            )

        if (self.agent_id == "villager_imran" or self.role == VillageRole.HERBALIST) and h > 40.0 and t > 40.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.BREW_HEALING_BROTH,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=min(100.0, h + 12.0),
                    projected_hydration=min(100.0, t + 12.0),
                    projected_energy=max(0.0, e - 1.2),
                    projected_coins=c + 3.8,
                    projected_inventory=inv.with_delta(CommodityType.HEALING_BROTH, 1.5),
                    projected_status=min(1.0, s + 0.050),
                    distance_cost=dist_to("tavern") * 0.06,
                    target_zone="tavern",
                    rationale_note="Brew medicinal herb broth to restore village health.",
                )
            )

        if (self.agent_id == "villager_nila" or self.role == VillageRole.BARD) and h > 45.0 and t > 45.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.SING_BALLAD,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=min(100.0, e + 6.0),
                    projected_coins=c + 3.6,
                    projected_inventory=inv,
                    projected_status=min(1.0, s + 0.055),
                    distance_cost=dist_to("tavern") * 0.05,
                    target_zone="tavern",
                    rationale_note="Perform a lute ballad at the Golden Hearth Tavern to uplift morale.",
                )
            )

        if self.affective_state.dominant_emotion == EmotionCategory.GUILT or (s < 0.065 and h > 45.0):
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.COMMUNITY_SERVICE,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=max(0.0, e - 1.5),
                    projected_coins=c,
                    projected_inventory=inv,
                    projected_status=min(1.0, s + 0.085),
                    distance_cost=dist_to("village_commons") * 0.05,
                    target_zone="village_commons",
                    rationale_note="Perform public community service to rehabilitate tarnished reputation.",
                )
            )

        if self.affective_state.dominant_emotion == EmotionCategory.GRIEF and h > 35.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.MOURN_AT_CHAPEL,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=min(100.0, e + 10.0),
                    projected_coins=c,
                    projected_inventory=inv,
                    projected_status=min(1.0, s + 0.040),
                    distance_cost=dist_to("chapel") * 0.05,
                    target_zone="chapel",
                    rationale_note="Mourn at St. Aurelia Chapel to find solace from grief.",
                )
            )

        # ======================================================================
        # UNETHICAL / CRIMINAL PATHWAYS (§2.3: Evaluated via Risk-Reward Utility)
        # ======================================================================
        # 9. STEAL_GRANARY: High immediate food + wealth gain, penalized by P_detect * sanction
        if v_obs.granary_wheat_available >= 2.0:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.STEAL_GRANARY,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=min(100.0, h + 45.0),
                    projected_hydration=t,
                    projected_energy=min(100.0, e + 10.0),
                    projected_coins=c,
                    projected_inventory=inv.with_delta(CommodityType.WHEAT, 3.5),
                    projected_status=s,
                    detection_probability=p_det,
                    distance_cost=dist_to("village_commons") * 0.4,
                    coin_penalty=12.0,
                    status_loss=0.10,
                    target_zone="village_commons",
                    rationale_note=(
                        f"Illicitly pilfer 3.5 wheat from Granary (P_detect={p_det:.2f})."
                    ),
                )
            )

        # 10. HOARD_AND_GOUGE: White-collar market manipulation when holding grain
        if inv.get(CommodityType.WHEAT) >= 2.5:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.HOARD_AND_GOUGE,
                    current_hunger=h,
                    current_hydration=t,
                    current_energy=e,
                    current_coins=c,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=s,
                    projected_hunger=h,
                    projected_hydration=t,
                    projected_energy=e,
                    projected_coins=c + 3.0 * wheat_p * 1.65,
                    projected_inventory=inv.with_delta(CommodityType.WHEAT, -2.0),
                    projected_status=s,
                    detection_probability=min(0.95, p_det * 0.85),
                    distance_cost=dist_to("central_market") * 0.4,
                    coin_penalty=6.0,
                    status_loss=0.07,
                    target_zone="central_market",
                    rationale_note="Hoard scarce wheat and price-gouge at 165% market clearing price.",
                )
            )

        return self.utility_engine.rank_candidate_actions(candidates)

    def _step_toward(self, current: Position, target: Position) -> tuple[ActionType, VillageActionType]:
        """Selects the single-step Manhattan movement toward target coordinate."""
        dx = target.x - current.x
        dy = target.y - current.y
        if abs(dx) >= abs(dy) and dx != 0:
            if dx > 0:
                return ActionType.RIGHT, VillageActionType.MOVE_EAST
            return ActionType.LEFT, VillageActionType.MOVE_WEST
        if dy != 0:
            if dy > 0:
                return ActionType.DOWN, VillageActionType.MOVE_SOUTH
            return ActionType.UP, VillageActionType.MOVE_NORTH
        return ActionType.STAY, VillageActionType.STAY

    def decide(self) -> tuple[Action, DecisionTrace]:
        """Deliberates over legal and unethical pathways and emits a verified DecisionTrace."""
        t0 = time.perf_counter_ns()

        # Fallback if running in a standard non-village GridWorld via BaseAgent contract
        if self._last_village_obs is None:
            obs = self._last_observation
            step_id = obs.step if obs else 0
            act = Action(
                agent_id=self.agent_id,
                action_type=ActionType.COLLECT if (obs and obs.energy < 80) else ActionType.STAY,
                parameters={"village_action": VillageActionType.STAY.value},
            )
            trace = DecisionTrace(
                agent_id=self.agent_id,
                step_id=step_id,
                algorithm_name="social_utility_ethics",
                decision_type="village_deliberation",
                metrics={"candidates_evaluated": 1, "runtime_ns": time.perf_counter_ns() - t0},
                rationale={"village_action": "STAY", "mode": "gridworld_fallback"},
            )
            self.last_trace = trace
            return act, trace

        v_obs = self._last_village_obs
        ranked = self.evaluate_all_pathways(v_obs)
        self.last_evaluations = ranked
        best = ranked[0]

        # Determine target coordinate for spatial execution
        target_pos = v_obs.position
        if best.action_type == VillageActionType.REST_HUT:
            target_pos = v_obs.home_hut_position
        elif best.target_zone and best.target_zone in v_obs.zone_centers:
            target_pos = v_obs.zone_centers[best.target_zone]

        # Execute immediate action if within 2 tiles of zone center or if self-targeted
        dist = v_obs.position.manhattan_distance(target_pos)
        if best.target_zone == "self" or dist <= 2:
            exec_v_act = best.action_type
            prim_act = ActionType.COLLECT
        else:
            prim_act, exec_v_act = self._step_toward(v_obs.position, target_pos)

        elapsed_ns = max(1, time.perf_counter_ns() - t0)
        unethical_evals = [c for c in ranked if c.is_unethical]
        best_unethical_u = (
            unethical_evals[0].expected_total_utility if unethical_evals else -999.0
        )

        rationale: dict[str, Any] = {
            "village_action": exec_v_act.value,
            "strategic_goal_action": best.action_type.value,
            "target_zone": best.target_zone,
            "is_unethical": best.is_unethical,
            "expected_total_utility": best.expected_total_utility,
            "u_phys_current": best.u_phys_current,
            "u_phys_projected": best.u_phys_projected,
            "u_econ_projected": best.u_econ_projected,
            "u_soc_projected": best.u_soc_projected,
            "detection_probability": best.detection_probability,
            "expected_risk_penalty": best.expected_risk_penalty,
            "best_unethical_utility": best_unethical_u,
            "dominant_emotion": self.affective_state.dominant_emotion.value,
            "pad_vector": list(self.affective_state.to_vector()),
            "rationale_note": best.rationale_note,
            "top_candidates": [
                {
                    "action": c.action_type.value,
                    "expected_u": c.expected_total_utility,
                    "risk_penalty": c.expected_risk_penalty,
                    "is_unethical": c.is_unethical,
                }
                for c in ranked[:4]
            ],
        }

        trace = DecisionTrace(
            agent_id=self.agent_id,
            step_id=v_obs.step,
            algorithm_name="social_utility_ethics",
            decision_type="unethical_pathway" if best.is_unethical else "legal_pathway",
            metrics={
                "candidates_evaluated": len(ranked),
                "expected_utility": best.expected_total_utility,
                "detection_risk": best.detection_probability,
                "societal_status": round(v_obs.societal_status, 4),
                "pad_valence": round(self.affective_state.valence, 4),
                "pad_arousal": round(self.affective_state.arousal, 4),
                "pad_dominance": round(self.affective_state.dominance, 4),
                "runtime_ns": elapsed_ns,
            },
            rationale=rationale,
        )
        self.last_trace = trace

        action = Action(
            agent_id=self.agent_id,
            action_type=prim_act,
            parameters={
                "village_action": exec_v_act.value,
                "strategic_goal_action": best.action_type.value,
                "goal_zone": best.target_zone,
            },
        )
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        """Records cumulative reward and adapts subjective risk caution if punished."""
        self.cumulative_reward += float(reward)
        v_obs = self._last_village_obs
        if v_obs is not None and v_obs.in_pillory_ticks > 0:
            # Experiencing pillory sanction slightly tempers risk tolerance
            new_theta = max(0.05, round(self.utility_weights.theta_risk * 0.92, 4))
            self.utility_engine = UtilityEngine(
                weights=self.utility_weights.model_copy(update={"theta_risk": new_theta})
            )

    def reset(self) -> None:
        """Resets internal and belief state for a new episode or replay."""
        super().reset()
        self._last_village_obs = None
        self.belief_state = VillagerBeliefState()
        self.affective_state = AffectiveStateVector(agent_id=self.agent_id)
        self.affective_state = AffectiveStateVector()
        self.sabha_summoned = False
        self.cumulative_reward = 0.0
        self.last_trace = None
        self.last_evaluations = []
