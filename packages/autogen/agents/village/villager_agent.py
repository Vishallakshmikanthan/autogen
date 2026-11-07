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
    AffectiveStimulus,
    EmotionCategory,
)
from autogen.intelligence.village.attention import (
    Alert,
    HabituationTracker,
    SalienceAttentionEngine,
)
from autogen.intelligence.village.bt import arbitrate_intentions
from autogen.intelligence.village.conversation import (
    Conversation,
    DialogueAct,
    DialogueActType,
    compute_dialogue_act_utility,
)
from autogen.intelligence.village.gossip import Belief, BeliefStore
from autogen.intelligence.village.intention import (
    LAMBDA_ATT,
    Intention,
    action_serves_intention,
    compute_reaction_utility,
    compute_switching_cost,
    should_interrupt,
)
from autogen.intelligence.village.memory_retrieval import MemoryQuery, retrieve_top_k
from autogen.intelligence.village.memory_stream import MemoryStream
from autogen.intelligence.village.needs import NeedsEngine
from autogen.intelligence.village.personality import (
    CANONICAL_AGENT_TRAITS,
    Traits,
    weights_from_traits,
)
from autogen.intelligence.village.reflection import (
    Disposition,
    MemoryEpisode,
    ReflectionEngine,
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
from autogen.simulation.environment.village.events import PerceivedEvent
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
        traits: Traits | None = None,
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

        self.traits: Traits = traits or CANONICAL_AGENT_TRAITS.get(agent_id, Traits())
        derived_w, self.personality = weights_from_traits(self.traits, role=self.role)
        final_weights = utility_weights if utility_weights is not None else derived_w
        self.base_utility_weights = final_weights

        self.utility_engine = UtilityEngine(weights=final_weights)
        self.needs_engine = NeedsEngine(seed=42)
        self.circadian_priors: dict[str, float] = {}
        self.intention_scores: dict[str, float] = {}
        self.affective_engine = AffectivePADEngine(baseline=self.personality.pad_baseline)
        self.affective_state = AffectiveStateVector(
            agent_id=agent_id,
            valence=self.personality.pad_baseline[0],
            arousal=self.personality.pad_baseline[1],
            dominance=self.personality.pad_baseline[2],
        )

        self.intention = Intention(kind="ROUTINE", commitment=1.0)
        self.needs_engine = NeedsEngine(seed=42)
        self.utility_engine = UtilityEngine(weights=self.base_utility_weights)
        self.intention_scores = {}
        self.attention_tracker = HabituationTracker()
        self.attention_engine = SalienceAttentionEngine(self.attention_tracker)
        self.pending_alerts: list[Alert] = []
        self.notices: list[PerceivedEvent] = []
        self.dispositions: dict[str, Disposition] = {}
        self.unreflected_episodes: list[MemoryEpisode] = []
        self.memory_stream = MemoryStream(agent_id=self.agent_id, capacity=300)
        self.belief_store = BeliefStore(agent_id=self.agent_id)
        self.active_conversation: Conversation | None = None
        self.last_dialogue_act: DialogueAct | None = None
        self.last_gaze_pos: Position | None = None
        self.last_gaze_reason: str = ""

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
        """Ingests rich sensory VillageObservation from VillageWorld and processes attention (§2.3, §2.4)."""
        self._last_village_obs = village_obs

        # Phase 2b Attention: Score salience, filter to alerts and notices
        alerts, notices, _ = self.attention_engine.process_percepts(
            percepts=village_obs.percepts,
            traits=self.traits,
            arousal=self.affective_state.arousal,
            step=village_obs.step,
        )
        self.notices = notices
        self.pending_alerts = [a for a in alerts if a.eligible_step <= village_obs.step]

        # Orientation / Gaze towards most salient event
        if alerts:
            top_p = alerts[0].percept.event
            self.last_gaze_pos = top_p.position
            self.last_gaze_reason = top_p.kind
        elif village_obs.percepts:
            self.last_gaze_pos = village_obs.percepts[0].event.position
            self.last_gaze_reason = village_obs.percepts[0].event.kind

        # Encode significant environmental percepts into MemoryStream (§4.1)
        for p in village_obs.percepts:
            if p.event.kind in ("THUNDER", "RAIN_ONSET", "CRY_OUT"):
                self.memory_stream.create_and_add(
                    step=village_obs.step,
                    kind="observation",
                    location=village_obs.current_zone.value if village_obs.current_zone else "",
                    summary=f"Witnessed {p.event.kind} (intensity={p.event.intensity:.2f})",
                    tags=("weather", "storm" if "RAIN" in p.event.kind or "THUNDER" in p.event.kind else "sound"),
                    valence=p.event.valence_hint,
                    pad=self.affective_state.to_tuple(),
                    self_involvement=1.0 if p.event.is_global else 0.5,
                )

        # Affective appraisal of threat alerts (§2.2, D4 fix: thunderstorms produce fear/distress)
        for alert in self.pending_alerts:
            ev = alert.percept.event
            if alert.threat > 0.0 or ev.kind in ("THUNDER", "CRY_OUT"):
                fear_stim = AffectiveStimulus(
                    stimulus_type="threat_alert",
                    stimulus_name=f"alert_{ev.kind.lower()}",
                    delta_v=-0.50 * alert.salience * (0.60 + 0.40 * self.traits.neuroticism),
                    delta_a=+0.70 * alert.salience * (0.60 + 0.40 * self.traits.neuroticism),
                    delta_d=-0.40 * alert.salience,
                )
                self.affective_state = self.affective_engine.step_emotion(
                    agent_id=self.agent_id,
                    step=village_obs.step,
                    current_state=self.affective_state,
                    stimuli=[fear_stim],
                )

        # Intention lifecycle management (§5.4, D1 fix)
        is_raining = any(p.event.kind in ("THUNDER", "RAIN_ONSET") for p in village_obs.percepts)
        if self.intention.kind in ("SHELTER", "RESPOND"):
            if not is_raining:
                self.intention = self.intention.mark_done()
        elif self.intention.kind == "GREET":
            if self.intention.progress >= 0.50 or (village_obs.step - self.intention.started_step) >= 2:
                self.intention = self.intention.mark_done()
        elif self.intention.is_expired(village_obs.step) or self.intention.is_terminal():
            self.intention = self.intention.mark_done()

        # Intention arbitration: check for interruptions
        for alert in self.pending_alerts:
            if should_interrupt(alert, self.intention, self.traits.conscientiousness):
                kind = alert.percept.event.kind
                if kind in ("THUNDER", "RAIN_ONSET") or alert.threat > 0.40:
                    self.intention = Intention(
                        kind="SHELTER",
                        started_step=village_obs.step,
                        commitment=0.95,
                        max_duration=12,
                    )
                    break
                elif kind == "CRY_OUT":
                    self.intention = Intention(
                        kind="INVESTIGATE_NOISE",
                        started_step=village_obs.step,
                        target_pos=alert.percept.event.position,
                        commitment=0.85,
                        max_duration=10,
                    )
                    break
                elif kind == "ARRIVAL":
                    self.intention = Intention(
                        kind="GREET",
                        started_step=village_obs.step,
                        target_id=alert.percept.event.source_id,
                        commitment=0.75,
                        max_duration=6,
                    )
                    break

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
            formatted_entry = f"[Step {out.step}] {out.description}"
            if out.description and not any(m == formatted_entry or m.endswith(out.description) for m in memories):
                memories.append(formatted_entry)
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

                is_harm = out.category in ("defect", "crime_caught", "crime_unseen")
                is_help = out.category in ("coop", "trade")
                val = -0.60 if is_harm else (0.50 if is_help else 0.0)
                self.unreflected_episodes.append(
                    MemoryEpisode(
                        episode_id=f"ep_{out.step}_{len(self.unreflected_episodes)}_{out.actor_id}",
                        step=out.step,
                        other_id=out.actor_id,
                        valence=val,
                        importance=0.75 if is_harm else 0.40,
                        is_harm=is_harm,
                        is_help=is_help,
                        description=out.description,
                    )
                )

                # Store first-hand belief regarding observed harm/help (§3.3)
                if is_harm:
                    self.belief_store.put(
                        Belief(
                            belief_id=f"belief:{self.agent_id}:{out.actor_id}:{out.step}",
                            subject=out.actor_id,
                            claim="crime" if "crime" in out.category or "steal" in str(out.action_type).lower() else "defect",
                            valence=-0.70,
                            confidence=0.95,
                            source_chain=(self.agent_id,),
                            first_hand=True,
                            heard_at_step=out.step,
                        )
                    )
                elif is_help:
                    self.belief_store.put(
                        Belief(
                            belief_id=f"belief:{self.agent_id}:{out.actor_id}:{out.step}",
                            subject=out.actor_id,
                            claim="cooperation" if out.category == "coop" else "fair_trade",
                            valence=0.60,
                            confidence=0.90,
                            source_chain=(self.agent_id,),
                            first_hand=True,
                            heard_at_step=out.step,
                        )
                    )

            # Ingest into structured MemoryStream (§4.1)
            is_actor = out.actor_id == self.agent_id
            is_target = out.target_id == self.agent_id if out.target_id else False
            self_inv = 1.0 if (is_actor or is_target) else 0.5
            is_harm_out = out.category in ("defect", "crime_caught", "crime_unseen")
            is_help_out = out.category in ("coop", "trade")
            val_out = -0.60 if is_harm_out else (0.50 if is_help_out else 0.0)

            out_tags = [out.category]
            if is_harm_out:
                out_tags.append("theft" if "steal" in out.action_type.value.lower() or "crime" in out.category else "defect")
            if is_help_out:
                out_tags.append("kindness")
            if out.target_id:
                out_tags.append(f"target_{out.target_id}")

            self.memory_stream.create_and_add(
                step=out.step,
                kind="action" if is_actor else "observation",
                actors=(out.actor_id, out.target_id) if out.target_id else (out.actor_id,),
                location=v_obs.current_zone.value if v_obs.current_zone else "",
                summary=out.description,
                tags=tuple(out_tags),
                valence=val_out,
                pad=self.affective_state.to_tuple(),
                self_involvement=self_inv,
                consequence=0.20 if is_actor or is_target else 0.0,
            )

        # Periodic forgetting / pruning (§4.3)
        self.memory_stream.prune(v_obs.step)

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

        # 11. Humanlike Reactive Pathways (§2.5)
        dist_shelter = min(
            dist_to("chapel"),
            dist_to("private_huts"),
            float(v_obs.position.manhattan_distance(v_obs.home_hut_position)),
        )
        candidates.append(
            self.utility_engine.evaluate_action(
                action_type=VillageActionType.SHELTER,
                current_hunger=h,
                current_hydration=t,
                current_energy=e,
                current_coins=c,
                current_inventory=inv,
                clearing_prices=prices,
                current_status=s,
                projected_hunger=h,
                projected_hydration=t,
                projected_energy=min(100.0, e + 5.0),
                projected_coins=c,
                projected_inventory=inv,
                projected_status=s,
                distance_cost=dist_shelter * 0.08,
                target_zone="shelter",
                rationale_note="Seek immediate indoor shelter from thunderstorm or hazards.",
            )
        )

        candidates.append(
            self.utility_engine.evaluate_action(
                action_type=VillageActionType.INVESTIGATE_NOISE,
                current_hunger=h,
                current_hydration=t,
                current_energy=e,
                current_coins=c,
                current_inventory=inv,
                clearing_prices=prices,
                current_status=s,
                projected_hunger=h,
                projected_hydration=t,
                projected_energy=max(0.0, e - 2.0),
                projected_coins=c,
                projected_inventory=inv,
                projected_status=s + 0.005,
                distance_cost=dist_to("town_well") * 0.3,
                target_zone="town_well",
                rationale_note="Investigate sudden auditory disturbance or cry.",
            )
        )

        has_nearby_peer = any(
            v_obs.position.manhattan_distance(vv.position) <= 3
            for vv in v_obs.visible_villagers
        )
        has_arrival_alert = any(a.percept.event.kind == "ARRIVAL" for a in self.pending_alerts)
        if has_nearby_peer or has_arrival_alert:
            candidates.append(
                self.utility_engine.evaluate_action(
                    action_type=VillageActionType.GREET,
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
                    projected_inventory=inv,
                    projected_status=s + 0.01,
                    distance_cost=0.0,
                    target_zone="self",
                    rationale_note="Exchange greetings with a nearby villager.",
                )
            )

        # Identify nearest visible peer for relational utility (§4.5)
        nearest_peer_id: str | None = None
        if v_obs.visible_villagers:
            nearest_peer = min(
                v_obs.visible_villagers,
                key=lambda vv: v_obs.position.manhattan_distance(vv.position),
            )
            nearest_peer_id = nearest_peer.agent_id

        disp = self.dispositions.get(nearest_peer_id) if nearest_peer_id else None
        u_rel_peer = 2.0 * disp.confidence * (disp.fondness - disp.grudge) if disp else 0.0

        # Crime pattern beliefs for guards (§4.5)
        crime_beliefs = [
            m for m in self.memory_stream.all_items()
            if m.kind == "belief" and "crime_pattern" in m.tags
        ]

        # Modulate candidates with u_react, C_switch, and u_rel (§2.5, §4.5)
        modulated: list[ActionUtilityEvaluation] = []
        for cand in candidates:
            u_react = compute_reaction_utility(cand.action_type, self.pending_alerts, self.traits)
            c_switch = compute_switching_cost(
                cand.action_type,
                self.intention,
                self.personality.switching_cost_scale,
            )
            u_rel = 0.0
            if cand.action_type in (VillageActionType.GREET, VillageActionType.COURTSHIP_BOND):
                u_rel = u_rel_peer
            elif cand.action_type == VillageActionType.PATROL_GUARD and self.role == VillageRole.GUARD:
                if any(cand.target_zone in b.tags or b.location == cand.target_zone for b in crime_beliefs):
                    u_rel = 0.50

            u_prime = cand.expected_total_utility + LAMBDA_ATT * u_react - c_switch + u_rel
            note = cand.rationale_note
            if u_react > 0.0 or c_switch > 0.0 or u_rel != 0.0:
                note += f" [u_react={u_react:.2f}, c_switch={c_switch:.2f}, u_rel={u_rel:.2f}]"
            modulated.append(
                cand.model_copy(
                    update={
                        "expected_total_utility": round(u_prime, 4),
                        "rationale_note": note,
                    }
                )
            )

        return self.utility_engine.rank_candidate_actions(modulated)

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

        # Phase 2e: Dynamic Needs, Prepotency Gating & Drive Pressures (§5.3)
        max_threat = max((a.threat for a in self.pending_alerts), default=0.0)
        need_st = self.needs_engine.step(
            obs=v_obs,
            traits=self.traits,
            contact_quality=0.0,
            executed_action=None,
            max_village_status=1.0,
            max_threat=max_threat,
        )
        dynamic_w = self.needs_engine.get_dynamic_weights(self.base_utility_weights)
        self.utility_engine.weights = dynamic_w

        # Reset expired or completed intention to allow fresh deliberative arbitration
        if self.intention.is_terminal() or self.intention.is_expired(v_obs.step):
            self.intention = Intention(kind="ROUTINE", started_step=v_obs.step, commitment=0.70)

        # Deliberate across all potential village pathways
        all_cands = self.evaluate_all_pathways(v_obs)

        # Phase 3: Behavior Tree Intention Arbitration (§5.4)
        has_emerg = any(
            a.threat > 0.40 or a.percept.event.kind in ("THUNDER", "RAIN_ONSET")
            for a in self.pending_alerts
        )
        self.intention, self.intention_scores = arbitrate_intentions(
            current_intention=self.intention,
            current_step=v_obs.step,
            need_state=need_st,
            weights=dynamic_w,
            role=self.role,
            time_of_day=v_obs.time_of_day,
            candidate_evals=all_cands,
            switching_cost_scale=self.personality.switching_cost_scale,
            has_active_emergency=has_emerg,
        )

        # Restrict candidate action selection to intention-serving actions ∪ safety actions (§5.4)
        filtered_cands = [
            c for c in all_cands
            if action_serves_intention(c.action_type, self.intention)
            or c.action_type in (VillageActionType.CONSUME_RATION, VillageActionType.SHELTER)
            or (c.is_unethical and need_st.d_phys > 0.70)
        ]
        ranked = filtered_cands if filtered_cands else all_cands
        self.last_evaluations = ranked
        best = ranked[0]

        # Advance intention progress if action aligns
        if action_serves_intention(best.action_type, self.intention):
            self.intention = self.intention.with_progress(0.35)

        # Determine target coordinate for spatial execution
        target_pos = v_obs.position
        if best.action_type == VillageActionType.REST_HUT:
            target_pos = v_obs.home_hut_position
        elif best.action_type == VillageActionType.DRAW_WATER:
            target_pos = v_obs.zone_centers.get("town_well", v_obs.position)
        elif best.action_type == VillageActionType.TRADE_MARKET:
            target_pos = v_obs.zone_centers.get("market_square", v_obs.position)
        elif best.action_type == VillageActionType.HARVEST_FIELD:
            target_pos = v_obs.zone_centers.get("farm_fields", v_obs.position)
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

        best_u_react = compute_reaction_utility(best.action_type, self.pending_alerts, self.traits)
        best_c_switch = compute_switching_cost(
            best.action_type,
            self.intention,
            self.personality.switching_cost_scale,
        )

        # Context Query and Memory Retrieval (§4.2)
        nearest_peer_id: str | None = None
        if v_obs.visible_villagers:
            nearest_peer = min(
                v_obs.visible_villagers,
                key=lambda vv: v_obs.position.manhattan_distance(vv.position),
            )
            nearest_peer_id = nearest_peer.agent_id

        mem_query = MemoryQuery(
            step=v_obs.step,
            actors=(nearest_peer_id,) if nearest_peer_id else (),
            location=v_obs.current_zone.value if v_obs.current_zone else "",
            tags=(self.intention.kind.lower(),),
            pad=self.affective_state.to_tuple(),
        )
        retrieved = retrieve_top_k(self.memory_stream, mem_query, k=5, rehearse=True)
        retrieved_mids = [m.mid for m, _ in retrieved]

        disp_best = self.dispositions.get(nearest_peer_id) if nearest_peer_id else None
        best_u_rel = 0.0
        if best.action_type in (VillageActionType.GREET, VillageActionType.COURTSHIP_BOND) and disp_best:
            best_u_rel = 2.0 * disp_best.confidence * (disp_best.fondness - disp_best.grudge)
        elif best.action_type == VillageActionType.PATROL_GUARD and self.role == VillageRole.GUARD:
            crime_beliefs = [m for m in self.memory_stream.all_items() if m.kind == "belief" and "crime_pattern" in m.tags]
            if any(best.target_zone in b.tags or b.location == best.target_zone for b in crime_beliefs):
                best_u_rel = 0.50

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
            "alerts": [a.percept.event.kind for a in self.pending_alerts],
            "u_react": round(best_u_react, 4),
            "c_switch": round(best_c_switch, 4),
            "u_rel": round(best_u_rel, 4),
            "retrieved_memory_ids": retrieved_mids,
            "intention": self.intention.kind,
            "intention_status": self.intention.status,
            "intention_score": round(self.intention.score, 3),
            "intention_step_index": self.intention.step_index,
            "needs": {
                "deficits": list(need_st.deficits_tuple),
                "gates": list(need_st.gates_tuple),
                "pressures": list(need_st.pressures_tuple),
            },
            "gaze": {
                "target_pos": [self.last_gaze_pos.x, self.last_gaze_pos.y] if self.last_gaze_pos else [v_obs.position.x, v_obs.position.y],
                "reason": self.last_gaze_reason,
            },
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

    def reflect_nightly(
        self,
        completed_action: VillageActionType | None = None,
        is_drought: bool = False,
        granary_wheat: float = 100.0,
    ) -> None:
        """Consolidates episodic memory into dispositions at rest or night cycle (§4.4)."""
        self.dispositions = ReflectionEngine.decay_grudges(self.dispositions, self.traits)

        # Legacy episodes consolidation
        if self.unreflected_episodes:
            self.dispositions = ReflectionEngine.reflect_episodes(
                unreflected=self.unreflected_episodes,
                current_dispositions=self.dispositions,
                traits=self.traits,
                is_drought=is_drought,
                granary_wheat=granary_wheat,
            )
            self.unreflected_episodes.clear()

        # Structured MemoryStream consolidation (§4.4)
        unref_items = self.memory_stream.unreflected()
        should_ref, _ = ReflectionEngine.should_reflect(unref_items, completed_action)
        if should_ref and unref_items:
            self.dispositions, gen_refs = ReflectionEngine.reflect_memory_items(
                unreflected=unref_items,
                current_dispositions=self.dispositions,
                traits=self.traits,
                self_id=self.agent_id,
                is_drought=is_drought,
                granary_wheat=granary_wheat,
            )
            cur_step = self._last_village_obs.step if self._last_village_obs else 0
            for ref_dict in gen_refs:
                self.memory_stream.create_and_add(
                    step=cur_step,
                    kind="reflection",
                    actors=ref_dict["actors"],
                    summary=ref_dict["summary"],
                    tags=ref_dict["tags"],
                    valence=ref_dict["valence"],
                    fixed_importance=ref_dict["importance"],
                    evidence_of=ref_dict["evidence_of"],
                )
            crime_beliefs = ReflectionEngine.detect_crime_patterns(
                self.memory_stream.all_items(),
                self.agent_id,
                cur_step,
            )
            for b in crime_beliefs:
                if not any(m.summary == b["summary"] for m in self.memory_stream.all_items()):
                    self.memory_stream.create_and_add(
                        step=cur_step,
                        kind="belief",
                        actors=b["actors"],
                        location=b["location"],
                        summary=b["summary"],
                        tags=b["tags"],
                        fixed_importance=b["importance"],
                        evidence_of=b["evidence_of"],
                    )
            self.memory_stream.mark_reflected([m.mid for m in unref_items])

    def choose_dialogue_act(
        self,
        partner_id: str,
        current_affinity: float,
        p_overheard: float = 0.0,
        partner_is_high_status: bool = False,
    ) -> DialogueAct:
        """Selects optimal communicative act during a conversation turn (§3.2)."""
        candidate_acts: list[DialogueAct] = [
            DialogueAct(
                act_type=DialogueActType.SMALL_TALK,
                speaker_id=self.agent_id,
                target_id=partner_id,
                valence=0.20,
                utterance_hint="Casual friendly small talk",
            ),
            DialogueAct(
                act_type=DialogueActType.FAREWELL,
                speaker_id=self.agent_id,
                target_id=partner_id,
                valence=0.0,
                utterance_hint="Polite goodbye",
            ),
        ]

        # Consider GOSSIP or SHARE_NEWS if holding beliefs about other villagers
        beliefs = self.belief_store.all_beliefs()
        for b in beliefs:
            if b.subject not in (self.agent_id, partner_id):
                candidate_acts.append(
                    DialogueAct(
                        act_type=DialogueActType.GOSSIP if b.valence < 0 else DialogueActType.SHARE_NEWS,
                        speaker_id=self.agent_id,
                        target_id=partner_id,
                        subject_id=b.subject,
                        claim=b.claim,
                        valence=b.valence,
                        is_risky=(b.valence < -0.30),
                        utterance_hint=f"Discussing {b.subject}'s {b.claim}",
                    )
                )
                break

        # If holding grudge toward partner, consider COMPLAIN
        disp = self.dispositions.get(partner_id)
        if disp and disp.grudge > 0.40:
            candidate_acts.append(
                DialogueAct(
                    act_type=DialogueActType.COMPLAIN,
                    speaker_id=self.agent_id,
                    target_id=partner_id,
                    valence=-0.50,
                    utterance_hint="Venting frustration about past behavior",
                )
            )

        best_act = candidate_acts[0]
        best_u = -9999.0
        for act in candidate_acts:
            u = compute_dialogue_act_utility(
                act,
                traits=self.traits,
                w_soc=self.utility_weights.w_soc,
                w_rel=self.utility_weights.w_rel if hasattr(self.utility_weights, "w_rel") else 10.0,
                current_affinity=current_affinity,
                speaker_pad=self.affective_state.to_tuple(),
                p_overheard=p_overheard,
                is_subject_high_status=partner_is_high_status,
            )
            if u > best_u:
                best_u = u
                best_act = act

        self.last_dialogue_act = best_act
        return best_act

    def reset(self) -> None:
        """Resets internal and belief state for a new episode or replay."""
        super().reset()
        self._last_village_obs = None
        self.belief_state = VillagerBeliefState()
        self.affective_state = AffectiveStateVector(
            agent_id=self.agent_id,
            valence=self.personality.pad_baseline[0],
            arousal=self.personality.pad_baseline[1],
            dominance=self.personality.pad_baseline[2],
        )
        self.intention = Intention(kind="ROUTINE", commitment=1.0)
        self.needs_engine = NeedsEngine(seed=42)
        self.utility_engine = UtilityEngine(weights=self.base_utility_weights)
        self.intention_scores = {}
        self.attention_tracker = HabituationTracker()
        self.attention_engine = SalienceAttentionEngine(self.attention_tracker)
        self.pending_alerts = []
        self.notices = []
        self.unreflected_episodes = []
        self.memory_stream = MemoryStream(agent_id=self.agent_id, capacity=300)
        self.belief_store = BeliefStore(agent_id=self.agent_id)
        self.active_conversation = None
        self.last_dialogue_act = None
        self.last_gaze_pos = None
        self.last_gaze_reason = ""
        self.sabha_summoned = False
        self.cumulative_reward = 0.0
        self.last_trace = None
        self.last_evaluations = []
