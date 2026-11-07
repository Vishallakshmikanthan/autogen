"""VillageSimulationController: Orchestrates VillageWorld, Trust PageRank, Agents, and Dialogue."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from autogen.agents.village.villager_agent import VillagerAgent
from autogen.core.rng import SeedManager
from autogen.core.types import DecisionTrace
from autogen.intelligence.village.contagion import EmotionalContagionEngine
from autogen.intelligence.village.conversation import (
    Conversation,
    ConversationEngine,
    ConvState,
    DialogueActType,
)
from autogen.intelligence.village.gossip import Belief, GossipEngine
from autogen.intelligence.village.institutions_and_lineage import (
    ChildBirthRecord,
    FuneralRecord,
    GramSabhaEngine,
    HouseholdPartnership,
    JusticeRedemptionEngine,
    LifecycleEngine,
    LineageRecord,
    VillageAssemblyRecord,
)
from autogen.intelligence.village.memory_stream import MemoryItem
from autogen.intelligence.village.persona.llm_bridge import LLMPersonaBridge, PersonaProfile
from autogen.intelligence.village.persona.symbolic_dialogue import VillageDialogueEvent
from autogen.intelligence.village.persona.village_bus import VillageDialogueBus
from autogen.intelligence.village.personality import CANONICAL_AGENT_TRAITS, Traits
from autogen.intelligence.village.reflection import ReflectionEngine
from autogen.intelligence.village.relationships import VillageRelationshipStore
from autogen.intelligence.village.social_graph import TrustGraphConfig, VillageTrustGraph
from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights
from autogen.simulation.core.contracts import Action
from autogen.simulation.environment.village.commodities import CommodityType
from autogen.simulation.environment.village.events import WorldEvent
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageRole,
    VillageWorld,
    VillageWorldConfig,
)


class HumanlikeConfig(BaseModel):
    """Ablation toggles for humanlike agent layers (§2, §4, §8, Module M22-H)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_personality: bool = True
    enable_attention: bool = True
    enable_intentions: bool = True
    enable_contagion: bool = True
    enable_reflection: bool = True
    enable_weather: bool = True
    enable_needs: bool = True
    enable_routines: bool = True
    enable_memory_stream: bool = True
    enable_conversation: bool = True
    enable_gossip: bool = True


class VillageSimulationController:
    """High-cohesion controller coordinating the 4 decoupled layers of AUTOGEN Village:
    1. Simulation Physics & Economy (`VillageWorld`)
    2. Social Capital & PageRank Trust Graph (`VillageTrustGraph`)
    3. Rational Utility & Ethics Intelligence (`VillagerAgent`)
    4. Downstream Dual-Mode Persona & Dialogue Bus (`LLMPersonaBridge`, `VillageDialogueBus`)
    """

    def __init__(
        self,
        world_config: VillageWorldConfig | None = None,
        trust_config: TrustGraphConfig | None = None,
        enable_llm_persona: bool = False,
        humanlike_config: HumanlikeConfig | None = None,
    ) -> None:
        self.humanlike_config = humanlike_config or HumanlikeConfig()
        self.world = VillageWorld(config=world_config)
        self.seed_manager = SeedManager(self.world.config.seed)
        self.trust_graph = VillageTrustGraph(
            agent_ids=list(self.world.villagers.keys()),
            config=trust_config,
        )
        self.dialogue_bus = VillageDialogueBus(base_hearing_radius=12.0)
        self.persona_bridge = LLMPersonaBridge(enable_external_llm=enable_llm_persona)
        self.relationships = VillageRelationshipStore()

        self.lifecycle_engine = LifecycleEngine(seed=self.world.config.seed)
        self.gram_sabha_engine = GramSabhaEngine(seed=self.world.config.seed)
        self.justice_engine = JusticeRedemptionEngine(seed=self.world.config.seed)

        self.lineages: dict[str, LineageRecord] = {}
        self.partnerships: list[HouseholdPartnership] = []
        self.child_births: list[ChildBirthRecord] = []
        self.funerals: list[FuneralRecord] = []
        self.active_sabha: VillageAssemblyRecord | None = None
        self.sabha_ticks_remaining: int = 0

        self.agents: dict[str, VillagerAgent] = {}
        self.personas: dict[str, PersonaProfile] = {}
        self.traces_by_agent: dict[str, DecisionTrace] = {}
        self.last_utterances: dict[str, str] = {}
        self.last_mean_contagion_delta: float = 0.0
        self.active_conversations: dict[str, Conversation] = {}
        self._init_agents()

    def _init_agents(self) -> None:
        """Instantiates VillagerAgents with role-grounded utility weights and risk tolerances."""
        rng = self.seed_manager.spawn_stream("villager_traits")
        self.agents = {}
        self.personas = {}

        trait_adjectives = (
            "cooperative",
            "industrious",
            "ambitious",
            "cautious",
            "opportunistic",
            "shrewd",
            "principled",
            "vigilant",
            "steadfast",
            "curious",
        )

        for idx, (aid, v_state) in enumerate(sorted(self.world.villagers.items())):
            if self.humanlike_config.enable_personality:
                agent_traits = CANONICAL_AGENT_TRAITS.get(aid, Traits())
                agent = VillagerAgent(
                    agent_id=aid,
                    name=v_state.name,
                    role=v_state.role,
                    traits=agent_traits,
                )
                theta_risk = agent.utility_weights.theta_risk
            else:
                if v_state.role == VillageRole.GUARD:
                    w_soc = 18.0
                    w_econ = 1.0
                    theta_risk = 0.08
                elif v_state.role == VillageRole.MERCHANT:
                    w_soc = 11.0
                    w_econ = 1.65
                    theta_risk = float(round(rng.uniform(0.28, 0.55), 3))
                elif v_state.role == VillageRole.WANDERER:
                    w_soc = 9.5
                    w_econ = 1.1
                    theta_risk = float(round(rng.uniform(0.35, 0.65), 3))
                else:
                    w_soc = 14.0
                    w_econ = 1.15
                    theta_risk = float(round(rng.uniform(0.15, 0.40), 3))

                weights = VillagerUtilityWeights(
                    w_phys=2.3,
                    w_econ=w_econ,
                    w_soc=w_soc,
                    theta_risk=theta_risk,
                )
                agent = VillagerAgent(
                    agent_id=aid,
                    name=v_state.name,
                    role=v_state.role,
                    utility_weights=weights,
                )

            self.agents[aid] = agent
            self.personas[aid] = PersonaProfile(
                agent_id=aid,
                name=v_state.name,
                role=v_state.role.value,
                trait_summary=trait_adjectives[idx % len(trait_adjectives)],
                risk_tolerance=theta_risk,
            )

        status_map = self.trust_graph.compute_societal_status()
        self.world.update_societal_statuses(status_map)
        self.lineages = self.lifecycle_engine.initialize_canonical_lineage(
            [(aid, v.name) for aid, v in sorted(self.world.villagers.items())]
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Resets the entire village simulation to step 0."""
        eff_seed = seed if seed is not None else self.world.config.seed
        self.world.reset(eff_seed)
        self.seed_manager = SeedManager(eff_seed)
        self.trust_graph = VillageTrustGraph(
            agent_ids=list(self.world.villagers.keys()),
            config=self.trust_graph.config,
        )
        self.dialogue_bus = VillageDialogueBus(base_hearing_radius=12.0)
        self.relationships = VillageRelationshipStore()
        self.lifecycle_engine = LifecycleEngine(seed=eff_seed)
        self.gram_sabha_engine = GramSabhaEngine(seed=eff_seed)
        self.justice_engine = JusticeRedemptionEngine(seed=eff_seed)
        self.partnerships = []
        self.child_births = []
        self.funerals = []
        self.active_sabha = None
        self.sabha_ticks_remaining = 0
        self.traces_by_agent = {}
        self.active_conversations = {}
        self._init_agents()
        return self.get_full_state_payload()

    def convoke_sabha(
        self,
        emergency_reason: str | None = None,
        accused_id: str | None = None,
    ) -> dict[str, Any]:
        """Summons the Gram Sabha Town Assembly at the Town Square amphitheater (§5)."""
        statuses = {aid: v.societal_status for aid, v in self.world.villagers.items() if v.is_alive}
        names = {aid: v.name for aid, v in self.world.villagers.items()}
        offenders: list[tuple[str, str, int]] = []
        if accused_id and accused_id in self.world.villagers:
            v_acc = self.world.villagers[accused_id]
            offenders.append((accused_id, "Granary Theft / Hoarding", max(1, v_acc.crimes_committed)))
        else:
            for aid, v in sorted(self.world.villagers.items()):
                if v.crimes_detected > 0 or v.crimes_committed > 0:
                    offenders.append((aid, "Granary Pilfering & Defection", v.crimes_committed))

        record = self.gram_sabha_engine.convene_sabha(
            step=self.world.step_count,
            granary_wheat=self.world.granary.storage.get(CommodityType.WHEAT),
            well_water=self.world.well.water_level,
            drought_active=self.world.drought_active,
            agent_statuses=statuses,
            agent_names=names,
            trust_lookup=self.trust_graph.get_trust,
            known_offenders=offenders,
            emergency_reason=emergency_reason,
        )
        self.active_sabha = record
        self.sabha_ticks_remaining = 4
        for ag in self.agents.values():
            ag.sabha_summoned = True
        return self.get_full_state_payload()

    def trigger_courtship_and_birth(
        self,
        parent_a_id: str = "villager_arun",
        parent_b_id: str = "villager_meera",
        child_name: str | None = None,
    ) -> dict[str, Any]:
        """Forms a household partnership and spawns a new child villager with inherited traits (§3.2)."""
        if parent_a_id not in self.world.villagers or parent_b_id not in self.world.villagers:
            return self.get_full_state_payload()

        pa = self.world.villagers[parent_a_id]
        pb = self.world.villagers[parent_b_id]
        ag_a = self.agents[parent_a_id]
        ag_b = self.agents[parent_b_id]

        # Ensure mutual trust and partnership record
        self.trust_graph.record_direct_experience(parent_a_id, parent_b_id, is_cooperative=True)
        self.trust_graph.record_direct_experience(parent_b_id, parent_a_id, is_cooperative=True)

        existing_pair = next(
            (
                p
                for p in self.partnerships
                if {p.partner_a_id, p.partner_b_id} == {parent_a_id, parent_b_id}
            ),
            None,
        )
        if existing_pair is None:
            existing_pair = HouseholdPartnership(
                partnership_id=f"household_{len(self.partnerships) + 1}",
                partner_a_id=parent_a_id,
                partner_b_id=parent_b_id,
                shared_hut_id=pa.home_hut_id,
                formed_at_step=self.world.step_count,
                mutual_trust=round(
                    0.5
                    * (
                        self.trust_graph.get_trust(parent_a_id, parent_b_id)
                        + self.trust_graph.get_trust(parent_b_id, parent_a_id)
                    ),
                    4,
                ),
                combined_grain_reserve=round(
                    pa.inventory.get(CommodityType.WHEAT) + pb.inventory.get(CommodityType.WHEAT),
                    2,
                ),
            )
            self.partnerships.append(existing_pair)

        child_idx = len(self.child_births) + 1
        c_id = f"villager_child_{child_idx}"
        c_name = child_name or ("Kiran" if child_idx == 1 else f"Asha-{child_idx}")

        lin_a = self.lineages.get(parent_a_id, LineageRecord(agent_id=parent_a_id, name=pa.name))
        lin_b = self.lineages.get(parent_b_id, LineageRecord(agent_id=parent_b_id, name=pb.name))

        birth_rec, child_lin = self.lifecycle_engine.spawn_child_record(
            step=self.world.step_count,
            child_id=c_id,
            child_name=c_name,
            parent_a=lin_a,
            parent_b=lin_b,
            weights_a=(
                ag_a.utility_weights.theta_risk,
                ag_a.utility_weights.w_soc,
                ag_a.utility_weights.w_econ,
            ),
            weights_b=(
                ag_b.utility_weights.theta_risk,
                ag_b.utility_weights.w_soc,
                ag_b.utility_weights.w_econ,
            ),
        )
        self.child_births.append(birth_rec)
        self.lineages[c_id] = child_lin

        # Expand trust graph with parental trust inheritance
        self.trust_graph.add_agent(c_id, parent_ids=(parent_a_id, parent_b_id))

        # Spawn physical child villager in VillageWorld near parent hut
        hut = self.world.huts.get(pa.home_hut_id)
        spawn_pos = hut.position if hut else pa.position
        v_child = self.world.spawn_custom_villager(
            agent_id=c_id,
            name=c_name,
            role=pa.role,
            position=spawn_pos,
            hunger=90.0,
            hydration=90.0,
            energy=95.0,
            coins=12.0,
        )
        self.world.villagers[c_id] = v_child.model_copy(
            update={
                "archetype_title": f"Gen-{child_lin.generation} Apprentice ({pa.name} & {pb.name})",
                "age_ticks": 12,
                "generation": child_lin.generation,
                "life_stage": "Child",
                "dominant_emotion": "Joy / Gratitude",
                "pad_vector": (0.65, 0.45, 0.20),
            }
        )
        child_weights = VillagerUtilityWeights(
            w_phys=2.3,
            w_econ=birth_rec.inherited_w_econ,
            w_soc=birth_rec.inherited_w_soc,
            theta_risk=birth_rec.inherited_theta_risk,
        )
        self.agents[c_id] = VillagerAgent(
            agent_id=c_id,
            name=c_name,
            role=pa.role,
            utility_weights=child_weights,
        )
        self.personas[c_id] = PersonaProfile(
            agent_id=c_id,
            name=c_name,
            role=pa.role.value,
            trait_summary="youthful",
            risk_tolerance=birth_rec.inherited_theta_risk,
        )
        status_map = self.trust_graph.compute_societal_status()
        self.world.update_societal_statuses(status_map)
        return self.get_full_state_payload()

    def trigger_thunderstorm(self, intensity: float = 0.90) -> list[WorldEvent]:
        """Manually triggers an immediate thunderstorm event across the village world (§2.2)."""
        return self.world.weather_process.trigger_thunderstorm(
            step=self.world.step_count, intensity=intensity
        )

    def step(self) -> dict[str, Any]:
        """Executes one complete tick of the AUTOGEN Village Society."""
        # Phase 1 & 2: Sensory observation and belief update
        for aid in sorted(self.agents.keys()):
            v_state = self.world.villagers[aid]
            if not v_state.is_alive:
                continue
            obs = self.world.get_observation(aid)
            v_obs = self.world.get_village_observation(aid)
            agent = self.agents[aid]
            agent.observe(obs)
            agent.observe_village(v_obs)
            agent.update_belief()

        # Phase 2d: Synchronous emotional contagion pass across neighboring villagers (§3.4)
        if self.humanlike_config.enable_contagion:
            agent_states = {aid: a.affective_state for aid, a in self.agents.items()}
            pos_map = {
                aid: self.world.villagers[aid].position
                for aid in self.agents.keys()
                if self.world.villagers[aid].is_alive
            }
            personalities = {aid: a.personality for aid, a in self.agents.items()}
            next_affective = EmotionalContagionEngine.step_contagion(
                agents=agent_states,
                positions=pos_map,
                personalities=personalities,
                relationships=self.relationships,
            )
            deltas = [
                float(
                    (
                        (next_affective[aid].valence - agent_states[aid].valence) ** 2
                        + (next_affective[aid].arousal - agent_states[aid].arousal) ** 2
                        + (next_affective[aid].dominance - agent_states[aid].dominance) ** 2
                    )
                    ** 0.5
                )
                for aid in agent_states
                if aid in next_affective
            ]
            self.last_mean_contagion_delta = float(sum(deltas) / max(1, len(deltas)))
            for aid, st in next_affective.items():
                self.agents[aid].affective_state = st

        # Phase 3: Deliberation & DecisionTrace emission
        actions: dict[str, Action] = {}
        for aid in sorted(self.agents.keys()):
            v_state = self.world.villagers[aid]
            if not v_state.is_alive:
                continue
            act, trace = self.agents[aid].decide()
            actions[aid] = act
            self.traces_by_agent[aid] = trace

        # Phase 4: Environment physics, economy & crime detection resolution
        self.world.step_village(actions)

        # Phase 5: Update Trust Graph and directed Relationships
        for outcome in self.world.step_outcomes:
            if outcome.target_id:
                if outcome.category in ("coop", "trade"):
                    self.relationships.record_interaction(outcome.actor_id, outcome.target_id, quality=0.40)
                elif outcome.category in ("defect", "crime_caught", "crime_unseen"):
                    self.relationships.record_interaction(outcome.actor_id, outcome.target_id, quality=-0.60)
            if self.humanlike_config.enable_needs:
                contact_q = 0.50 if outcome.action_type == VillageActionType.GREET else (0.30 if outcome.category == "trade" else 0.0)
                if contact_q > 0.0:
                    if outcome.actor_id in self.agents:
                        nst_a = self.agents[outcome.actor_id].needs_engine.state
                        self.agents[outcome.actor_id].needs_engine.state = nst_a.model_copy(
                            update={"contact_score": round(nst_a.contact_score + contact_q, 4)}
                        )
                    if outcome.target_id and outcome.target_id in self.agents:
                        nst_t = self.agents[outcome.target_id].needs_engine.state
                        self.agents[outcome.target_id].needs_engine.state = nst_t.model_copy(
                            update={"contact_score": round(nst_t.contact_score + contact_q, 4)}
                        )
        self.relationships.step_decay()

        for outcome in self.world.step_outcomes:
            if outcome.category == "coop":
                # Public philanthropy / trade boosts trust among witnesses & partners
                observers = (
                    list(outcome.witness_ids)
                    if outcome.witness_ids
                    else list(self.world.villagers.keys())
                )
                if outcome.target_id and outcome.target_id not in observers:
                    observers.append(outcome.target_id)
                self.trust_graph.record_public_event(
                    actor_id=outcome.actor_id,
                    event_type="coop",
                    witness_ids=observers,
                )
            elif outcome.category == "defect":
                observers = (
                    list(outcome.witness_ids)
                    if outcome.witness_ids
                    else list(self.world.villagers.keys())
                )
                self.trust_graph.record_public_event(
                    actor_id=outcome.actor_id,
                    event_type="defect",
                    witness_ids=observers,
                )
            elif outcome.category == "crime_caught":
                # Caught crime is announced at the Town Square pillory
                self.trust_graph.record_public_event(
                    actor_id=outcome.actor_id,
                    event_type="crime_caught",
                    witness_ids=list(self.world.villagers.keys()),
                )

        new_statuses = self.trust_graph.compute_societal_status()
        self.world.update_societal_statuses(new_statuses)

        # Phase 5c: Sleep-time reflection and grudge consolidation (§4.4)
        if self.humanlike_config.enable_reflection:
            is_night = self.world.time_of_day >= 22.0 or self.world.time_of_day <= 5.0
            granary_wheat = float(self.world.granary.storage.get(CommodityType.WHEAT))
            for ag in self.agents.values():
                v_phys = self.world.villagers.get(ag.agent_id)
                comp_act = v_phys.last_action if v_phys else None
                if is_night or ag.intention.kind in ("REST_HUT", "REST") or comp_act == VillageActionType.REST_HUT:
                    ag.reflect_nightly(
                        completed_action=comp_act,
                        is_drought=getattr(self.world, "drought_active", False),
                        granary_wheat=granary_wheat,
                    )

            # Nudge dyadic relationship affinity baseline from reflective dispositions (§4.5)
            for aid, ag in self.agents.items():
                for tid, disp in ag.dispositions.items():
                    cur_aff = self.relationships.get_affinity(aid, tid)
                    target_aff = (1.0 + disp.valence) / 2.0
                    nudge_aff = cur_aff + 0.05 * (target_aff - cur_aff)
                    self.relationships.set_affinity(aid, tid, float(round(nudge_aff, 4)))

        # Phase 5b: Update Justice Forensics, Redemption Service, Lifecycle, and Gram Sabha
        for outcome in self.world.step_outcomes:
            if outcome.category in ("crime_caught", "crime_unseen"):
                v_actor = self.world.villagers.get(outcome.actor_id)
                self.justice_engine.record_forensic_trace(
                    step=self.world.step_count,
                    culprit_id=outcome.actor_id,
                    culprit_name=v_actor.name if v_actor else outcome.actor_id,
                    crime_action=outcome.action_type.value,
                    zone_id=v_actor.current_goal_zone if v_actor else "village_commons",
                    detection_prob=outcome.detection_probability,
                    witness_ids=outcome.witness_ids,
                )
            elif outcome.action_type.value == "COMMUNITY_SERVICE":
                v_actor = self.world.villagers.get(outcome.actor_id)
                self.justice_engine.perform_redemption_service(
                    step=self.world.step_count,
                    actor_id=outcome.actor_id,
                    actor_name=v_actor.name if v_actor else outcome.actor_id,
                    service_type="Granary Commons Restitution",
                    trust_graph=self.trust_graph,
                )
                # Witnessing community service triggers stochastic forgiveness check (§4.4)
                for witness_id in outcome.witness_ids:
                    if witness_id in self.agents and outcome.actor_id in self.agents[witness_id].dispositions:
                        w_agent = self.agents[witness_id]
                        w_disp = w_agent.dispositions[outcome.actor_id]
                        w_rng = self.seed_manager.spawn_stream(f"forgiveness:{witness_id}")
                        new_disp, forgiven = ReflectionEngine.check_forgiveness(
                            disposition=w_disp,
                            traits=w_agent.traits,
                            witnessed_community_service=True,
                            steps_since_harm=self.world.step_count,
                            rng=w_rng,
                        )
                        if forgiven:
                            w_agent.dispositions[outcome.actor_id] = new_disp

        if self.sabha_ticks_remaining > 0:
            self.sabha_ticks_remaining -= 1
            if self.sabha_ticks_remaining == 0:
                for ag in self.agents.values():
                    ag.sabha_summoned = False
        else:
            should_c, reason = self.gram_sabha_engine.should_convoke(
                step=self.world.step_count,
                granary_wheat=self.world.granary.storage.get(CommodityType.WHEAT),
                uninvestigated_crimes=sum(1 for t in self.justice_engine.forensic_traces if not t.investigated),
                force_convoke=any(o.category == "crime_caught" for o in self.world.step_outcomes),
            )
            if should_c:
                self.convoke_sabha(emergency_reason=reason)

        pad_sync: dict[str, tuple[str, tuple[float, float, float]]] = {}
        lifecycle_sync: dict[str, tuple[int, float, str, int]] = {}
        for aid, v in self.world.villagers.items():
            agent_obj = self.agents.get(aid)
            if agent_obj is not None:
                pad_sync[aid] = (
                    agent_obj.affective_state.dominant_emotion.value,
                    agent_obj.affective_state.to_tuple(),
                )
            new_age = v.age_ticks + 1
            max_e = self.lifecycle_engine.compute_max_energy(new_age)
            l_stage = self.lifecycle_engine.compute_life_stage(new_age, is_alive=v.is_alive).value
            lin = self.lineages.get(aid)
            gen = lin.generation if lin else v.generation
            if lin is not None:
                self.lineages[aid] = lin.model_copy(
                    update={
                        "age_ticks": new_age,
                        "life_stage": self.lifecycle_engine.compute_life_stage(new_age, is_alive=v.is_alive),
                        "max_energy_cap": max_e,
                    }
                )
            lifecycle_sync[aid] = (new_age, max_e, l_stage, gen)

        self.world.update_affective_and_lifecycle_states(pad_sync, lifecycle_sync)

        # Phase 5d: Multi-tick Conversation State Machine & Lossy Gossip Transmission (§3.1, §3.3)
        if self.humanlike_config.enable_conversation:
            ended_convs: list[str] = []
            for conv_id, conv in sorted(self.active_conversations.items()):
                p1, p2 = conv.participants
                v1 = self.world.villagers.get(p1)
                v2 = self.world.villagers.get(p2)

                if not v1 or not v1.is_alive or not v2 or not v2.is_alive:
                    conv.state = ConvState.ABANDONED
                    conv.abandoned_by = p1 if (not v1 or not v1.is_alive) else p2
                    ConversationEngine.settle_conversation(
                        conv, self.relationships, step=self.world.step_count
                    )
                    ended_convs.append(conv_id)
                    continue

                dist = v1.position.manhattan_distance(v2.position)
                if dist > 5:
                    conv.state = ConvState.ABANDONED
                    conv.abandoned_by = p1
                    ConversationEngine.settle_conversation(
                        conv, self.relationships, step=self.world.step_count
                    )
                    ended_convs.append(conv_id)
                    continue

                ag1 = self.agents[p1]
                ag2 = self.agents[p2]
                active_alerts = list(ag1.pending_alerts) + list(ag2.pending_alerts)

                speaker_id = conv.turn_owner
                listener_id = conv.other_participant(speaker_id)
                speaker_agent = self.agents[speaker_id]
                listener_agent = self.agents[listener_id]
                speaker_v = self.world.villagers[speaker_id]
                listener_v = self.world.villagers[listener_id]

                p_overheard = 0.0
                for bystander_id, bystander_v in self.world.villagers.items():
                    if bystander_id in (p1, p2) or not bystander_v.is_alive:
                        continue
                    b_dist = speaker_v.position.manhattan_distance(bystander_v.position)
                    aud = max(0.0, 1.0 - (b_dist / 8.0))
                    if aud > p_overheard:
                        p_overheard = aud

                cur_aff = self.relationships.get_affinity(speaker_id, listener_id)

                chosen_act = speaker_agent.choose_dialogue_act(
                    partner_id=listener_id,
                    current_affinity=cur_aff,
                    p_overheard=p_overheard,
                    partner_is_high_status=(listener_v.societal_status > 0.12),
                )

                new_state, emitted_act, reason = ConversationEngine.step_conversation(
                    conv,
                    active_alerts=active_alerts,
                    relationships=self.relationships,
                    chosen_act=chosen_act,
                )

                if emitted_act is not None:
                    speaker_agent.last_dialogue_act = emitted_act

                    if (
                        emitted_act.act_type in (DialogueActType.GOSSIP, DialogueActType.SHARE_NEWS)
                        and self.humanlike_config.enable_gossip
                    ):
                        cand_belief: Belief | None = None
                        if emitted_act.subject_id and emitted_act.claim:
                            cand_belief = speaker_agent.belief_store.get(
                                emitted_act.subject_id, emitted_act.claim
                            )
                        if cand_belief is None:
                            all_b = speaker_agent.belief_store.all_beliefs()
                            if all_b:
                                cand_belief = all_b[0]

                        if cand_belief is not None:
                            g_rng = self.seed_manager.spawn_stream(
                                f"gossip:{conv_id}:{conv.turns}:{self.world.step_count}"
                            )
                            accepted_belief, accepted, g_reason = GossipEngine.transmit(
                                speaker_id=speaker_id,
                                listener_id=listener_id,
                                belief=cand_belief,
                                speaker_arousal=speaker_agent.affective_state.arousal,
                                speaker_affinity_to_subject=self.relationships.get_affinity(
                                    speaker_id, cand_belief.subject
                                ),
                                listener_trust_in_speaker=self.trust_graph.get_trust(
                                    listener_id, speaker_id
                                ),
                                listener_status_of_speaker=speaker_v.societal_status,
                                listener_affinity_for_speaker=self.relationships.get_affinity(
                                    listener_id, speaker_id
                                ),
                                listener_openness=listener_agent.traits.openness,
                                listener_prior_belief=listener_agent.belief_store.get(
                                    cand_belief.subject, cand_belief.claim
                                ),
                                rng=g_rng,
                                step=self.world.step_count,
                            )
                            if accepted and accepted_belief is not None:
                                listener_agent.belief_store.put(accepted_belief)
                                if self.humanlike_config.enable_memory_stream:
                                    listener_agent.memory_stream.add(
                                        MemoryItem(
                                            mid=f"{listener_id}:{self.world.step_count}:{len(listener_agent.memory_stream)}",
                                            step=self.world.step_count,
                                            kind="conversation",
                                            actors=tuple(sorted((speaker_id, listener_id))),
                                            location=speaker_v.current_goal_zone or "village_commons",
                                            summary=f"Heard rumor from {speaker_v.name}: {accepted_belief.subject} engaged in {accepted_belief.claim}.",
                                            tags=tuple(sorted(("gossip", accepted_belief.claim, accepted_belief.subject))),
                                            valence=accepted_belief.valence,
                                            pad=listener_agent.affective_state.to_tuple(),
                                            importance=0.60,
                                            last_access_step=self.world.step_count,
                                        )
                                    )

                if new_state in (
                    ConvState.CLOSING,
                    ConvState.INTERRUPTED,
                    ConvState.ABANDONED,
                    ConvState.IDLE,
                ):
                    settle_res = ConversationEngine.settle_conversation(
                        conv, self.relationships, step=self.world.step_count
                    )
                    if self.humanlike_config.enable_memory_stream:
                        for pid in (p1, p2):
                            other_p = p2 if pid == p1 else p1
                            other_name = (
                                self.world.villagers[other_p].name
                                if other_p in self.world.villagers
                                else other_p
                            )
                            self.agents[pid].memory_stream.add(
                                MemoryItem(
                                    mid=f"{pid}:{self.world.step_count}:{len(self.agents[pid].memory_stream)}",
                                    step=self.world.step_count,
                                    kind="conversation",
                                    actors=tuple(sorted((p1, p2))),
                                    location=self.world.villagers[pid].current_goal_zone or "village_commons",
                                    summary=f"Conversed with {other_name}: {settle_res.get('outcome', 'finished')}.",
                                    tags=tuple(sorted(("social", "conversation", other_p))),
                                    valence=float(settle_res.get("quality", 0.0)),
                                    pad=self.agents[pid].affective_state.to_tuple(),
                                    importance=0.45,
                                    last_access_step=self.world.step_count,
                                )
                            )
                    ended_convs.append(conv_id)

            for cid in ended_convs:
                if cid in self.active_conversations:
                    c_obj = self.active_conversations[cid]
                    for pid in c_obj.participants:
                        if pid in self.agents and self.agents[pid].active_conversation is c_obj:
                            self.agents[pid].active_conversation = None
                    del self.active_conversations[cid]

            free_agents = [
                aid
                for aid, a in sorted(self.agents.items())
                if a.active_conversation is None and self.world.villagers[aid].is_alive
            ]
            for i in range(len(free_agents)):
                aid1 = free_agents[i]
                if self.agents[aid1].active_conversation is not None:
                    continue
                v1 = self.world.villagers[aid1]
                for j in range(i + 1, len(free_agents)):
                    aid2 = free_agents[j]
                    if self.agents[aid2].active_conversation is not None:
                        continue
                    v2 = self.world.villagers[aid2]
                    dist = v1.position.manhattan_distance(v2.position)
                    is_greet = (
                        v1.last_action == VillageActionType.GREET
                        or v2.last_action == VillageActionType.GREET
                    )
                    is_waking = not (self.world.time_of_day >= 22.0 or self.world.time_of_day <= 5.0)
                    not_storm = not self.world.weather_process.is_raining
                    if not_storm and is_waking and ((is_greet and dist <= 3) or (dist <= 2 and v1.energy > 20 and v2.energy > 20)):
                        initiator = (
                            aid1
                            if (v1.last_action == VillageActionType.GREET or self.agents[aid1].traits.extraversion >= self.agents[aid2].traits.extraversion)
                            else aid2
                        )
                        recipient = aid2 if initiator == aid1 else aid1
                        c_id = f"conv_{initiator}_{recipient}_{self.world.step_count}"
                        new_c = ConversationEngine.start_conversation(
                            conv_id=c_id,
                            initiator_id=initiator,
                            recipient_id=recipient,
                            step=self.world.step_count,
                        )
                        self.active_conversations[c_id] = new_c
                        self.agents[initiator].active_conversation = new_c
                        self.agents[recipient].active_conversation = new_c
                        break

        # Phase 6: Notify agent learning & downstream Persona Dialogue Bus (All Villagers Converse)
        positions_map = {aid: v.position for aid, v in self.world.villagers.items() if v.is_alive}

        agents_context: list[dict[str, Any]] = []
        peer_data: dict[str, dict[str, Any]] = {}

        act_desc_map = {
            "HARVEST_FIELD": "harvesting ripe golden wheat in the southern fields",
            "DRAW_WATER": "drawing cool fresh water from the town well",
            "TRADE_MARKET": "trading and haggling goods at the central market square",
            "TITHE_GRANARY": "tithing a sheaf of wheat to the Granary Commons",
            "STEAL_GRANARY": "covertly taking grain from the Granary Commons",
            "STEAL_HUT_CHEST": "stealing provisions from a neighbor's storage chest",
            "PATROL_GUARD": "patrolling the stone crossroads and guarding granary chests",
            "FORAGE_FOREST": "gathering firewood and wild berries near the forest",
            "REST_HUT": "resting after a hard day's labor and locking the hut",
            "CLAIM_FAMINE_RELIEF": "requesting emergency famine relief from the granary",
            "MILL_WHEAT": "grinding wheat into fine flour and baking bread at the Old Windmill",
            "FORGE_TOOLS": "forging iron tools at the Blacksmith Smithy anvil",
            "BREW_HEALING_BROTH": "brewing medicinal herb broth at the Apothecary hearth",
            "PRESIDE_SABHA": "presiding over the Gram Sabha Town Assembly",
            "ATTEND_SABHA": "deliberating in the Gram Sabha council circle at Town Square",
            "SING_BALLAD": "performing a lute ballad chronicling village deeds",
            "INVESTIGATE_CRIME": "inspecting forensic footprints and interviewing witnesses",
            "COMMUNITY_SERVICE": "performing public community service for rehabilitation",
            "COURTSHIP_BOND": "strengthening household kinship and courtship bonds",
            "MOURN_AT_CHAPEL": "lighting a memorial candle at St. Aurelia Chapel",
        }

        for aid in sorted(self.agents.keys()):
            v_state = self.world.villagers[aid]
            if not v_state.is_alive:
                continue

            agent = self.agents[aid]
            agent.learn(
                reward=v_state.hunger * 0.02 + v_state.societal_status * 2.0,
                terminal=not v_state.is_alive,
            )

            saved_trace = self.traces_by_agent.get(aid)
            v_act = (
                str(saved_trace.rationale.get("village_action", "STAY"))
                if saved_trace
                else "STAY"
            )

            # Find nearest visible peer as conversation recipient
            nearest_peer_id: str | None = None
            nearest_peer_name: str | None = None
            best_dist = 999
            for other_id, other_v in sorted(self.world.villagers.items()):
                if other_id == aid or not other_v.is_alive:
                    continue
                d = v_state.position.manhattan_distance(other_v.position)
                if d <= 12 and d < best_dist:
                    best_dist = d
                    nearest_peer_id = other_id
                    nearest_peer_name = other_v.name

            act_desc = act_desc_map.get(v_act, v_act.replace("_", " ").lower())
            trust_val = (
                self.trust_graph.get_trust(aid, nearest_peer_id)
                if nearest_peer_id
                else 0.5
            )

            peer_data[aid] = {
                "nearest_peer_id": nearest_peer_id,
                "nearest_peer_name": nearest_peer_name,
                "action": v_act,
                "trust_score": trust_val,
            }

            agents_context.append(
                {
                    "agent_id": aid,
                    "name": self.personas[aid].name,
                    "role": self.personas[aid].role,
                    "action": v_act,
                    "action_desc": act_desc,
                    "recipient_name": nearest_peer_name,
                    "hunger": v_state.hunger,
                    "coins": v_state.coins,
                }
            )

        # Unified batch dialogue generation for all 10 villagers in ONE single call
        dialogues = self.persona_bridge.generate_village_dialogues_batch(agents_context)

        for aid, data in peer_data.items():
            v_state = self.world.villagers[aid]
            saved_trace = self.traces_by_agent.get(aid)
            if saved_trace is None:
                continue

            ag_obj = self.agents[aid]
            if ag_obj.last_dialogue_act is not None and ag_obj.last_dialogue_act.utterance_hint:
                utterance = f"[{ag_obj.last_dialogue_act.act_type.value}] {ag_obj.last_dialogue_act.utterance_hint}"
            else:
                utterance = dialogues.get(
                    aid,
                    self.persona_bridge._make_symbolic_villager_dialogue(
                        self.personas[aid].name,
                        self.personas[aid].role,
                        data["action"],
                        data["nearest_peer_name"],
                    ),
                )
            self.last_utterances[aid] = utterance

            d_event = VillageDialogueEvent(
                step=self.world.step_count,
                speaker_id=aid,
                speaker_name=self.personas[aid].name,
                speaker_role=self.personas[aid].role,
                recipient_id=data["nearest_peer_id"],
                recipient_name=data["nearest_peer_name"],
                utterance=utterance,
                action_chosen=data["action"],
                is_unethical="STEAL" in data["action"],
                mode="nemotron_llm" if self.persona_bridge.is_llm_active else "symbolic",
                position=(v_state.position.x, v_state.position.y),
                trust_score=round(data["trust_score"], 3),
            )
            self.dialogue_bus.broadcast(
                event=d_event,
                agent_positions=positions_map,
                trust_graph=self.trust_graph,
            )

        return self.get_full_state_payload()

    def compute_canonical_hash(self) -> str:
        """Computes a deterministic SHA-256 hash over world state, inventories, and trust graph."""
        payload: dict[str, Any] = {
            "step": self.world.step_count,
            "time_of_day": round(self.world.time_of_day, 4),
            "well_water": round(self.world.well.water_level, 4),
            "granary": self.world.granary.storage.to_plain_dict(),
            "villagers": [
                {
                    "id": aid,
                    "pos": [v.position.x, v.position.y],
                    "h": round(v.hunger, 4),
                    "t": round(v.hydration, 4),
                    "e": round(v.energy, 4),
                    "coins": round(v.coins, 4),
                    "status": round(v.societal_status, 5),
                    "inv": v.inventory.to_plain_dict(),
                    "act": v.last_action.value,
                }
                for aid, v in sorted(self.world.villagers.items())
            ],
        }
        if self.humanlike_config.enable_intentions:
            payload["intentions"] = {
                aid: f"{self.agents[aid].intention.kind}:{self.agents[aid].intention.status}"
                for aid in sorted(self.agents.keys())
            }
        if self.humanlike_config.enable_needs:
            payload["needs_phys"] = {
                aid: round(self.agents[aid].needs_engine.state.d_phys, 4)
                for aid in sorted(self.agents.keys())
            }
        if self.humanlike_config.enable_conversation:
            payload["conversations"] = {
                cid: f"{c.state.value}:{c.turn_owner}:{c.turns}"
                for cid, c in sorted(self.active_conversations.items())
            }
        if self.humanlike_config.enable_gossip:
            payload["belief_digests"] = {
                aid: self.agents[aid].belief_store.digest()
                for aid in sorted(self.agents.keys())
            }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_full_state_payload(self) -> dict[str, Any]:
        """Serializes the complete village state for the Research Studio and 3D viewer."""
        villager_cards: list[dict[str, Any]] = []
        for aid, v in sorted(self.world.villagers.items()):
            agent = self.agents[aid]
            hut = self.world.huts.get(v.home_hut_id)
            trace = self.traces_by_agent.get(aid)
            pad_st = agent.affective_state
            lin = self.lineages.get(aid)

            # Humanlike telemetry fields (§7)
            gaze_payload = {
                "target_pos": [agent.last_gaze_pos.x, agent.last_gaze_pos.y] if agent.last_gaze_pos else [v.position.x, v.position.y],
                "target_id": None,
                "reason": agent.last_gaze_reason or "ambient",
            }
            alert_lvl = round(agent.pending_alerts[0].salience, 3) if agent.pending_alerts else 0.0
            emote_str = pad_st.dominant_emotion.value.lower().split()[0]
            intention_payload = {
                "kind": agent.intention.kind,
                "progress": round(agent.intention.progress, 2),
                "commitment": round(agent.intention.commitment, 2),
                "step_index": agent.intention.step_index,
                "status": agent.intention.status,
                "score": round(agent.intention.score, 3),
                "interrupted_by": (
                    agent.pending_alerts[0].percept.event.kind if agent.pending_alerts else None
                ),
            }
            needs_payload = {
                "deficits": {
                    "phys": round(agent.needs_engine.state.d_phys, 4),
                    "safety": round(agent.needs_engine.state.d_safety, 4),
                    "belonging": round(agent.needs_engine.state.d_belonging, 4),
                    "esteem": round(agent.needs_engine.state.d_esteem, 4),
                    "growth": round(agent.needs_engine.state.d_growth, 4),
                },
                "gates": {
                    "phys": round(agent.needs_engine.state.g_phys, 4),
                    "safety": round(agent.needs_engine.state.g_safety, 4),
                    "belonging": round(agent.needs_engine.state.g_belonging, 4),
                    "esteem": round(agent.needs_engine.state.g_esteem, 4),
                    "growth": round(agent.needs_engine.state.g_growth, 4),
                },
                "weights": {
                    "phys": round(agent.utility_weights.w_phys, 4),
                    "soc": round(agent.utility_weights.w_soc, 4),
                    "econ": round(agent.utility_weights.w_econ, 4),
                },
                "satiation": dict(agent.needs_engine.state.satiation),
            }
            thought_str = "Observing village activities calmly."
            if agent.dispositions:
                first_disp = list(agent.dispositions.values())[0]
                other_name = self.world.villagers[first_disp.target_id].name if first_disp.target_id in self.world.villagers else first_disp.target_id
                thought_str = ReflectionEngine.render_thought(first_disp, other_name)

            villager_cards.append(
                {
                    "agent_id": aid,
                    "name": v.name,
                    "role": v.role.value,
                    "archetype_title": v.archetype_title,
                    "position": {"x": v.position.x, "y": v.position.y},
                    "pos_3d": list(self.world.grid_to_3d_coords(v.position)),
                    "home_3d": list(hut.world_3d_coords) if hut else [0.0, 0.0],
                    "hunger": round(v.hunger, 2),
                    "hydration": round(v.hydration, 2),
                    "energy": round(v.energy, 2),
                    "max_energy": round(v.max_energy, 2),
                    "health": round(v.health, 2),
                    "coins": round(v.coins, 2),
                    "inventory": v.inventory.to_plain_dict(),
                    "secret_chest_inventory": (
                        hut.chest_inventory.to_plain_dict() if hut else {}
                    ),
                    "societal_status": round(v.societal_status, 4),
                    "status_pct": round(v.societal_status * 100.0, 1),
                    "age_ticks": v.age_ticks,
                    "generation": v.generation,
                    "life_stage": v.life_stage,
                    "dominant_emotion": pad_st.dominant_emotion.value,
                    "pad_vector": list(pad_st.to_vector()),
                    "movement_speed_multiplier": pad_st.movement_speed_multiplier,
                    "aura_color_hex": pad_st.aura_color_hex,
                    "parent_ids": list(lin.parent_ids) if lin else [],
                    "spouse_id": lin.spouse_id if lin else None,
                    "in_pillory_ticks": v.in_pillory_ticks,
                    "crimes_committed": v.crimes_committed,
                    "crimes_detected": v.crimes_detected,
                    "tithes_contributed": round(v.tithes_contributed, 2),
                    "last_action": v.last_action.value,
                    "last_utterance": self.last_utterances.get(aid, ""),
                    "current_goal_zone": v.current_goal_zone,
                    "utility_weights": agent.utility_weights.model_dump(),
                    "believed_detection_risk": round(
                        agent.belief_state.believed_detection_risk, 4
                    ),
                    "episodic_memories": list(agent.belief_state.episodic_memories[-6:]),
                    "decision_trace": trace.model_dump() if trace else None,
                    "gaze": gaze_payload,
                    "alert_level": alert_lvl,
                    "emote": emote_str,
                    "intention": intention_payload,
                    "thought": thought_str,
                    "relationships": self.relationships.get_all_for_agent(aid),
                    "needs": needs_payload,
                    "memory": {
                        "stream_size": len(agent.memory_stream),
                        "unreflected_count": len(agent.memory_stream.unreflected()),
                        "recent_reflections": [m.summary for m in agent.memory_stream.all_items() if m.kind == "reflection"][-3:],
                    },
                    "conversation": {
                        "active": agent.active_conversation is not None,
                        "conv_id": agent.active_conversation.conv_id if agent.active_conversation else None,
                        "partner_id": agent.active_conversation.other_participant(aid) if agent.active_conversation else None,
                        "state": agent.active_conversation.state.value if agent.active_conversation else "IDLE",
                        "turn_owner": agent.active_conversation.turn_owner if agent.active_conversation else None,
                        "turns": agent.active_conversation.turns if agent.active_conversation else 0,
                        "last_act": agent.last_dialogue_act.act_type.value if agent.last_dialogue_act else None,
                        "engagement": round(agent.active_conversation.engagement.get(aid, 0.0), 3) if agent.active_conversation else 0.0,
                    },
                    "beliefs": [
                        {
                            "subject": b.subject,
                            "claim": b.claim,
                            "valence": round(b.valence, 3),
                            "confidence": round(b.confidence, 3),
                            "source_chain": list(b.source_chain),
                            "first_hand": b.first_hand,
                            "times_heard": b.times_heard,
                            "hop_count": b.hop_count,
                        }
                        for b in agent.belief_store.all_beliefs()[:5]
                    ],
                    "beliefs_count": len(agent.belief_store),
                }
            )

        return {
            "step": self.world.step_count,
            "state_hash": self.compute_canonical_hash(),
            "time_of_day": self.world.time_of_day,
            "ambient_light": self.world.get_ambient_light(),
            "drought_active": self.world.drought_active,
            "grid": {"width": self.world.config.width, "height": self.world.config.height},
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "zone_type": z.zone_type.value,
                    "label": z.label,
                    "x_min": z.x_min,
                    "y_min": z.y_min,
                    "x_max": z.x_max,
                    "y_max": z.y_max,
                    "center": {"x": z.center.x, "y": z.center.y},
                    "world_3d": list(z.world_3d_coords),
                    "radius_3d": z.radius_3d,
                }
                for z in self.world.zones.values()
            ],
            "economy": {
                "clearing_prices": {
                    k.value: round(p, 3) for k, p in self.world.market.clearing_prices.items()
                },
                "granary_storage": self.world.granary.storage.to_plain_dict(),
                "well_water_level": round(self.world.well.water_level, 2),
                "recent_trades": [t.model_dump() for t in self.world.market.recent_trades[-10:]],
            },
            "institutions": {
                "sabha_active": self.sabha_ticks_remaining > 0,
                "sabha_ticks_remaining": self.sabha_ticks_remaining,
                "active_sabha": self.active_sabha.model_dump() if self.active_sabha else None,
                "assembly_history": [
                    a.model_dump() for a in self.gram_sabha_engine.assemblies[-5:]
                ],
                "enacted_policies": list(self.gram_sabha_engine.enacted_policies),
                "mandatory_tithe_rate": self.gram_sabha_engine.mandatory_tithe_rate,
            },
            "demographics": {
                "lineages": {k: r.model_dump() for k, r in self.lineages.items()},
                "partnerships": [p.model_dump() for p in self.partnerships],
                "child_births": [b.model_dump() for b in self.child_births],
                "funerals": [f.model_dump() for f in self.funerals],
            },
            "justice": {
                "forensic_traces": [
                    ft.model_dump() for ft in self.justice_engine.forensic_traces[-10:]
                ],
                "redemption_records": {
                    f"{rec.actor_id}_{idx}": rec.model_dump()
                    for idx, rec in enumerate(self.justice_engine.redemption_records)
                },
            },
            "villagers": villager_cards,
            "social_graph": self.trust_graph.to_network_payload(),
            "dialogue_events": [
                e.model_dump() for e in self.dialogue_bus.get_recent_events(limit=20)
            ],
            "conversations": {
                "active_count": len(self.active_conversations),
                "active": [
                    {
                        "conv_id": c.conv_id,
                        "participants": list(c.participants),
                        "state": c.state.value,
                        "turn_owner": c.turn_owner,
                        "turns": c.turns,
                        "topic": c.topic_stack[-1] if c.topic_stack else "general",
                        "last_act": c.history[-1].act_type.value if c.history else None,
                    }
                    for c in sorted(self.active_conversations.values(), key=lambda x: x.conv_id)
                ],
            },
            "step_outcomes": [o.model_dump() for o in self.world.step_outcomes],
            "mean_contagion_delta": round(self.last_mean_contagion_delta, 4),
            "cinematic_export": self.world.export_village_world_json(),
        }
