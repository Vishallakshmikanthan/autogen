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
        self._init_agents()
        return self.get_full_state_payload()
        self.sabha_ticks_remaining = 0
        self.traces_by_agent = {}
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
            for ag in self.agents.values():
                if is_night or ag.intention.kind in ("REST_HUT", "REST"):
                    ag.reflect_nightly()

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
                aid: self.agents[aid].intention.kind
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
                "interrupted_by": agent.pending_alerts[0].percept.event.kind if agent.pending_alerts else None,
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
            "step_outcomes": [o.model_dump() for o in self.world.step_outcomes],
            "cinematic_export": self.world.export_village_world_json(),
        }
