"""VillageSimulationController: Orchestrates VillageWorld, Trust PageRank, Agents, and Dialogue."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from autogen.agents.village.villager_agent import VillagerAgent
from autogen.core.rng import SeedManager
from autogen.core.types import DecisionTrace
from autogen.intelligence.village.persona.llm_bridge import LLMPersonaBridge, PersonaProfile
from autogen.intelligence.village.persona.symbolic_dialogue import VillageDialogueEvent
from autogen.intelligence.village.persona.village_bus import VillageDialogueBus
from autogen.intelligence.village.social_graph import TrustGraphConfig, VillageTrustGraph
from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights
from autogen.simulation.core.contracts import Action
from autogen.simulation.environment.village.village_grid import (
    VillageRole,
    VillageWorld,
    VillageWorldConfig,
)


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
    ) -> None:
        self.world = VillageWorld(config=world_config)
        self.seed_manager = SeedManager(self.world.config.seed)
        self.trust_graph = VillageTrustGraph(
            agent_ids=list(self.world.villagers.keys()),
            config=trust_config,
        )
        self.dialogue_bus = VillageDialogueBus(base_hearing_radius=12.0)
        self.persona_bridge = LLMPersonaBridge(enable_external_llm=enable_llm_persona)

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
            self.agents[aid] = VillagerAgent(
                agent_id=aid,
                name=v_state.name,
                role=v_state.role,
                utility_weights=weights,
            )
            self.personas[aid] = PersonaProfile(
                agent_id=aid,
                name=v_state.name,
                role=v_state.role.value,
                trait_summary=trait_adjectives[idx % len(trait_adjectives)],
                risk_tolerance=theta_risk,
            )

        status_map = self.trust_graph.compute_societal_status()
        self.world.update_societal_statuses(status_map)

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
        self.traces_by_agent = {}
        self._init_agents()
        return self.get_full_state_payload()

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

        # Phase 5: Update Trust Graph from ground-truth witnessed outcomes & recompute PageRank
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

        # Phase 6: Notify agent learning & downstream Persona Dialogue Bus
        positions_map = {aid: v.position for aid, v in self.world.villagers.items() if v.is_alive}

        # Phase 6: Notify agent learning & downstream Persona Dialogue Bus (All 10 Villagers Converse)
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
        payload = {
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
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_full_state_payload(self) -> dict[str, Any]:
        """Serializes the complete village state for the Research Studio and 3D viewer."""
        villager_cards: list[dict[str, Any]] = []
        for aid, v in sorted(self.world.villagers.items()):
            agent = self.agents[aid]
            hut = self.world.huts.get(v.home_hut_id)
            trace = self.traces_by_agent.get(aid)
            villager_cards.append(
                {
                    "agent_id": aid,
                    "name": v.name,
                    "role": v.role.value,
                    "position": {"x": v.position.x, "y": v.position.y},
                    "pos_3d": list(self.world.grid_to_3d_coords(v.position)),
                    "home_3d": list(hut.world_3d_coords) if hut else [0.0, 0.0],
                    "hunger": round(v.hunger, 2),
                    "hydration": round(v.hydration, 2),
                    "energy": round(v.energy, 2),
                    "health": round(v.health, 2),
                    "coins": round(v.coins, 2),
                    "inventory": v.inventory.to_plain_dict(),
                    "secret_chest_inventory": (
                        hut.chest_inventory.to_plain_dict() if hut else {}
                    ),
                    "societal_status": round(v.societal_status, 4),
                    "status_pct": round(v.societal_status * 100.0, 1),
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
            "villagers": villager_cards,
            "social_graph": self.trust_graph.to_network_payload(),
            "dialogue_events": [
                e.model_dump() for e in self.dialogue_bus.get_recent_events(limit=20)
            ],
            "step_outcomes": [o.model_dump() for o in self.world.step_outcomes],
            "cinematic_export": self.world.export_village_world_json(),
        }
