"""MemoryAugmentedAgent: decorator wrapping any BaseAgent with retrievable memory."""

from __future__ import annotations

from typing import Any

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.baseline.simple_reflex_agent import DIRECTION_DELTAS
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.memory.contracts import EpisodicTransition, MemoryConfig
from autogen.memory.episodic_memory import EpisodicMemory
from autogen.memory.explorer import MemoryExplorerAPI
from autogen.memory.retrieval import CompositeRetrieval, ProximityRetrieval
from autogen.memory.spatial_memory import SpatialMemory
from autogen.memory.working_memory import WorkingMemory
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.state.types import ActionType


class MemoryAugmentedAgent(BaseAgent):
    """Decorator/wrapper equipping any BaseAgent with working, episodic, and spatial memory."""

    def __init__(
        self,
        inner_agent: BaseAgent,
        config: MemoryConfig | None = None,
    ) -> None:
        caps = AgentCapabilities(
            memory=True,
            belief=inner_agent.capabilities.belief,
            planning=inner_agent.capabilities.planning,
            learning=inner_agent.capabilities.learning,
        )
        super().__init__(agent_id=inner_agent.agent_id, capabilities=caps)
        self.inner_agent = inner_agent
        self.config = config or MemoryConfig()

        self.working_memory = WorkingMemory(
            capacity=self.config.working_memory_capacity,
            policy=self.config.eviction_policy,
        )
        self.episodic_memory = EpisodicMemory(capacity=self.config.episodic_memory_capacity)
        self.spatial_memory = SpatialMemory(config=self.config)
        self.explorer = MemoryExplorerAPI(
            self.working_memory, self.episodic_memory, self.spatial_memory
        )

        self._proximity_retriever = ProximityRetrieval()
        self._composite_retriever = CompositeRetrieval()
        self._last_action: Action | None = None
        self._decision_observation: Observation | None = None

    def observe(self, observation: Observation) -> None:
        """Phase 1: Ingests observation into working and spatial memory before inner agent."""
        super().observe(observation)
        self.working_memory.push(observation)
        self.spatial_memory.update_from_observation(observation)
        self.inner_agent.observe(observation)

    def update_belief(self) -> None:
        """Phase 2: Updates internal representations."""
        self.inner_agent.update_belief()

    def decide(self) -> tuple[Action, DecisionTrace]:
        """Phase 3: Queries memory, deliberates, and records memory influence in trace."""
        if self._last_observation is None:
            raise RuntimeError("Agent must observe before deciding.")

        obs = self._last_observation
        pos = (obs.position.x, obs.position.y)

        # 1. Check for immediate visible food in current observation
        visible_food = [c for c in obs.visible_cells if c.food > 0.0]

        # 2. Query spatial memory for known unvisited food
        known_food_cells = self.spatial_memory.get_known_food(
            min_confidence=self.config.confidence_floor
        )
        retrieved_food = self._composite_retriever.retrieve(
            known_food_cells,
            {"position": obs.position, "current_step": obs.step},
            k=3,
        )

        memory_used: list[dict[str, Any]] = [
            {
                "target": (c.position.x, c.position.y),
                "food": c.last_seen_food,
                "confidence": c.confidence,
                "staleness": obs.step - c.last_observed_tick,
            }
            for c in retrieved_food
        ]

        # If current observation has visible food or food at feet, inner agent handles it
        # If no immediate food is in sight, but spatial memory holds high-confidence remembered food:
        navigated_from_memory = False
        action: Action
        trace: DecisionTrace

        cell_lookup = {(c.position.x, c.position.y): c for c in obs.visible_cells}

        if not visible_food and retrieved_food:
            best_target = retrieved_food[0].position
            target_pos = (best_target.x, best_target.y)
            target_dist = abs(target_pos[0] - pos[0]) + abs(target_pos[1] - pos[1])

            best_dir: ActionType | None = None
            best_score = -1.0
            scores: dict[ActionType, float] = dict.fromkeys(
                [
                    ActionType.UP,
                    ActionType.DOWN,
                    ActionType.LEFT,
                    ActionType.RIGHT,
                    ActionType.STAY,
                    ActionType.COLLECT,
                ],
                0.0,
            )

            for act, (dx, dy) in DIRECTION_DELTAS.items():
                nxt = (pos[0] + dx, pos[1] + dy)
                nxt_cell = cell_lookup.get(nxt)
                if nxt_cell and TERRAIN_PROPERTIES[nxt_cell.terrain].walkable:
                    nxt_dist = abs(target_pos[0] - nxt[0]) + abs(target_pos[1] - nxt[1])
                    dir_score = 60.0 + (target_dist - nxt_dist) * 10.0
                    scores[act] = dir_score
                    if dir_score > best_score:
                        best_score = dir_score
                        best_dir = act

            if best_dir is not None:
                navigated_from_memory = True
                action = Action(agent_id=self.agent_id, action_type=best_dir)
                builder = (
                    DecisionTraceBuilder(self.agent_id, obs.step, "memory_augmented_navigation")
                    .set_mechanism("spatial_memory_retrieval")
                    .set_observation_summary(obs)
                    .set_belief_summary(
                        {
                            "remembered_food_count": len(retrieved_food),
                            "target_food": target_pos,
                            "target_confidence": retrieved_food[0].confidence,
                        }
                    )
                    .set_selected_action(
                        best_dir,
                        f"Navigating to remembered food at {target_pos} (conf={retrieved_food[0].confidence:.2f}).",
                    )
                    .add_metric("spatial_memory_size", len(self.spatial_memory))
                    .add_metric("working_memory_size", len(self.working_memory))
                )
                for act, sc in scores.items():
                    builder.add_candidate_action(act, sc)
                trace = builder.build()

        if not navigated_from_memory:
            # Fall back to inner agent deliberation
            action, raw_trace = self.inner_agent.decide()
            # Augment raw trace with memory attribution
            trace_rationale = dict(raw_trace.rationale)
            trace_rationale["memory_used"] = memory_used
            trace_rationale["working_memory_len"] = len(self.working_memory)
            trace_rationale["spatial_memory_len"] = len(self.spatial_memory)

            trace_metrics = dict(raw_trace.metrics)
            trace_metrics["spatial_memory_size"] = len(self.spatial_memory)
            trace_metrics["episodic_memory_size"] = len(self.episodic_memory)

            trace = raw_trace.model_copy(
                update={"rationale": trace_rationale, "metrics": trace_metrics}
            )

        self._last_action = action
        self._decision_observation = obs
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        """Phase 7: Records transition (t, o, a, r, o', done) in episodic memory."""
        if self._decision_observation is not None and self._last_action is not None:
            transition = EpisodicTransition(
                step=self._decision_observation.step,
                state_observation=self._decision_observation,
                action=self._last_action,
                reward=reward,
                next_observation=self._last_observation,
                terminal=terminal,
                metadata={"reward": reward, "agent_id": self.agent_id},
            )
            self.episodic_memory.push(transition)

        self.inner_agent.learn(reward, terminal)

    def reset(self) -> None:
        """Resets working, episodic, and spatial memory alongside inner agent."""
        super().reset()
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.spatial_memory.clear()
        self._last_action = None
        self._decision_observation = None
        self.inner_agent.reset()
