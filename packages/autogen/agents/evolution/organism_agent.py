"""Artificial Life Organism Agent with phenotypic metabolism and lineage records (SYS-15, Module M16)."""

from __future__ import annotations

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.evolution.alife import OrganismTraits
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class OrganismAgent(BaseAgent):
    """Ecosystem organism agent governed by phenotypic traits and metabolic energy budgets."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT, ActionType.STAY)

    def __init__(
        self,
        agent_id: str,
        traits: OrganismTraits | None = None,
        generation: int = 0,
        parent_ids: list[str] | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities())
        self.traits = traits or OrganismTraits()
        self.generation = generation
        self.parent_ids = parent_ids or []
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        self._step_counter += 1
        builder = DecisionTraceBuilder(
            agent_id=self.agent_id,
            step_id=self._step_counter,
            algorithm_name="OrganismMetabolicController",
        )

        obs = self._last_observation
        builder.set_observation_summary(obs)
        builder.set_belief_summary({
            "speed": self.traits.speed,
            "metabolic_rate": self.traits.metabolic_rate,
            "vision_range": self.traits.vision_range,
            "generation": self.generation,
        })

        # Decision heuristic: if food visible nearby, move towards it; if low energy, rest/stay
        chosen_action_type = ActionType.STAY
        energy = obs.energy if obs else 50.0
        if obs and obs.visible_cells:
            # Look for resource cell
            food_cells = [cell for cell in obs.visible_cells if getattr(cell, "resource", 0) > 0]
            if food_cells:
                target = food_cells[0]
                dx = target.position.x - obs.position.x
                dy = target.position.y - obs.position.y
                if abs(dx) > abs(dy):
                    chosen_action_type = ActionType.RIGHT if dx > 0 else ActionType.LEFT
                else:
                    chosen_action_type = ActionType.DOWN if dy > 0 else ActionType.UP
            elif energy > 30.0:
                # Random foraging crawl
                dirs = [ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT]
                chosen_action_type = dirs[self._step_counter % len(dirs)]

        for act_type in self.ACTION_MAP:
            score = 1.0 if act_type == chosen_action_type else 0.0
            builder.add_candidate_action(act_type, score)

        builder.set_selected_action(
            chosen_action_type,
            reason=f"Organism foraging action {chosen_action_type.value}",
        )
        builder.add_metric("metabolic_rate", self.traits.metabolic_rate)
        builder.add_metric("speed", self.traits.speed)
        builder.add_metric("generation", self.generation)

        action = Action(agent_id=self.agent_id, action_type=chosen_action_type)
        trace = builder.build()
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self._step_counter = 0
