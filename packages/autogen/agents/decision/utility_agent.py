"""Utility Agent maximizing expected utility over belief states (SYS-10, Module M11)."""

from __future__ import annotations

from typing import Any

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.decision.contracts import UtilityWeights
from autogen.intelligence.decision.utility import MultiAttributeUtility
from autogen.intelligence.probabilistic.grid_belief_filter import GridBeliefFilter
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import (
    DIRECTION_DISPLACEMENTS,
    TERRAIN_PROPERTIES,
    ActionType,
    Position,
)


class UtilityAgent(BaseAgent):
    """Decision-theoretic autonomous agent maximizing multi-attribute expected utility under uncertainty."""

    def __init__(
        self,
        agent_id: str,
        grid_width: int = 15,
        grid_height: int = 15,
        weights: UtilityWeights | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(belief=True, planning=True))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.utility_evaluator = MultiAttributeUtility(weights or UtilityWeights())
        self.belief_filter = GridBeliefFilter(width=grid_width, height=grid_height)
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step
        my_pos = observation.position
        detected = len(observation.visible_agents) > 0
        self.belief_filter.update((my_pos, 4, detected))

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "MultiAttributeUtility")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        cur_pos = self._last_observation.position
        candidates = [ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT, ActionType.STAY]
        utilities: dict[ActionType, float] = {}

        walkable_map = {c.position: TERRAIN_PROPERTIES[c.terrain].walkable for c in self._last_observation.visible_cells}
        food_map = {c.position: c.food for c in self._last_observation.visible_cells}

        for act in candidates:
            dx, dy = DIRECTION_DISPLACEMENTS[act]
            next_pos = cur_pos.offset(dx, dy)

            # Check walkable bounds
            if not (0 <= next_pos.x < self.grid_width and 0 <= next_pos.y < self.grid_height):
                utilities[act] = -1000.0
                trace_builder.add_candidate_action(act, utilities[act])
                continue

            if next_pos in walkable_map and not walkable_map[next_pos]:
                utilities[act] = -500.0
                trace_builder.add_candidate_action(act, utilities[act])
                continue

            # Multi-attribute utility evaluation
            energy_val = self._last_observation.energy - (0.5 if act == ActionType.STAY else 1.0)
            health_val = self._last_observation.health
            food_val = food_map.get(next_pos, 0.0)
            hazard_risk = self.belief_filter.prob(next_pos)

            u = self.utility_evaluator.evaluate(
                energy=energy_val,
                health=health_val,
                food=food_val,
                hazard_risk=hazard_risk,
            )
            utilities[act] = u
            trace_builder.add_candidate_action(act, u)

        best_action = max(candidates, key=lambda a: utilities[a])
        reason = f"Max Expected Utility EU({best_action.value}) = {utilities[best_action]:.2f}"
        trace_builder.set_selected_action(best_action, reason)
        trace_builder.add_metric("best_eu", utilities[best_action])
        trace_builder.add_metric("entropy_bits", self.belief_filter.entropy())

        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=best_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        self._last_observation = None
        self.belief_filter = GridBeliefFilter(width=self.grid_width, height=self.grid_height)
        self.latest_trace = None
        self._step_counter = 0
