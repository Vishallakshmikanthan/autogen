"""Bayesian Hazard Agent tracking hidden hazards via GridBeliefFilter (SYS-09, Module M10)."""

from __future__ import annotations

from typing import Any

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.probabilistic.grid_belief_filter import GridBeliefFilter
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import (
    DIRECTION_DISPLACEMENTS,
    TERRAIN_PROPERTIES,
    ActionType,
    Position,
)


class BayesianHazardAgent(BaseAgent):
    """Autonomous agent that tracks unseen hazards using a GridBeliefFilter and maximizes expected utility."""

    def __init__(
        self,
        agent_id: str,
        grid_width: int = 15,
        grid_height: int = 15,
        capabilities: AgentCapabilities | None = None,
        hazard_penalty: float = 100.0,
        food_reward: float = 10.0,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(belief=True, memory=True))
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.hazard_penalty = hazard_penalty
        self.food_reward = food_reward

        self.belief_filter = GridBeliefFilter(width=grid_width, height=grid_height)
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        """Phase 1: Updates GridBeliefFilter using sensor observations."""
        super().observe(observation)
        self._step_counter = observation.step
        my_pos = observation.position

        # Check if any other agent (e.g. predator or hostile) is detected in visual range
        detected_in_view = len(observation.visible_agents) > 0
        sensor_radius = 5  # Typical visual perception radius

        # Update belief filter (predict motion + correct sensor)
        self.belief_filter.update((my_pos, sensor_radius, detected_in_view))

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        """Phase 3: Chooses action maximizing expected utility under current spatial belief."""
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "BayesianBeliefFilter")
        trace_builder.set_observation_summary(self._last_observation)

        entropy_val = self.belief_filter.entropy()
        top_states = self.belief_filter.most_likely_states(k=5)

        trace_builder.add_metric("belief_entropy_bits", entropy_val)
        trace_builder.add_metric("top1_prob", top_states[0][1] if top_states else 0.0)

        belief_summary = {
            "entropy": entropy_val,
            "top_states": [((p.x, p.y), float(prob)) for p, prob in top_states],
        }
        trace_builder.set_belief_summary(belief_summary)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_pos = self._last_observation.position

        # Candidate actions: 4 cardinal directions and STAY
        candidates = [ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT, ActionType.STAY]
        utilities: dict[ActionType, float] = {}

        # Look up visible walkable cells and resources
        walkable_map = {c.position: TERRAIN_PROPERTIES[c.terrain].walkable for c in self._last_observation.visible_cells}
        food_map = {c.position: c.food for c in self._last_observation.visible_cells}

        for act in candidates:
            dx, dy = DIRECTION_DISPLACEMENTS[act]
            next_pos = current_pos.offset(dx, dy)

            # Check walkable bounds
            if not (0 <= next_pos.x < self.grid_width and 0 <= next_pos.y < self.grid_height):
                utilities[act] = -1000.0  # Out of bounds
                trace_builder.add_candidate_action(act, utilities[act])
                continue

            if next_pos in walkable_map and not walkable_map[next_pos]:
                utilities[act] = -500.0  # Impassable terrain
                trace_builder.add_candidate_action(act, utilities[act])
                continue

            # Expected utility calculation:
            # EU(a) = ResourceGain - ExpectedHazardRisk - StepCost
            food_val = food_map.get(next_pos, 0.0)
            hazard_prob = self.belief_filter.prob(next_pos)
            expected_hazard_cost = hazard_prob * self.hazard_penalty
            step_cost = 0.5 if act == ActionType.STAY else 1.0

            eu = (food_val * self.food_reward) - expected_hazard_cost - step_cost
            utilities[act] = eu
            trace_builder.add_candidate_action(act, eu)

        best_action = max(candidates, key=lambda a: utilities[a])
        reason = f"Max expected utility EU={utilities[best_action]:.2f} (hazard risk at target: {self.belief_filter.prob(current_pos.offset(*DIRECTION_DISPLACEMENTS[best_action])):.3f})"

        trace_builder.set_selected_action(best_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        return Action(agent_id=self.agent_id, action_type=best_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        """Resets agent state and re-initializes belief filter to uniform prior."""
        self._last_observation = None
        self.belief_filter = GridBeliefFilter(width=self.grid_width, height=self.grid_height)
        self.latest_trace = None
        self._step_counter = 0

