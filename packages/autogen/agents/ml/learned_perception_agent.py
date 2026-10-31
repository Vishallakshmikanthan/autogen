"""Learned Perception Agent powered by Classical ML models (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.ml.applications import LearnedPerceptionComponent
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position


class LearnedPerceptionAgent(BaseAgent):
    """Agent using trained ML models (LearnedPerceptionComponent) to evaluate cell safety and yield."""

    DIRECTIONS: tuple[tuple[ActionType, int, int], ...] = (
        (ActionType.UP, 0, 1),
        (ActionType.DOWN, 0, -1),
        (ActionType.LEFT, -1, 0),
        (ActionType.RIGHT, 1, 0),
    )

    def __init__(
        self,
        agent_id: str,
        perception_component: LearnedPerceptionComponent,
        capabilities: AgentCapabilities | None = None,
        danger_threshold: float = 0.40,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.perception_component = perception_component
        self.danger_threshold = danger_threshold
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "LearnedPerceptionPolicy")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_pos = self._last_observation.position
        evaluations: list[dict[str, Any]] = []

        best_action = ActionType.STAY
        min_danger = 1.0
        max_yield = -np.inf

        for action_type, dx, dy in self.DIRECTIONS:
            target_pos = Position(current_pos.x + dx, current_pos.y + dy)
            # Feature extraction for target cell: [x_norm, y_norm, is_obstacle]
            feat = np.array([float(target_pos.x), float(target_pos.y), 0.0], dtype=np.float64)
            pred_danger = self.perception_component.assess_danger(feat)
            pred_yield = self.perception_component.estimate_yield(feat)

            # Record candidate score (utility = yield - danger * 10)
            candidate_score = float(pred_yield - pred_danger * 10.0)
            trace_builder.add_candidate_action(action_type, candidate_score)

            evaluations.append({
                "action": action_type.value,
                "target_pos": (target_pos.x, target_pos.y),
                "predicted_danger": pred_danger,
                "predicted_yield": pred_yield,
            })

            # Preference: avoid dangerous cells (danger < threshold), then maximize yield
            if pred_danger < self.danger_threshold:
                if pred_yield > max_yield:
                    max_yield = pred_yield
                    best_action = action_type
                    min_danger = pred_danger
            elif pred_danger < min_danger:
                min_danger = pred_danger
                best_action = action_type

        trace_builder.set_belief_summary({
            "current_pos": (current_pos.x, current_pos.y),
            "danger_threshold": self.danger_threshold,
            "candidate_evaluations": evaluations,
        })
        trace_builder.add_metric("chosen_predicted_danger", min_danger)
        trace_builder.add_metric(
            "chosen_predicted_yield",
            max_yield if max_yield != -np.inf else 0.0,
        )

        reason = f"ML model selected {best_action.value} with predicted danger={min_danger:.3f}"
        trace_builder.set_selected_action(best_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        return Action(agent_id=self.agent_id, action_type=best_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        self.latest_trace = None
        self._step_counter = 0