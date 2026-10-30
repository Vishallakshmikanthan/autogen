"""Fluent builder for DecisionTrace satisfying strict explainability requirements."""

from __future__ import annotations

import time
from typing import Any

from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType


class DecisionTraceBuilder:
    """Builder for constructing authentic, verifiable DecisionTrace objects."""

    def __init__(self, agent_id: str, step_id: int, algorithm_name: str) -> None:
        self._agent_id = agent_id
        self._step_id = step_id
        self._algorithm_name = algorithm_name
        self._decision_type = "action_selection"
        self._selected_action: ActionType | None = None
        self._mechanism_name: str = algorithm_name
        self._reason: str = ""
        self._candidate_actions: dict[str, float] = {}
        self._observation_summary: dict[str, Any] = {}
        self._belief_summary: dict[str, Any] = {}
        self._metrics: dict[str, float | int | str] = {}

    def set_decision_type(self, decision_type: str) -> DecisionTraceBuilder:
        self._decision_type = decision_type
        return self

    def set_mechanism(self, mechanism_name: str) -> DecisionTraceBuilder:
        self._mechanism_name = mechanism_name
        return self

    def set_observation_summary(self, observation: Observation | None) -> DecisionTraceBuilder:
        if observation is not None:
            self._observation_summary = {
                "step": observation.step,
                "position": (observation.position.x, observation.position.y),
                "energy": observation.energy,
                "health": observation.health,
                "visible_cells_count": len(observation.visible_cells),
                "visible_agents_count": len(observation.visible_agents),
            }
        return self

    def set_belief_summary(self, summary: dict[str, Any]) -> DecisionTraceBuilder:
        self._belief_summary = dict(summary)
        return self

    def add_candidate_action(
        self,
        action: ActionType,
        score: float,
    ) -> DecisionTraceBuilder:
        self._candidate_actions[action.value] = float(score)
        return self

    def set_selected_action(self, action: ActionType, reason: str = "") -> DecisionTraceBuilder:
        self._selected_action = action
        if reason:
            self._reason = reason
        return self

    def add_metric(self, key: str, value: float | int | str) -> DecisionTraceBuilder:
        self._metrics[key] = value
        return self

    def build(self) -> DecisionTrace:
        if self._selected_action is None:
            raise ValueError("Cannot build DecisionTrace without a selected action.")

        rationale = {
            "mechanism": self._mechanism_name,
            "selected_action": self._selected_action.value,
            "reason": self._reason,
            "candidate_actions": self._candidate_actions,
            "observation_summary": self._observation_summary,
            "belief_summary": self._belief_summary,
        }

        return DecisionTrace(
            agent_id=self._agent_id,
            step_id=self._step_id,
            timestamp_ns=time.time_ns(),
            algorithm_name=self._algorithm_name,
            decision_type=self._decision_type,
            metrics=self._metrics,
            rationale=rationale,
        )
