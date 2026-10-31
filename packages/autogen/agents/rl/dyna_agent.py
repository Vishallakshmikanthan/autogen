"""Dyna-Q Agent integrating direct learning with simulated model planning (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.rl.dyna import DynaQ
from autogen.intelligence.rl.encoders import TabularGridEncoder
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class DynaAgent(BaseAgent):
    """Reinforcement learning agent executing model-based Dyna-Q planning."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        dyna_q: DynaQ | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True, planning=True))
        self.dyna_q = dyna_q or DynaQ(num_actions=4, n_planning_steps=10, seed=42)
        self.encoder = TabularGridEncoder()

        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0
        self._prev_state: Any = None
        self._prev_action_idx: int | None = None

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "ModelBasedDynaQ")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_state = self.encoder.encode(self._last_observation)
        q_vals = self.dyna_q.q_table.get_q(current_state)

        is_exploratory = self.dyna_q.rng.random() < self.dyna_q.epsilon
        if is_exploratory:
            chosen_idx = int(self.dyna_q.rng.integers(0, self.dyna_q.num_actions))
        else:
            chosen_idx = self.dyna_q.q_table.best_action(current_state, self.dyna_q.rng)

        chosen_action = self.ACTION_MAP[chosen_idx]

        for idx, act in enumerate(self.ACTION_MAP):
            trace_builder.add_candidate_action(act, float(q_vals[idx]))

        trace_builder.add_metric("is_exploratory", 1 if is_exploratory else 0)
        trace_builder.add_metric("n_planning_steps", self.dyna_q.n_planning_steps)
        trace_builder.add_metric("model_size", len(self.dyna_q.model))

        reason = (
            f"Dyna-Q action {chosen_action.value} (model_states={len(self.dyna_q.observed_states)}, Q={q_vals[chosen_idx]:.4f})"
        )
        trace_builder.set_selected_action(chosen_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        self._prev_state = current_state
        self._prev_action_idx = chosen_idx

        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        if self._prev_state is not None and self._prev_action_idx is not None and self._last_observation is not None:
            curr_state = self.encoder.encode(self._last_observation)
            self.dyna_q.update(
                state=self._prev_state,
                action=self._prev_action_idx,
                reward=reward,
                next_state=curr_state,
                terminated=terminal,
            )
        if terminal:
            self._prev_state = None
            self._prev_action_idx = None

    def reset(self) -> None:
        self.latest_trace = None
        self._step_counter = 0
        self._prev_state = None
        self._prev_action_idx = None

