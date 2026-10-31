"""SARSA Agent executing on-policy tabular control (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.rl.encoders import TabularGridEncoder
from autogen.intelligence.rl.tabular import SARSA
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class SarsaAgent(BaseAgent):
    """Reinforcement learning agent executing on-policy SARSA control."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        sarsa: SARSA | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.sarsa = sarsa or SARSA(num_actions=4, seed=42)
        self.encoder = TabularGridEncoder()

        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0
        self._prev_state: Any = None
        self._prev_action_idx: int | None = None
        self._next_action_idx: int | None = None

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "TabularSARSA")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_state = self.encoder.encode(self._last_observation)
        q_vals = self.sarsa.q_table.get_q(current_state)

        if self._next_action_idx is not None:
            chosen_idx = self._next_action_idx
            self._next_action_idx = None
        else:
            chosen_idx = self.sarsa.select_action(current_state, explore=True)

        chosen_action = self.ACTION_MAP[chosen_idx]

        for idx, act in enumerate(self.ACTION_MAP):
            trace_builder.add_candidate_action(act, float(q_vals[idx]))

        trace_builder.add_metric("epsilon", float(self.sarsa.epsilon))
        trace_builder.add_metric("chosen_q", float(q_vals[chosen_idx]))

        reason = f"SARSA action {chosen_action.value} (Q={q_vals[chosen_idx]:.4f})"
        trace_builder.set_selected_action(chosen_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        self._prev_state = current_state
        self._prev_action_idx = chosen_idx

        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        if self._prev_state is not None and self._prev_action_idx is not None and self._last_observation is not None:
            curr_state = self.encoder.encode(self._last_observation)
            next_act_idx = self.sarsa.select_action(curr_state, explore=True) if not terminal else None
            self._next_action_idx = next_act_idx

            self.sarsa.update(
                state=self._prev_state,
                action=self._prev_action_idx,
                reward=reward,
                next_state=curr_state,
                terminated=terminal,
                next_action=next_act_idx,
            )
        if terminal:
            self._prev_state = None
            self._prev_action_idx = None
            self._next_action_idx = None

    def reset(self) -> None:
        self.latest_trace = None
        self._step_counter = 0
        self._prev_state = None
        self._prev_action_idx = None
        self._next_action_idx = None

