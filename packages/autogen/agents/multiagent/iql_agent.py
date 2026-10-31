"""Independent Q-Learning Agent for Multi-Agent Systems (SYS-14, Module M15)."""

from __future__ import annotations

from typing import Any

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.rl.encoders import TabularGridEncoder
from autogen.intelligence.rl.tabular import QLearning
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class IQLAgent(BaseAgent):
    """Decentralized multi-agent reinforcement learning agent driven by Independent Q-Learning."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        q_learning: QLearning | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.q_learning = q_learning or QLearning(num_actions=4, seed=42)
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
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "IndependentQLearning")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_state = self.encoder.encode(self._last_observation)
        q_vals = self.q_learning.q_table.get_q(current_state)

        # Explore vs exploit
        is_exploratory = self.q_learning.rng.random() < self.q_learning.epsilon
        if is_exploratory:
            chosen_idx = int(self.q_learning.rng.integers(0, self.q_learning.num_actions))
        else:
            chosen_idx = self.q_learning.q_table.best_action(current_state, self.q_learning.rng)

        chosen_action = self.ACTION_MAP[chosen_idx]

        for idx, act in enumerate(self.ACTION_MAP):
            trace_builder.add_candidate_action(act, float(q_vals[idx]))

        trace_builder.add_metric("is_exploratory", 1 if is_exploratory else 0)
        trace_builder.add_metric("epsilon", float(self.q_learning.epsilon))
        trace_builder.add_metric("max_q", float(np.max(q_vals)))

        reason = (
            f"IQL random exploratory action {chosen_action.value}"
            if is_exploratory
            else f"IQL greedy action {chosen_action.value} (Q={q_vals[chosen_idx]:.4f})"
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
            self.q_learning.update(
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
        super().reset()
        self._prev_state = None
        self._prev_action_idx = None
        self._step_counter = 0
