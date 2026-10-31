"""Proximal Policy Optimization (PPO) Agent with glass-box DecisionTraces (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.neural.contracts import NetworkDiagnostics
from autogen.intelligence.neural.policy_gradient import PPO
from autogen.intelligence.rl.encoders import FeatureVectorEncoder
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class PPOAgent(BaseAgent):
    """Deep reinforcement learning agent driven by Proximal Policy Optimization (PPO)."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        ppo: PPO | None = None,
        state_dim: int = 4,
        update_interval: int = 32,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.state_dim = state_dim
        self.num_actions = len(self.ACTION_MAP)
        self.ppo = ppo or PPO(state_dim=state_dim, num_actions=self.num_actions, seed=42)
        self.encoder = FeatureVectorEncoder()
        self.update_interval = update_interval

        self.latest_trace: DecisionTrace | None = None
        self.latest_diagnostics: NetworkDiagnostics | None = None
        self._step_counter = 0

        # Rollout storage
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._old_probs: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._dones: list[bool] = []

        self._curr_state: np.ndarray | None = None
        self._curr_action_idx: int | None = None
        self._curr_prob_dist: np.ndarray | None = None
        self._curr_value: float = 0.0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "PPOPolicyGradient")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_state = self.encoder.encode(self._last_observation)
        action_idx, probs, value = self.ppo.select_action(current_state)
        chosen_action = self.ACTION_MAP[action_idx]

        for idx, act in enumerate(self.ACTION_MAP):
            trace_builder.add_candidate_action(act, float(probs[idx]))

        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))
        trace_builder.add_metric("state_value", float(value))
        trace_builder.add_metric("entropy", entropy)
        trace_builder.add_metric("max_prob", float(np.max(probs)))

        reason = f"PPO stochastic policy action {chosen_action.value} (p={probs[action_idx]:.3f}, V={value:.3f})"
        trace_builder.set_selected_action(chosen_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        self._curr_state = current_state
        self._curr_action_idx = action_idx
        self._curr_prob_dist = probs
        self._curr_value = value

        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        if self._curr_state is not None and self._curr_action_idx is not None and self._curr_prob_dist is not None:
            self._states.append(self._curr_state)
            self._actions.append(self._curr_action_idx)
            self._old_probs.append(self._curr_prob_dist)
            self._rewards.append(float(reward))
            self._values.append(self._curr_value)
            self._dones.append(terminal)

            if len(self._states) >= self.update_interval or terminal:
                self._update_policy()

        if terminal:
            self._curr_state = None
            self._curr_action_idx = None
            self._curr_prob_dist = None

    def _update_policy(self) -> None:
        T = len(self._states)
        if T == 0:
            return

        # Compute Generalized Advantage Estimation (GAE) / TD returns
        returns = np.zeros(T, dtype=np.float64)
        advantages = np.zeros(T, dtype=np.float64)

        running_g = 0.0
        for t in reversed(range(T)):
            if self._dones[t]:
                running_g = self._rewards[t]
            else:
                running_g = self._rewards[t] + self.ppo.gamma * running_g
            returns[t] = running_g
            advantages[t] = running_g - self._values[t]

        states_arr = np.array(self._states, dtype=np.float64)
        actions_arr = np.array(self._actions, dtype=int)
        old_probs_arr = np.array(self._old_probs, dtype=np.float64)

        self.latest_diagnostics = self.ppo.train_step(
            states=states_arr,
            actions=actions_arr,
            old_probs=old_probs_arr,
            returns=returns,
            advantages=advantages,
        )

        self._states.clear()
        self._actions.clear()
        self._old_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

    def reset(self) -> None:
        super().reset()
        self._states.clear()
        self._actions.clear()
        self._old_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()
        self._curr_state = None
        self._curr_action_idx = None
        self._curr_prob_dist = None
        self._step_counter = 0
