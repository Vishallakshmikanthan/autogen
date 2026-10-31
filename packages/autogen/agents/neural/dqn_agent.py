"""Deep Q-Network Agent with glass-box DecisionTraces (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.neural.contracts import NetworkDiagnostics
from autogen.intelligence.neural.dqn import DeepQNetwork, ReplayBuffer
from autogen.intelligence.rl.encoders import FeatureVectorEncoder
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class DQNAgent(BaseAgent):
    """Deep reinforcement learning agent driven by Deep Q-Networks with authentic DecisionTraces."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        dqn: DeepQNetwork | None = None,
        state_dim: int = 4,
        replay_capacity: int = 5000,
        batch_size: int = 32,
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.state_dim = state_dim
        self.num_actions = len(self.ACTION_MAP)
        self.dqn = dqn or DeepQNetwork(
            state_dim=state_dim,
            num_actions=self.num_actions,
            seed=42,
        )
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=42)
        self.encoder = FeatureVectorEncoder()

        self.batch_size = batch_size
        self.epsilon = float(initial_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)

        self.latest_trace: DecisionTrace | None = None
        self.latest_diagnostics: NetworkDiagnostics | None = None
        self._step_counter = 0

        self._prev_state: np.ndarray | None = None
        self._prev_action_idx: int | None = None

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "DeepQNetwork")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_state = self.encoder.encode(self._last_observation)
        q_vals = self.dqn.get_q_values(current_state)

        # Epsilon-greedy action selection
        is_exploratory = self.dqn.rng.random() < self.epsilon
        if is_exploratory:
            chosen_idx = int(self.dqn.rng.integers(0, self.num_actions))
        else:
            chosen_idx = int(np.argmax(q_vals))

        chosen_action = self.ACTION_MAP[chosen_idx]

        for idx, act in enumerate(self.ACTION_MAP):
            trace_builder.add_candidate_action(act, float(q_vals[idx]))

        trace_builder.add_metric("is_exploratory", 1 if is_exploratory else 0)
        trace_builder.add_metric("epsilon", float(self.epsilon))
        trace_builder.add_metric("max_q", float(np.max(q_vals)))
        trace_builder.add_metric("mean_q", float(np.mean(q_vals)))
        if self.latest_diagnostics is not None:
            trace_builder.add_metric("loss", float(self.latest_diagnostics.loss))
            trace_builder.add_metric("grad_norm", float(self.latest_diagnostics.gradient_norm))

        reason = (
            f"DQN random exploration {chosen_action.value} (eps={self.epsilon:.3f})"
            if is_exploratory
            else f"DQN greedy action {chosen_action.value} (Q={q_vals[chosen_idx]:.4f})"
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
            # Store in experience replay buffer
            self.replay_buffer.add(
                state=self._prev_state,
                action=self._prev_action_idx,
                reward=reward,
                next_state=curr_state,
                terminated=terminal,
            )

            # Perform DQN optimization update if replay buffer has sufficient samples
            if len(self.replay_buffer) >= self.batch_size:
                self.latest_diagnostics = self.dqn.update(self.replay_buffer, self.batch_size)

            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if terminal:
            self._prev_state = None
            self._prev_action_idx = None

    def reset(self) -> None:
        super().reset()
        self._prev_state = None
        self._prev_action_idx = None
        self._step_counter = 0
