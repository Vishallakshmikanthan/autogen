"""Decentralized Execution MARL Agent (CTDE) (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.neural.activations import ReLU, Softmax
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.rl.encoders import FeatureVectorEncoder
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class MARLAgent(BaseAgent):
    """Decentralized actor executing local policies trained under Centralized Training."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        actor_net: NeuralNetwork | None = None,
        obs_dim: int = 4,
        capabilities: AgentCapabilities | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        self.actor_net = actor_net or NeuralNetwork([
            DenseLayer(obs_dim, 32, ReLU(), init_method="he", seed=seed),
            DenseLayer(32, len(self.ACTION_MAP), Softmax(), init_method="xavier", seed=seed + 1),
        ])
        self.encoder = FeatureVectorEncoder()

        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "DecentralizedMARLActor")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_obs = self.encoder.encode(self._last_observation)
        probs = self.actor_net.predict(current_obs.reshape(1, -1))[0]
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)

        chosen_idx = int(self.rng.choice(len(self.ACTION_MAP), p=probs))
        chosen_action = self.ACTION_MAP[chosen_idx]

        for idx, act in enumerate(self.ACTION_MAP):
            trace_builder.add_candidate_action(act, float(probs[idx]))

        entropy = -float(np.sum(probs * np.log(probs)))
        trace_builder.add_metric("entropy", entropy)
        trace_builder.add_metric("max_prob", float(np.max(probs)))

        reason = f"CTDE decentralized actor choice {chosen_action.value} (p={probs[chosen_idx]:.3f})"
        trace_builder.set_selected_action(chosen_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        # Parameter updates handled via centralized CTDE controller during joint training
        pass

    def reset(self) -> None:
        super().reset()
        self._step_counter = 0
