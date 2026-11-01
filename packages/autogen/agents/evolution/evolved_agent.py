"""Evolved Agent driven by evolved neural or symbolic controllers with glass-box traces (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.evolution.gp import GPNode
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.rl.encoders import FeatureVectorEncoder
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class EvolvedAgent(BaseAgent):
    """Autonomous agent parameterized by an evolved controller (NeuralNetwork or GPNode)."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT, ActionType.STAY)

    def __init__(
        self,
        agent_id: str,
        controller: NeuralNetwork | GPNode | None = None,
        state_dim: int = 4,
        generation: int = 0,
        genome_id: str = "",
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(learning=True))
        self.controller = controller
        self.state_dim = state_dim
        self.generation = generation
        self.genome_id = genome_id
        self.encoder = FeatureVectorEncoder()

        self._last_state: np.ndarray | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._last_state = self.encoder.encode(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        self._step_counter += 1
        builder = DecisionTraceBuilder(
            agent_id=self.agent_id,
            step_id=self._step_counter,
            algorithm_name="EvolvedController",
        )

        obs = self._last_observation
        builder.set_observation_summary(obs)
        builder.set_belief_summary({
            "generation": self.generation,
            "genome_id": self.genome_id,
        })

        state = self._last_state
        if state is None:
            state = np.zeros(self.state_dim, dtype=np.float64)

        num_actions = len(self.ACTION_MAP)
        action_scores = np.zeros(num_actions, dtype=np.float64)

        if isinstance(self.controller, NeuralNetwork):
            # Forward pass through evolved neural network
            logits = self.controller.forward(state.reshape(1, -1), training=False).ravel()
            if len(logits) >= num_actions:
                action_scores = logits[:num_actions]
            else:
                action_scores[: len(logits)] = logits
        elif isinstance(self.controller, GPNode):
            # Evaluate GP tree on state to get scalar direction/action index
            val = float(self.controller.evaluate(state)[0])
            action_idx = int(abs(round(val))) % num_actions
            action_scores[action_idx] = 1.0
        else:
            # Default heuristic/random fallback
            action_scores[0] = 1.0

        best_action_idx = int(np.argmax(action_scores))
        chosen_action_type = self.ACTION_MAP[best_action_idx]

        for i, act_type in enumerate(self.ACTION_MAP):
            builder.add_candidate_action(act_type, float(action_scores[i]))

        builder.set_selected_action(
            chosen_action_type,
            reason=f"Evolved policy selected {chosen_action_type.value} with score {action_scores[best_action_idx]:.3f}",
        )
        builder.add_metric("generation", self.generation)
        builder.add_metric("max_score", float(action_scores[best_action_idx]))

        action = Action(agent_id=self.agent_id, action_type=chosen_action_type)
        trace = builder.build()
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        """Evolved agents undergo generational selection rather than online backprop."""
        pass

    def reset(self) -> None:
        super().reset()
        self._last_state = None
        self._step_counter = 0
