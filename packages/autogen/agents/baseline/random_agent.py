"""Seeded baseline RandomAgent."""

from __future__ import annotations

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class RandomAgent(BaseAgent):
    """Selects actions uniformly at random from an independent pseudo-random stream."""

    def __init__(
        self,
        agent_id: str,
        seed: int = 42,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities())
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._action_count = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        step = self._last_observation.step if self._last_observation else 0
        all_actions = [
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
            ActionType.STAY,
            ActionType.COLLECT,
        ]

        uniform_prob = 1.0 / len(all_actions)
        chosen_idx = int(self.rng.integers(0, len(all_actions)))
        selected = all_actions[chosen_idx]
        self._action_count += 1

        builder = (
            DecisionTraceBuilder(self.agent_id, step, "random_action_selection")
            .set_mechanism("uniform_pseudo_random")
            .set_observation_summary(self._last_observation)
            .set_selected_action(selected, f"Uniformly sampled action {selected.value}")
            .add_metric("action_count", self._action_count)
            .add_metric("seed", self.seed)
        )

        for act in all_actions:
            builder.add_candidate_action(act, uniform_prob)

        trace = builder.build()
        action = Action(agent_id=self.agent_id, action_type=selected)
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self._action_count = 0
