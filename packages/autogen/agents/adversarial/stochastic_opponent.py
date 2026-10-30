"""Stochastic opponent agent executing mixed action probabilities (SYS-10, Module M07)."""

from __future__ import annotations

import random
from typing import Sequence

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position, TERRAIN_PROPERTIES


class StochasticOpponent(BaseAgent):
    """Opponent agent executing actions drawn from an explicit stochastic distribution."""

    def __init__(
        self,
        agent_id: str,
        seed: int = 42,
        capabilities: AgentCapabilities | None = None,
        action_weights: dict[ActionType, float] | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities())
        self.rng = random.Random(seed)
        self.action_weights = action_weights or {
            ActionType.UP: 1.0,
            ActionType.DOWN: 1.0,
            ActionType.LEFT: 1.0,
            ActionType.RIGHT: 1.0,
            ActionType.STAY: 0.5,
        }
        self.blocked_cells: set[Position] = set()
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

        for cell in observation.visible_cells:
            if not TERRAIN_PROPERTIES[cell.terrain].walkable:
                self.blocked_cells.add(cell.position)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            trace = (
                DecisionTraceBuilder(self.agent_id, self._step_counter, "StochasticPolicy")
                .set_selected_action(ActionType.STAY, "No observation available")
                .build()
            )
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        my_pos = self._last_observation.position
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "StochasticPolicy")
        trace_builder.set_observation_summary(self._last_observation)

        legal_acts: list[ActionType] = [ActionType.STAY]
        deltas = [
            (ActionType.UP, 0, -1),
            (ActionType.DOWN, 0, 1),
            (ActionType.LEFT, -1, 0),
            (ActionType.RIGHT, 1, 0),
        ]
        for act, dx, dy in deltas:
            cand = Position(x=my_pos.x + dx, y=my_pos.y + dy)
            if cand not in self.blocked_cells:
                legal_acts.append(act)

        weights = [self.action_weights.get(act, 1.0) for act in legal_acts]
        total_w = sum(weights)
        norm_probs = [w / total_w for w in weights]

        for act, prob in zip(legal_acts, norm_probs):
            trace_builder.add_candidate_action(act, prob)

        chosen_act = self.rng.choices(legal_acts, weights=weights, k=1)[0]
        chosen_prob = norm_probs[legal_acts.index(chosen_act)]

        trace_builder.set_selected_action(
            chosen_act,
            f"Stochastic action sample (p={chosen_prob:.3f})",
        )
        trace_builder.add_metric("selection_probability", chosen_prob)

        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_act), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.blocked_cells.clear()
        self.latest_trace = None
