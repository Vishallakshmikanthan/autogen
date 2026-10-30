"""Dummy, scripted, and adversarial agents for test harness verification."""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class ScriptedAgent(BaseAgent):
    """Executes a pre-defined sequence of actions deterministically."""

    def __init__(
        self,
        agent_id: str,
        script: list[ActionType],
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities)
        self.script = list(script)
        self.cursor = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if not self.script:
            act_type = ActionType.STAY
        else:
            act_type = self.script[self.cursor % len(self.script)]
            self.cursor += 1

        action = Action(agent_id=self.agent_id, action_type=act_type)
        trace = DecisionTrace(
            agent_id=self.agent_id,
            step_id=self._last_observation.step if self._last_observation else 0,
            algorithm_name="scripted",
            decision_type="action_selection",
            metrics={"cursor": self.cursor},
            rationale={"script_action": act_type.value},
        )
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.cursor = 0


class RandomWalkAgent(BaseAgent):
    """Picks random actions from an independent seeded NumPy generator."""

    def __init__(
        self,
        agent_id: str,
        seed: int = 1234,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities)
        self.rng = np.random.default_rng(seed)
        self._step_count = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        actions = [
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
            ActionType.STAY,
            ActionType.COLLECT,
        ]
        chosen = actions[int(self.rng.integers(0, len(actions)))]
        self._step_count += 1
        action = Action(agent_id=self.agent_id, action_type=chosen)
        trace = DecisionTrace(
            agent_id=self.agent_id,
            step_id=self._last_observation.step if self._last_observation else 0,
            algorithm_name="random_walk",
            decision_type="random_choice",
            metrics={"step_count": self._step_count},
            rationale={"chosen_random": chosen.value},
        )
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self._step_count = 0


class HostileAgent(BaseAgent):
    """Malicious agent that actively attempts to mutate observations and bypass environment boundaries."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.mutation_attempt_failed = False

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        # Attempt 1: Mutate observation attributes (must be blocked by frozen=True)
        attr_to_attack = "energy"
        try:
            setattr(observation, attr_to_attack, 999999.0)
        except (ValidationError, TypeError):
            self.mutation_attempt_failed = True

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        action = Action(agent_id=self.agent_id, action_type=ActionType.STAY)
        trace = DecisionTrace(
            agent_id=self.agent_id,
            step_id=self._last_observation.step if self._last_observation else 0,
            algorithm_name="hostile_mutator",
            decision_type="attack",
            metrics={"mutation_blocked": 1 if self.mutation_attempt_failed else 0},
        )
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
