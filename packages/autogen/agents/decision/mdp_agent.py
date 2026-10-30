"""MDP Agent executing exact optimal policy pi* (SYS-10, Module M11)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.decision.contracts import MDPSolution, TabularMDP
from autogen.intelligence.decision.solvers import value_iteration
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position


class MDPAgent(BaseAgent):
    """Autonomous agent executing an exact optimal policy pi* computed via MDP dynamic programming."""

    ACTION_MAP: dict[str, ActionType] = {
        "UP": ActionType.UP,
        "DOWN": ActionType.DOWN,
        "LEFT": ActionType.LEFT,
        "RIGHT": ActionType.RIGHT,
    }

    def __init__(
        self,
        agent_id: str,
        solution: MDPSolution | None = None,
        mdp: TabularMDP | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(planning=True))
        if solution is not None:
            self.solution = solution
        elif mdp is not None:
            self.solution = value_iteration(mdp)
        else:
            raise ValueError("MDPAgent requires either an MDPSolution or TabularMDP.")

        # Build map from state label "(x,y)" to state index
        self.state_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(self.solution.state_labels)
        }
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "ExactMDPPolicy")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        pos = self._last_observation.position
        state_label = f"({pos.x},{pos.y})"

        if state_label in self.state_to_idx:
            s_idx = self.state_to_idx[state_label]
            v_val = float(self.solution.v_star[s_idx])
            best_a_idx = int(self.solution.policy[s_idx])
            act_name = self.solution.action_labels[best_a_idx]
            chosen_action = self.ACTION_MAP.get(act_name, ActionType.STAY)

            trace_builder.add_metric("v_star", v_val)
            trace_builder.add_metric("optimal_action_idx", best_a_idx)

            # Record candidate Q-values
            for a_idx, name in enumerate(self.solution.action_labels):
                q_val = float(self.solution.q_star[s_idx, a_idx])
                action_type = self.ACTION_MAP.get(name, ActionType.STAY)
                trace_builder.add_candidate_action(action_type, q_val)

            reason = f"Optimal policy pi*({state_label}) = {act_name} with V*={v_val:.4f}"
            trace_builder.set_selected_action(chosen_action, reason)
        else:
            chosen_action = ActionType.STAY
            trace_builder.set_selected_action(ActionType.STAY, f"State {state_label} not found in MDP state space")

        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        self._last_observation = None
        self.latest_trace = None
        self._step_counter = 0

