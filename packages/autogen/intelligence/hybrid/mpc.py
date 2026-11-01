"""Model Predictive Control (MPC) receding horizon planner (SYS-15, Module M17)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from autogen.intelligence.hybrid.mcts import MCTSEnvironmentModel
from autogen.simulation.state.types import ActionType


class ModelPredictiveController:
    """Receding horizon planner evaluating candidate action sequences over forward environment models."""

    def __init__(
        self,
        model: MCTSEnvironmentModel,
        horizon: int = 5,
        num_samples: int = 30,
        discount_factor: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.horizon = max(1, horizon)
        self.num_samples = max(1, num_samples)
        self.gamma = discount_factor
        self.rng = np.random.default_rng(seed)

    def evaluate_sequence(self, root_state: Any, action_seq: list[ActionType]) -> float:
        """Simulates action sequence rollout over forward model and computes discounted return."""
        sim_state = copy.deepcopy(root_state)
        total_return = 0.0
        curr_gamma = 1.0

        for action in action_seq:
            sim_state, reward, terminal = self.model.step(sim_state, action)
            total_return += curr_gamma * reward
            curr_gamma *= self.gamma
            if terminal:
                break
        return float(total_return)

    def plan(self, current_state: Any) -> tuple[ActionType, dict[str, Any]]:
        """Plans over horizon H using random shooting trajectory sampling, returning the first action."""
        valid_actions = self.model.get_valid_actions(current_state)
        if not valid_actions:
            return ActionType.STAY, {"best_return": 0.0, "horizon": self.horizon}

        best_return = float("-inf")
        best_sequence: list[ActionType] = [valid_actions[0]]

        # Ensure every immediate legal action is tested at least once
        for action in valid_actions:
            # Complete remaining sequence randomly
            seq = [action]
            for _ in range(self.horizon - 1):
                idx = int(self.rng.integers(0, len(valid_actions)))
                seq.append(valid_actions[idx])

            ret = self.evaluate_sequence(current_state, seq)
            if ret > best_return:
                best_return = ret
                best_sequence = seq

        # Random shooting for remaining sample budget
        remaining_samples = max(0, self.num_samples - len(valid_actions))
        for _ in range(remaining_samples):
            seq = []
            for _ in range(self.horizon):
                idx = int(self.rng.integers(0, len(valid_actions)))
                seq.append(valid_actions[idx])

            ret = self.evaluate_sequence(current_state, seq)
            if ret > best_return:
                best_return = ret
                best_sequence = seq

        first_action = best_sequence[0]
        diagnostics = {
            "best_return": best_return,
            "planned_horizon": len(best_sequence),
            "planned_sequence": [act.value for act in best_sequence],
            "first_action": first_action.value,
        }
        return first_action, diagnostics

