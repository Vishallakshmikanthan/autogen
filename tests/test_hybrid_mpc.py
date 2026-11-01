"""Tests for Model Predictive Control (MPC) receding horizon planning (SYS-15, Module M17)."""

from __future__ import annotations

from typing import Any

from autogen.intelligence.hybrid.mpc import ModelPredictiveController
from autogen.simulation.state.types import ActionType


class LineWorld1DModel:
    """1D navigation model towards target coordinate x = 5."""

    def state_key(self, state: Any) -> str:
        return str(state)

    def get_valid_actions(self, state: Any) -> list[ActionType]:
        return [ActionType.LEFT, ActionType.RIGHT, ActionType.STAY]

    def step(self, state: Any, action: ActionType) -> tuple[Any, float, bool]:
        pos = int(state)
        if action == ActionType.RIGHT:
            pos += 1
        elif action == ActionType.LEFT:
            pos -= 1

        # Distance penalty from goal 5
        reward = -abs(pos - 5)
        terminal = (pos == 5)
        return pos, reward, terminal


def test_mpc_receding_horizon_planning() -> None:
    """Verifies that MPC plans trajectory sequences and selects the action steering towards the target."""
    model = LineWorld1DModel()
    mpc = ModelPredictiveController(model=model, horizon=4, num_samples=30, seed=42)

    # From x = 2, moving RIGHT reduces distance to 5
    action, diag = mpc.plan(current_state=2)
    assert action == ActionType.RIGHT
    assert diag["planned_horizon"] == 4
    assert diag["first_action"] == ActionType.RIGHT.value

    # From x = 7, moving LEFT reduces distance to 5
    action_left, diag_left = mpc.plan(current_state=7)
    assert action_left == ActionType.LEFT
    assert diag_left["first_action"] == ActionType.LEFT.value

