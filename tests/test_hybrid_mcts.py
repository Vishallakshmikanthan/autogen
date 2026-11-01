"""Tests for Monte Carlo Tree Search (MCTS) with canonical UCT and bandit convergence (SYS-15, Module M17)."""

from __future__ import annotations

from typing import Any

import numpy as np

from autogen.intelligence.hybrid.contracts import MCTSNode
from autogen.intelligence.hybrid.mcts import MonteCarloTreeSearch
from autogen.simulation.state.types import ActionType


class MultiArmedBanditModel:
    """Synthetic environment model testing MCTS multi-armed bandit convergence."""

    def __init__(self, arm_probs: dict[ActionType, float], seed: int = 42) -> None:
        self.arm_probs = arm_probs
        self.rng = np.random.default_rng(seed)

    def state_key(self, state: Any) -> str:
        return str(state)

    def get_valid_actions(self, state: Any) -> list[ActionType]:
        if state == "terminal":
            return []
        return list(self.arm_probs.keys())

    def step(self, state: Any, action: ActionType) -> tuple[Any, float, bool]:
        p = self.arm_probs[action]
        reward = 1.0 if self.rng.random() < p else 0.0
        return "terminal", reward, True


class FiniteDeterministicGameModel:
    """Finite 2-ply game tree with known minimax optimal action."""

    def state_key(self, state: Any) -> str:
        return str(state)

    def get_valid_actions(self, state: Any) -> list[ActionType]:
        if state == "root":
            return [ActionType.LEFT, ActionType.RIGHT]
        return []

    def step(self, state: Any, action: ActionType) -> tuple[Any, float, bool]:
        # LEFT yields reward 10.0; RIGHT yields reward 2.0
        if state == "root" and action == ActionType.LEFT:
            return "done_left", 10.0, True
        elif state == "root" and action == ActionType.RIGHT:
            return "done_right", 2.0, True
        return "done", 0.0, True


def test_mcts_bandit_convergence_oracle() -> None:
    """Oracle: MCTS must converge to the arm with highest expected payout (arm UP: p=0.9)."""
    arm_probs = {
        ActionType.UP: 0.9,     # Optimal arm
        ActionType.DOWN: 0.2,
        ActionType.LEFT: 0.2,
        ActionType.RIGHT: 0.2,
    }
    model = MultiArmedBanditModel(arm_probs=arm_probs, seed=42)
    mcts = MonteCarloTreeSearch(
        model=model,
        exploration_constant=1.414,
        rollout_depth=1,
        seed=42,
    )

    best_action, diagnostics = mcts.search(root_state="root", iterations=100)

    assert best_action == ActionType.UP
    assert diagnostics["child_visits"][ActionType.UP.value] > 50
    assert diagnostics["best_action_q"] > 0.7


def test_mcts_finite_game_minimax_oracle() -> None:
    """Oracle: MCTS must select the minimax optimal move (LEFT yielding 10.0 vs RIGHT yielding 2.0)."""
    model = FiniteDeterministicGameModel()
    mcts = MonteCarloTreeSearch(model=model, exploration_constant=1.0, seed=42)

    best_action, diagnostics = mcts.search(root_state="root", iterations=30)
    assert best_action == ActionType.LEFT
    assert diagnostics["best_action_q"] == 10.0


def test_mcts_uct_calculation() -> None:
    """Verifies UCT exploitation and exploration components."""
    root = MCTSNode(state_key="root", visits=10)
    child1 = MCTSNode(state_key="c1", parent=root, visits=5, total_value=10.0)  # Q = 2.0
    child2 = MCTSNode(state_key="c2", parent=root, visits=2, total_value=2.0)   # Q = 1.0

    root.children = {ActionType.UP: child1, ActionType.DOWN: child2}

    mcts = MonteCarloTreeSearch(model=FiniteDeterministicGameModel(), exploration_constant=1.414)
    best_act, best_child = mcts.select_uct_child(root)

    # Exploitation: c1 has 2.0; c2 has 1.0. With c=1.414, c1 UCT = 2.0 + 1.414*sqrt(ln(10)/5) = 2.0 + 0.959 = 2.959
    # c2 UCT = 1.0 + 1.414*sqrt(ln(10)/2) = 1.0 + 1.517 = 2.517 -> c1 wins
    assert best_act == ActionType.UP
    assert best_child == child1

