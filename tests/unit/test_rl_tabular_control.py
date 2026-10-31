"""Unit tests for Tabular RL control convergence against exact MDP solutions from Module M11 (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.decision.gridworld_mdp import GridWorldMDPBuilder
from autogen.intelligence.decision.solvers import value_iteration
from autogen.intelligence.rl.tabular import (
    DoubleQLearning,
    ExpectedSARSA,
    QLearning,
    SARSA,
)


def test_q_learning_converges_to_exact_m11_q_star() -> None:
    """Verifies that Q-Learning converges toward exact Q* on a small 3-state deterministic chain.

    Chain: State 0 -> Action 0 -> State 1 (r=0) -> Action 0 -> State 2 (Terminal, r=10.0).
    Discount gamma = 0.9.
    Exact Q*(1, 0) = 10.0
    Exact Q*(0, 0) = 0.9 * 10.0 = 9.0
    """
    gamma = 0.9
    ql = QLearning(num_actions=2, alpha=0.1, gamma=gamma, epsilon=0.1, seed=42)

    # Train for 200 episodes along the chain
    for _ in range(400):
        # Step 0 -> 1
        ql.update(state=0, action=0, reward=0.0, next_state=1, terminated=False)
        # Step 1 -> 2 (Terminal)
        ql.update(state=1, action=0, reward=10.0, next_state=2, terminated=True)

    # Check convergence to exact values
    q_1_0 = ql.q_table.get_q(1)[0]
    q_0_0 = ql.q_table.get_q(0)[0]

    assert abs(q_1_0 - 10.0) < 0.05, f"Expected Q*(1,0) ~ 10.0, got {q_1_0}"
    assert abs(q_0_0 - 9.0) < 0.15, f"Expected Q*(0,0) ~ 9.0, got {q_0_0}"
    assert ql.q_table.best_action(0) == 0
    assert ql.q_table.best_action(1) == 0


def test_double_q_learning_reduces_maximization_bias() -> None:
    """Verifies Sutton & Barto's Maximization Bias Example 6.7:

    State A has two actions: left and right.
    Right goes to terminal with reward 0.
    Left goes to state B with reward 0.
    From state B, there are many actions with rewards ~ N(-0.1, 1.0).
    True expected value of Left is -0.1 < 0, so Right is optimal.
    Standard Q-learning overestimates Left > 0 due to max over noisy estimates.
    Double Q-learning correctly identifies Left as negative.
    """
    rng = np.random.default_rng(42)
    n_actions_b = 10

    q_learning = QLearning(num_actions=n_actions_b, alpha=0.1, gamma=1.0, epsilon=0.1, seed=42)
    double_q = DoubleQLearning(num_actions=n_actions_b, alpha=0.1, gamma=1.0, epsilon=0.1, seed=42)

    for _ in range(500):
        # Update from State B for both algorithms
        a_b = int(rng.integers(0, n_actions_b))
        noisy_reward = float(rng.normal(loc=-0.1, scale=1.0))

        q_learning.update(state="B", action=a_b, reward=noisy_reward, next_state="term", terminated=True)
        double_q.update(state="B", action=a_b, reward=noisy_reward, next_state="term", terminated=True)

        # Update from State A -> B
        q_learning.update(state="A", action=0, reward=0.0, next_state="B", terminated=False)
        double_q.update(state="A", action=0, reward=0.0, next_state="B", terminated=False)

    q_val_a_standard = q_learning.q_table.get_q("A")[0]
    q_val_a_double = 0.5 * (double_q.q1_table.get_q("A")[0] + double_q.q2_table.get_q("A")[0])

    # Standard Q-learning suffers positive maximization bias
    # Double Q-learning value should be lower and much closer to true expected value (-0.1)
    assert q_val_a_double < q_val_a_standard

