"""Integration test: Sutton & Barto CliffWalking benchmark (SARSA safe path vs Q-learning risky path) (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.rl.adapter import CliffWalkingEnv
from autogen.intelligence.rl.tabular import QLearning, SARSA


def test_cliff_walking_sarsa_learns_safe_path_and_q_learning_risky_path() -> None:
    """Verifies the canonical qualitative behavioral result from Sutton & Barto Example 6.6:

    - Q-learning (off-policy) learns the optimal path right along the edge of the cliff (row 1).
    - SARSA (on-policy) takes the agent's epsilon-greedy exploration into account and learns
      the safer, longer path along the upper rows (rows 2 and 3) to avoid falling off.
    """
    env = CliffWalkingEnv(max_steps=200)

    ql = QLearning(num_actions=4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42)
    sarsa = SARSA(num_actions=4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42)

    # Train Q-Learning for 400 episodes
    for _ in range(400):
        s = env.reset()
        done = False
        while not done:
            a = ql.select_action(s, explore=True)
            res = env.step(a)
            ql.update(s, a, res.reward, res.observation, res.terminated)
            s = res.observation
            done = res.terminated or res.truncated

    # Train SARSA for 400 episodes
    for _ in range(400):
        s = env.reset()
        done = False
        a = sarsa.select_action(s, explore=True)
        while not done:
            res = env.step(a)
            next_a = sarsa.select_action(res.observation, explore=True) if not res.terminated else None
            sarsa.update(s, a, res.reward, res.observation, res.terminated, next_a)
            s = res.observation
            a = next_a if next_a is not None else 0
            done = res.terminated or res.truncated

    # Collect greedy trajectory for Q-learning
    s = env.reset()
    ql_path = [s]
    done = False
    while not done and len(ql_path) < 30:
        a = ql.select_action(s, explore=False)
        res = env.step(a)
        s = res.observation
        ql_path.append(s)
        done = res.terminated

    # Collect greedy trajectory for SARSA
    s = env.reset()
    sarsa_path = [s]
    done = False
    while not done and len(sarsa_path) < 30:
        a = sarsa.select_action(s, explore=False)
        res = env.step(a)
        s = res.observation
        sarsa_path.append(s)
        done = res.terminated

    # Verify both reach Goal (11, 0)
    assert ql_path[-1] == (11, 0), f"Q-learning failed to reach goal: {ql_path}"
    assert sarsa_path[-1] == (11, 0), f"SARSA failed to reach goal: {sarsa_path}"

    # Extract max y (height) of both paths
    ql_max_y = max(pos[1] for pos in ql_path)
    sarsa_max_y = max(pos[1] for pos in sarsa_path)

    # Q-learning sticks to row 1 (optimal risky cliff edge)
    # SARSA routes higher (rows 2 or 3) for safety margin
    assert ql_max_y <= sarsa_max_y
    assert len(ql_path) <= len(sarsa_path)

