"""Unit tests for RLTrainingHarness and strict seed separation (SYS-12, Module M13)."""

from __future__ import annotations

import pytest

from autogen.intelligence.rl.adapter import CliffWalkingEnv
from autogen.intelligence.rl.harness import RLTrainingHarness
from autogen.intelligence.rl.tabular import QLearning


def test_harness_train_eval_seed_separation() -> None:
    """Verifies that RLTrainingHarness generates strictly disjoint seed sequences for training and evaluation."""
    harness = RLTrainingHarness(
        algorithm=QLearning(num_actions=4, seed=42),
        env_factory=lambda seed: CliffWalkingEnv(max_steps=50),
        train_seed=12345,
        eval_seed=67890,
    )

    # Spawn 20 train seeds and 20 eval seeds
    train_seeds = [int(s.generate_state(1)[0]) for s in harness.train_seq.spawn(20)]
    eval_seeds = [int(s.generate_state(1)[0]) for s in harness.eval_seq.spawn(20)]

    # Assert sets are strictly disjoint
    assert set(train_seeds).isdisjoint(set(eval_seeds))


def test_harness_training_loop_and_evaluation() -> None:
    """Verifies that the training harness runs episodes, triggers periodic evaluation, and tracks metrics."""
    ql = QLearning(num_actions=4, alpha=0.5, gamma=0.99, epsilon=0.1, seed=42)
    harness = RLTrainingHarness(
        algorithm=ql,
        env_factory=lambda seed: CliffWalkingEnv(max_steps=40),
        train_seed=100,
        eval_seed=200,
    )

    metrics = harness.train(n_episodes=20, eval_interval=10, n_eval_episodes=5)
    assert len(metrics) == 2  # 20 episodes / 10 = 2 evaluations
    for m in metrics:
        assert isinstance(m.mean_return, float)
        assert isinstance(m.mean_length, float)
        assert m.epsilon == 0.1

