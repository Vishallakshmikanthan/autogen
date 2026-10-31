"""Unit tests for Multi-Armed Bandit algorithms (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.rl.bandits import (
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCBBandit,
)


def test_epsilon_greedy_decay_and_convergence() -> None:
    """Verifies that EpsilonGreedyBandit decays epsilon and converges to best arm."""
    rng = np.random.default_rng(42)
    arm_means = [0.1, 0.2, 0.9]  # Arm 2 is best
    bandit = EpsilonGreedyBandit(n_arms=3, eps_0=1.0, eps_min=0.01, decay_lambda=0.99, seed=42)

    assert bandit.current_epsilon == 1.0

    for _ in range(1000):
        arm = bandit.select_arm()
        reward = 1.0 if rng.random() < arm_means[arm] else 0.0
        bandit.update(arm, reward)

    assert bandit.current_epsilon <= 0.02
    assert int(np.argmax(bandit.q_estimates)) == 2


def test_ucb_bandit_finds_best_arm() -> None:
    """Verifies that UCBBandit identifies the optimal arm with sublinear exploration."""
    rng = np.random.default_rng(123)
    arm_means = [0.2, 0.8, 0.4]  # Arm 1 is best
    bandit = UCBBandit(n_arms=3, c=1.414, seed=42)

    for _ in range(800):
        arm = bandit.select_arm()
        reward = 1.0 if rng.random() < arm_means[arm] else 0.0
        bandit.update(arm, reward)

    # Arm 1 should be pulled the most
    assert int(np.argmax(bandit.counts)) == 1
    assert bandit.counts[1] > 500


def test_thompson_sampling_finds_best_arm() -> None:
    """Verifies that ThompsonSamplingBandit converges to optimal arm."""
    rng = np.random.default_rng(999)
    arm_means = [0.15, 0.85, 0.35]
    bandit = ThompsonSamplingBandit(n_arms=3, seed=42)

    for _ in range(800):
        arm = bandit.select_arm()
        reward = 1.0 if rng.random() < arm_means[arm] else 0.0
        bandit.update(arm, reward)

    assert int(np.argmax(bandit.counts)) == 1
    assert bandit.counts[1] > 500

