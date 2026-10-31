"""Multi-Armed Bandit algorithms implemented from scratch (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np


class EpsilonGreedyBandit:
    """Epsilon-Greedy multi-armed bandit with exponential epsilon decay."""

    def __init__(
        self,
        n_arms: int,
        eps_0: float = 1.0,
        eps_min: float = 0.01,
        decay_lambda: float = 0.995,
        seed: int = 42,
    ) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms = n_arms
        self.eps_0 = eps_0
        self.eps_min = eps_min
        self.decay_lambda = decay_lambda
        self.rng = np.random.default_rng(seed)

        self.q_estimates = np.zeros(n_arms, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.total_pulls = 0

    @property
    def current_epsilon(self) -> float:
        return max(self.eps_min, self.eps_0 * (self.decay_lambda ** self.total_pulls))

    def select_arm(self) -> int:
        if self.rng.random() < self.current_epsilon:
            return int(self.rng.integers(0, self.n_arms))
        max_q = float(np.max(self.q_estimates))
        best_arms = np.where(np.isclose(self.q_estimates, max_q, atol=1e-8))[0]
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total_pulls += 1
        # Incremental sample-average update: Q(a) <- Q(a) + (1/n) * (R - Q(a))
        n = self.counts[arm]
        self.q_estimates[arm] += (1.0 / n) * (reward - self.q_estimates[arm])


class UCBBandit:
    """Upper Confidence Bound (UCB1) multi-armed bandit algorithm."""

    def __init__(
        self,
        n_arms: int,
        c: float = 1.414,
        seed: int = 42,
    ) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms = n_arms
        self.c = c
        self.rng = np.random.default_rng(seed)

        self.q_estimates = np.zeros(n_arms, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.total_pulls = 0

    def select_arm(self) -> int:
        self.total_pulls += 1
        # Try each unvisited arm once first
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0])

        # UCB1 score: Q(a) + c * sqrt(ln(t) / N(a))
        confidence = self.c * np.sqrt(np.log(self.total_pulls) / self.counts)
        ucb_values = self.q_estimates + confidence
        max_ucb = float(np.max(ucb_values))
        best_arms = np.where(np.isclose(ucb_values, max_ucb, atol=1e-8))[0]
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.q_estimates[arm] += (1.0 / n) * (reward - self.q_estimates[arm])


class ThompsonSamplingBandit:
    """Thompson Sampling multi-armed bandit with Beta-Bernoulli conjugate priors."""

    def __init__(
        self,
        n_arms: int,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        seed: int = 42,
    ) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

        self.alphas = np.full(n_arms, alpha_0, dtype=np.float64)
        self.betas = np.full(n_arms, beta_0, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)

    def select_arm(self) -> int:
        # Sample theta_a ~ Beta(alpha_a, beta_a)
        samples = self.rng.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        # Binarize or clip continuous reward to [0, 1]
        r = float(np.clip(reward, 0.0, 1.0))
        self.alphas[arm] += r
        self.betas[arm] += (1.0 - r)

