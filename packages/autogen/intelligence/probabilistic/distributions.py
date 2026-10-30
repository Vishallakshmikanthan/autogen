"""Discrete probability distributions with strict normalization guarantees (SYS-09, Module M10)."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Generic, Mapping, TypeVar

import numpy as np

from autogen.intelligence.probabilistic.contracts import Distribution

T = TypeVar("T")


def logsumexp(logs: Sequence[float] | np.ndarray) -> float:
    """Computes log(sum(exp(logs))) in a numerically stable manner to prevent underflow."""
    arr = np.asarray(logs, dtype=np.float64)
    if len(arr) == 0:
        return -math.inf
    max_val = np.max(arr)
    if math.isinf(max_val) and max_val < 0:
        return -math.inf
    return float(max_val + np.log(np.sum(np.exp(arr - max_val))))


class DiscreteDistribution(Distribution[T], Generic[T]):
    """A finite discrete probability distribution with strict normalization guarantees."""

    def __init__(self, probabilities: Mapping[T, float]) -> None:
        if not probabilities:
            raise ValueError("DiscreteDistribution must contain at least one outcome.")

        # Filter out negative probabilities and normalize
        total = sum(probabilities.values())
        if total <= 0.0 or math.isnan(total):
            raise ValueError(f"Sum of probabilities must be strictly positive, got {total}.")

        # Store normalized probabilities
        self._probs: dict[T, float] = {k: float(v) / total for k, v in probabilities.items() if v > 0.0}
        if not self._probs:
            raise ValueError("No positive probability outcomes remaining after normalization.")

        # Re-normalize to guarantee exact sum = 1.0
        exact_sum = sum(self._probs.values())
        self._probs = {k: v / exact_sum for k, v in self._probs.items()}

        self._outcomes: tuple[T, ...] = tuple(self._probs.keys())
        self._values: np.ndarray = np.array([self._probs[o] for o in self._outcomes], dtype=np.float64)
        self._cdf: np.ndarray = np.cumsum(self._values)
        self._cdf[-1] = 1.0  # Guarantee boundary

    def prob(self, outcome: T) -> float:
        return self._probs.get(outcome, 0.0)

    def log_prob(self, outcome: T) -> float:
        p = self.prob(outcome)
        return math.log(p) if p > 0.0 else -math.inf

    def sample(self, rng: np.random.Generator) -> T:
        u = rng.random()
        idx = int(np.searchsorted(self._cdf, u))
        idx = min(idx, len(self._outcomes) - 1)
        return self._outcomes[idx]

    def support(self) -> tuple[T, ...]:
        return self._outcomes

    def entropy(self) -> float:
        """Computes Shannon entropy H(X) = -sum p(x) log2(p(x)) in bits."""
        ent = 0.0
        for p in self._probs.values():
            if p > 0.0:
                ent -= p * math.log2(p)
        return ent

    @property
    def mapping(self) -> dict[T, float]:
        return dict(self._probs)

    def __getitem__(self, outcome: T) -> float:
        return self.prob(outcome)

    def __iter__(self) -> Iterator[T]:
        return iter(self._outcomes)

    def __len__(self) -> int:
        return len(self._outcomes)

    def __str__(self) -> str:
        items = ", ".join(f"{k}: {v:.4f}" for k, v in self._probs.items())
        return f"DiscreteDistribution({items})"


def kl_divergence(p: DiscreteDistribution[T], q: DiscreteDistribution[T]) -> float:
    """Computes Kullback-Leibler divergence D_KL(P || Q) = sum p(x) log2(p(x) / q(x))."""
    div = 0.0
    for outcome, p_val in p.mapping.items():
        if p_val > 0.0:
            q_val = q.prob(outcome)
            if q_val <= 0.0:
                return math.inf
            div += p_val * math.log2(p_val / q_val)
    return div


class Bernoulli(DiscreteDistribution[int]):
    """Bernoulli distribution: P(X=1) = p, P(X=0) = 1-p."""

    def __init__(self, p: float) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Parameter p must be in [0, 1], got {p}")
        super().__init__({1: p, 0: 1.0 - p})
        self.p = p


class Binomial(DiscreteDistribution[int]):
    """Binomial distribution: P(X=k) = (n choose k) * p^k * (1-p)^(n-k)."""

    def __init__(self, n: int, p: float) -> None:
        if n < 0:
            raise ValueError(f"Parameter n must be non-negative, got {n}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Parameter p must be in [0, 1], got {p}")

        self.n = n
        self.p = p

        probs: dict[int, float] = {}
        for k in range(n + 1):
            comb = math.comb(n, k)
            prob_k = comb * (p**k) * ((1.0 - p) ** (n - k))
            probs[k] = prob_k

        super().__init__(probs)


class Categorical(DiscreteDistribution[T], Generic[T]):
    """Categorical distribution over an arbitrary set of categorical outcomes."""

    def __init__(self, outcomes: Sequence[T], weights: Sequence[float]) -> None:
        if len(outcomes) != len(weights):
            raise ValueError("Outcomes and weights must have identical length.")
        prob_dict = {outcomes[i]: weights[i] for i in range(len(outcomes))}
        super().__init__(prob_dict)


class DiscretizedGaussian(DiscreteDistribution[int]):
    """Discretized Gaussian distribution evaluated over integer range [min_x, max_x]."""

    def __init__(self, mean: float, std: float, min_x: int, max_x: int) -> None:
        if std <= 0.0:
            raise ValueError(f"Standard deviation must be strictly positive, got {std}")
        if min_x > max_x:
            raise ValueError("min_x cannot be greater than max_x")

        probs: dict[int, float] = {}
        var = std**2
        norm_factor = 1.0 / (math.sqrt(2.0 * math.pi) * std)

        for x in range(min_x, max_x + 1):
            pdf_val = norm_factor * math.exp(-0.5 * ((x - mean) ** 2) / var)
            probs[x] = pdf_val

        super().__init__(probs)
        self.mean = mean
        self.std = std


class Poisson(DiscreteDistribution[int]):
    """Poisson distribution: P(X=k) = (mu^k * e^(-mu)) / k! evaluated up to max_k."""

    def __init__(self, mu: float, max_k: int = 30) -> None:
        if mu <= 0.0:
            raise ValueError(f"Rate parameter mu must be strictly positive, got {mu}")
        if max_k < 0:
            raise ValueError("max_k must be non-negative")

        self.mu = mu
        self.max_k = max_k

        probs: dict[int, float] = {}
        exp_neg_mu = math.exp(-mu)
        cur_p = exp_neg_mu
        probs[0] = cur_p

        for k in range(1, max_k + 1):
            cur_p = (cur_p * mu) / k
            probs[k] = cur_p

        super().__init__(probs)
