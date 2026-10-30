"""Unit tests for discrete probability distributions and Bayesian updating against scipy.stats (Module M10)."""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy import stats

from autogen.intelligence.probabilistic.bayes import BayesUpdater
from autogen.intelligence.probabilistic.distributions import (
    Bernoulli,
    Binomial,
    Categorical,
    DiscretizedGaussian,
    DiscreteDistribution,
    Poisson,
    kl_divergence,
    logsumexp,
)


def test_bernoulli_distribution_oracle() -> None:
    """Verifies Bernoulli distribution against scipy.stats."""
    for p in (0.1, 0.5, 0.85):
        b = Bernoulli(p)
        assert abs(b.prob(1) - stats.bernoulli.pmf(1, p)) < 1e-10
        assert abs(b.prob(0) - stats.bernoulli.pmf(0, p)) < 1e-10
        assert abs(sum(b.mapping.values()) - 1.0) < 1e-12
        # Entropy check
        expected_ent = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        assert abs(b.entropy() - expected_ent) < 1e-10


def test_binomial_distribution_oracle() -> None:
    """Verifies Binomial PMF against scipy.stats.binom.pmf across parameter configurations."""
    n, p = 10, 0.4
    b = Binomial(n, p)
    for k in range(n + 1):
        custom_pmf = b.prob(k)
        oracle_pmf = stats.binom.pmf(k, n, p)
        assert abs(custom_pmf - oracle_pmf) < 1e-10
    assert abs(sum(b.mapping.values()) - 1.0) < 1e-12


def test_poisson_distribution_oracle() -> None:
    """Verifies Poisson distribution PMF against scipy.stats.poisson.pmf."""
    mu = 4.5
    pois = Poisson(mu, max_k=25)
    for k in range(20):
        custom_p = pois.prob(k)
        oracle_p = stats.poisson.pmf(k, mu)
        assert abs(custom_p - oracle_p) < 1e-7


def test_discretized_gaussian_properties() -> None:
    """Verifies that DiscretizedGaussian has symmetric peak around mean and sums to 1.0."""
    mean, std = 5.0, 1.5
    gauss = DiscretizedGaussian(mean, std, min_x=0, max_x=10)
    assert abs(sum(gauss.mapping.values()) - 1.0) < 1e-12
    # Mode should be at integer closest to mean
    assert max(gauss.support(), key=lambda x: gauss.prob(x)) == 5
    # Symmetric values around integer 5
    assert abs(gauss.prob(4) - gauss.prob(6)) < 1e-10
    assert abs(gauss.prob(3) - gauss.prob(7)) < 1e-10


def test_logsumexp_stability() -> None:
    """Tests logsumexp on very large numbers preventing overflow/underflow."""
    logs = [1000.0, 1000.0]
    # log(exp(1000) + exp(1000)) = 1000 + log(2)
    res = logsumexp(logs)
    assert abs(res - (1000.0 + math.log(2))) < 1e-10

    # Very small numbers
    logs_small = [-2000.0, -2000.0]
    res_small = logsumexp(logs_small)
    assert abs(res_small - (-2000.0 + math.log(2))) < 1e-10


def test_kl_divergence_self_zero() -> None:
    """Invariant: D_KL(P || P) == 0.0 and D_KL(P || Q) >= 0.0."""
    p = Categorical(["A", "B", "C"], [0.2, 0.5, 0.3])
    q = Categorical(["A", "B", "C"], [0.33, 0.33, 0.34])
    assert abs(kl_divergence(p, p)) < 1e-12
    assert kl_divergence(p, q) > 0.0


def test_bayes_updater_binary_and_log_odds() -> None:
    """Verifies Bayes theorem updating in linear probability and log-odds space."""
    # Medical test: P(Disease) = 0.01, P(+ | Disease) = 0.99, P(+ | No Disease) = 0.05
    prior = 0.01
    lik_h = 0.99
    lik_not_h = 0.05

    post = BayesUpdater.update_binary(prior, lik_h, lik_not_h)
    expected_post = (0.99 * 0.01) / (0.99 * 0.01 + 0.05 * 0.99)
    assert abs(post - expected_post) < 1e-10

    # Log-odds update must match exactly
    prior_lo = BayesUpdater.probability_to_log_odds(prior)
    llr = math.log(lik_h / lik_not_h)
    post_lo = BayesUpdater.update_log_odds(prior_lo, llr)
    post_prob_from_lo = BayesUpdater.log_odds_to_probability(post_lo)
    assert abs(post_prob_from_lo - expected_post) < 1e-10
