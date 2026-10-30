"""Bayes theorem updating in linear probability and log-odds space (SYS-09, Module M10)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeVar

from autogen.intelligence.probabilistic.distributions import DiscreteDistribution

T = TypeVar("T")


class BayesUpdater:
    """Exact Bayesian belief updater supporting scalar, log-odds, and full discrete distributions."""

    @staticmethod
    def update_binary(
        prior_h: float,
        likelihood_given_h: float,
        likelihood_given_not_h: float,
    ) -> float:
        """Applies Bayes rule to update binary hypothesis H given evidence E.

        P(H | E) = (P(E | H) * P(H)) / (P(E | H)*P(H) + P(E | ~H)*(1 - P(H)))
        """
        if not (0.0 <= prior_h <= 1.0):
            raise ValueError(f"Prior must be in [0, 1], got {prior_h}")
        if likelihood_given_h < 0.0 or likelihood_given_not_h < 0.0:
            raise ValueError("Likelihoods must be non-negative.")

        numerator = likelihood_given_h * prior_h
        denominator = numerator + likelihood_given_not_h * (1.0 - prior_h)

        if denominator == 0.0:
            return 0.0  # Evidence is impossible under both hypotheses

        return float(numerator / denominator)

    @staticmethod
    def probability_to_log_odds(p: float) -> float:
        """Converts probability p in (0, 1) to log-odds log(p / (1 - p))."""
        if p <= 0.0:
            return -math.inf
        if p >= 1.0:
            return math.inf
        return math.log(p / (1.0 - p))

    @staticmethod
    def log_odds_to_probability(l_odds: float) -> float:
        """Converts log-odds to probability p = 1 / (1 + exp(-l_odds))."""
        if math.isinf(l_odds):
            return 1.0 if l_odds > 0 else 0.0
        # Prevent overflow in exp
        if l_odds < -700.0:
            return 0.0
        if l_odds > 700.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-l_odds))

    @staticmethod
    def update_log_odds(
        prior_log_odds: float,
        log_likelihood_ratio: float,
    ) -> float:
        """Additive belief update in log-odds space:

        log O(H | E) = log O(H) + log(P(E | H) / P(E | ~H))
        """
        return prior_log_odds + log_likelihood_ratio

    @staticmethod
    def update_discrete(
        prior: DiscreteDistribution[T],
        likelihood_fn: Callable[[T], float],
    ) -> DiscreteDistribution[T]:
        """Updates a discrete hypothesis distribution given a likelihood function L(h) = P(E | H = h)."""
        updated_weights: dict[T, float] = {}
        for h in prior.support():
            p_prior = prior.prob(h)
            lik = likelihood_fn(h)
            if lik < 0.0:
                raise ValueError(f"Likelihood must be non-negative, got {lik} for hypothesis {h}")
            updated_weights[h] = p_prior * lik

        return DiscreteDistribution(updated_weights)
