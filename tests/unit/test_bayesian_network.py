"""Unit tests for Bayesian Network inference algorithms (Enumeration, Variable Elimination, Sampling) (Module M10)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.probabilistic.bayesian_network import (
    BayesNet,
    BayesNetNode,
    enumeration_ask,
    gibbs_sampling,
    likelihood_weighting,
    rejection_sampling,
    variable_elimination,
)


def _build_alarm_network() -> BayesNet:
    """Constructs the canonical Burglary-Earthquake Alarm network (Pearl 1988).

    B: Burglary, E: Earthquake, A: Alarm, J: JohnCalls, M: MaryCalls
    All variables are binary: 0 = False, 1 = True.
    """
    bn = BayesNet()

    # B: P(B=0)=0.999, P(B=1)=0.001
    bn.add_node(BayesNetNode("B", (), np.array([0.999, 0.001])))
    # E: P(E=0)=0.998, P(E=1)=0.002
    bn.add_node(BayesNetNode("E", (), np.array([0.998, 0.002])))

    # A CPT conditioned on (B, E)
    # CPT shape: (2, 2, 2) -> cpt[B, E, A]
    cpt_a = np.zeros((2, 2, 2))
    # B=0, E=0 -> P(A=1)=0.001
    cpt_a[0, 0] = [0.999, 0.001]
    # B=0, E=1 -> P(A=1)=0.29
    cpt_a[0, 1] = [0.71, 0.29]
    # B=1, E=0 -> P(A=1)=0.94
    cpt_a[1, 0] = [0.06, 0.94]
    # B=1, E=1 -> P(A=1)=0.95
    cpt_a[1, 1] = [0.05, 0.95]
    bn.add_node(BayesNetNode("A", ("B", "E"), cpt_a))

    # J CPT conditioned on A: cpt[A, J]
    cpt_j = np.zeros((2, 2))
    cpt_j[0] = [0.95, 0.05]  # P(J=1|A=0)=0.05
    cpt_j[1] = [0.10, 0.90]  # P(J=1|A=1)=0.90
    bn.add_node(BayesNetNode("J", ("A",), cpt_j))

    # M CPT conditioned on A: cpt[A, M]
    cpt_m = np.zeros((2, 2))
    cpt_m[0] = [0.99, 0.01]  # P(M=1|A=0)=0.01
    cpt_m[1] = [0.30, 0.70]  # P(M=1|A=1)=0.70
    bn.add_node(BayesNetNode("M", ("A",), cpt_m))

    return bn


def test_variable_elimination_matches_enumeration_oracle() -> None:
    """Oracle test: Variable Elimination produces identical posteriors to exhaustive Enumeration."""
    bn = _build_alarm_network()
    # Query: P(B | J=True, M=True)
    evidence = {"J": 1, "M": 1}

    post_enum = enumeration_ask("B", evidence, bn)
    post_ve = variable_elimination("B", evidence, bn)

    # Both must sum to 1.0
    assert abs(sum(post_enum.mapping.values()) - 1.0) < 1e-10
    assert abs(sum(post_ve.mapping.values()) - 1.0) < 1e-10

    # Values must match closely
    for val in (0, 1):
        assert abs(post_enum.prob(val) - post_ve.prob(val)) < 1e-10

    # AIMA reference value: P(B=1 | J=1, M=1) approx 0.284
    assert 0.27 < post_ve.prob(1) < 0.30


def test_sampling_algorithms_convergence() -> None:
    """Statistical verification: Sampling methods converge toward exact posterior."""
    bn = _build_alarm_network()
    evidence = {"B": 1}
    exact = variable_elimination("M", evidence, bn)
    p_exact = exact.prob(1)

    rng = np.random.default_rng(42)
    # Likelihood weighting
    lw_dist = likelihood_weighting("M", evidence, bn, num_samples=10000, rng=rng)
    assert abs(lw_dist.prob(1) - p_exact) < 0.03

    # Gibbs sampling
    gibbs_dist = gibbs_sampling("M", evidence, bn, num_samples=10000, rng=rng)
    assert abs(gibbs_dist.prob(1) - p_exact) < 0.03
