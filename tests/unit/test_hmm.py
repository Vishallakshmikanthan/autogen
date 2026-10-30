"""Unit tests for Hidden Markov Model (Forward, Backward, Viterbi, Smoothing) (Module M10)."""

from __future__ import annotations

import itertools
import math
import numpy as np
import pytest

from autogen.intelligence.probabilistic.hmm import HiddenMarkovModel


def _build_weather_umbrella_hmm() -> HiddenMarkovModel:
    """Canonical Weather-Umbrella HMM (AIMA Chapter 14).

    States: 0 = Rainy, 1 = Sunny
    Observations: 0 = NoUmbrella, 1 = Umbrella
    """
    pi0 = np.array([0.5, 0.5])
    # T[i, j] = P(S_t = j | S_{t-1} = i)
    trans = np.array([
        [0.7, 0.3],  # Rainy -> [Rainy, Sunny]
        [0.3, 0.7],  # Sunny -> [Rainy, Sunny]
    ])
    # E[i, k] = P(O_t = k | S_t = i)
    emiss = np.array([
        [0.1, 0.9],  # Rainy -> [NoUmbrella: 0.1, Umbrella: 0.9]
        [0.8, 0.2],  # Sunny -> [NoUmbrella: 0.8, Umbrella: 0.2]
    ])
    return HiddenMarkovModel(transition_matrix=trans, emission_matrix=emiss, initial_prior=pi0)


def test_hmm_forward_vs_brute_force_sequence_sum_oracle() -> None:
    """Oracle verification: Forward probability equals brute-force sum over all hidden state sequences."""
    hmm = _build_weather_umbrella_hmm()
    observations = [1, 1, 0]  # Umbrella, Umbrella, No Umbrella
    T = len(observations)

    # 1. Forward algorithm total probability
    alpha, log_lik = hmm.forward(observations)
    forward_total_prob = float(np.sum(alpha[-1]))

    # 2. Exhaustive brute-force summation over all 2^3 = 8 state paths
    brute_force_total_prob = 0.0
    for path in itertools.product(range(hmm.num_states), repeat=T):
        # Joint probability P(s_0, ..., s_{T-1}, o_0, ..., o_{T-1})
        p_path = hmm.initial_prior[path[0]] * hmm.emission_matrix[path[0], observations[0]]
        for t in range(1, T):
            p_path *= hmm.transition_matrix[path[t - 1], path[t]] * hmm.emission_matrix[path[t], observations[t]]
        brute_force_total_prob += p_path

    assert abs(forward_total_prob - brute_force_total_prob) < 1e-12
    assert abs(log_lik - math.log(brute_force_total_prob)) < 1e-10


def test_hmm_viterbi_vs_brute_force_argmax_oracle() -> None:
    """Oracle verification: Viterbi optimal path matches exhaustive brute-force maximum path."""
    hmm = _build_weather_umbrella_hmm()
    observations = [1, 1, 0, 1]
    T = len(observations)

    viterbi_path, viterbi_log_p = hmm.viterbi(observations)

    # Brute-force find argmax path
    best_path = None
    best_log_p = -math.inf
    for path in itertools.product(range(hmm.num_states), repeat=T):
        p_path = hmm.initial_prior[path[0]] * hmm.emission_matrix[path[0], observations[0]]
        for t in range(1, T):
            p_path *= hmm.transition_matrix[path[t - 1], path[t]] * hmm.emission_matrix[path[t], observations[t]]
        log_p = math.log(p_path) if p_path > 0 else -math.inf
        if log_p > best_log_p:
            best_log_p = log_p
            best_path = path

    assert viterbi_path == best_path
    assert abs(viterbi_log_p - best_log_p) < 1e-10


def test_hmm_forward_backward_smoothing() -> None:
    """Verifies that forward-backward smoothing produces valid probability distributions at each tick."""
    hmm = _build_weather_umbrella_hmm()
    observations = [1, 1, 0, 1, 0]
    result = hmm.forward_backward_smoothing(observations)

    # Every row of smoothed_posteriors must sum to 1.0
    for t in range(len(observations)):
        assert abs(np.sum(result.smoothed_posteriors[t]) - 1.0) < 1e-10

