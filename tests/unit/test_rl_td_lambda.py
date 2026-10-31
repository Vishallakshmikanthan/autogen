"""Unit tests for Monte Carlo, TD(0), n-step TD, and TD(lambda) (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.rl.td_lambda import MonteCarlo, NStepTD, TDLambda, TDZero


def test_monte_carlo_value_estimation() -> None:
    """Verifies that Monte Carlo computes discounted returns accurately."""
    mc = MonteCarlo(gamma=0.9, first_visit=True)

    # Episode: s0 (r=1) -> s1 (r=2) -> s2 (r=10, term)
    # G(s2) = 10.0
    # G(s1) = 2.0 + 0.9 * 10.0 = 11.0
    # G(s0) = 1.0 + 0.9 * 11.0 = 10.9
    episode = [("s0", 1.0), ("s1", 2.0), ("s2", 10.0)]
    mc.update_episode(episode)

    assert abs(mc.v_values["s2"] - 10.0) < 1e-10
    assert abs(mc.v_values["s1"] - 11.0) < 1e-10
    assert abs(mc.v_values["s0"] - 10.9) < 1e-10


def test_td_zero_convergence() -> None:
    """Verifies that TD(0) converges toward true state values along a Markov chain."""
    td0 = TDZero(alpha=0.1, gamma=0.9)

    for _ in range(300):
        td0.update(state="s0", reward=1.0, next_state="s1", terminated=False)
        td0.update(state="s1", reward=10.0, next_state="term", terminated=True)

    assert abs(td0.v_values["s1"] - 10.0) < 0.1
    assert abs(td0.v_values["s0"] - (1.0 + 0.9 * 10.0)) < 0.15


def test_n_step_td_prediction() -> None:
    """Verifies n-step TD computes multi-step targets accurately."""
    n_step = NStepTD(n=2, alpha=0.5, gamma=0.9)

    # Trajectory: s0 -> r=2 -> s1 -> r=4 -> s2 -> r=10 (term)
    states = ["s0", "s1", "s2"]
    rewards = [2.0, 4.0, 10.0]
    n_step.update_trajectory(states, rewards, terminated=True)

    assert "s0" in n_step.v_values
    assert "s1" in n_step.v_values
    assert n_step.v_values["s0"] > 0.0


def test_td_lambda_eligibility_traces() -> None:
    """Verifies that Watkins' Q(lambda) propagates TD errors backwards along eligibility traces."""
    td_lam = TDLambda(num_actions=2, alpha=0.5, gamma=0.9, lam=0.8, epsilon=0.0, seed=42)

    # Sequence: s0, a0 -> s1, a0 -> terminal, r=10
    # First step s0 -> s1 (r=0)
    td_lam.update(state="s0", action=0, reward=0.0, next_state="s1", terminated=False)
    # At this point, s0 trace is active
    assert td_lam.traces.get(("s0", 0), 0.0) > 0.0

    # Second step s1 -> terminal (r=10)
    td_lam.update(state="s1", action=0, reward=10.0, next_state="term", terminated=True)

    # With lambda > 0, the reward received at s1 should have updated s0 immediately!
    q_s0 = td_lam.q_table.get_q("s0")[0]
    assert q_s0 > 0.0, "Eligibility trace failed to propagate TD error to s0!"

