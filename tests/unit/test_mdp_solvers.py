"""Unit tests for exact MDP solvers (Linear Solve, Policy Evaluation, Value Iteration, Policy Iteration) (Module M11)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.decision.contracts import TabularMDP
from autogen.intelligence.decision.solvers import (
    iterative_policy_evaluation,
    linear_policy_evaluation,
    policy_iteration,
    value_iteration,
)


def _build_2state_analytical_mdp() -> TabularMDP:
    """Canonical 2-state analytical MDP.

    States: S0, S1.
    Actions: a0 (stay), a1 (switch).
    Transitions:
      T(S0, a0, S0) = 1.0, R = +1.0
      T(S0, a1, S1) = 1.0, R = 0.0
      T(S1, a0, S1) = 1.0, R = +2.0
      T(S1, a1, S0) = 1.0, R = 0.0
    Discount: gamma = 0.9.
    """
    T = np.zeros((2, 2, 2), dtype=np.float64)
    R = np.zeros((2, 2, 2), dtype=np.float64)

    # S0
    T[0, 0, 0] = 1.0
    R[0, 0, 0] = 1.0
    T[0, 1, 1] = 1.0
    R[0, 1, 1] = 0.0

    # S1
    T[1, 0, 1] = 1.0
    R[1, 0, 1] = 2.0
    T[1, 1, 0] = 1.0
    R[1, 1, 0] = 0.0

    return TabularMDP(
        num_states=2,
        num_actions=2,
        states=("S0", "S1"),
        actions=("stay", "switch"),
        transition_tensor=T,
        reward_tensor=R,
        gamma=0.9,
    )


def test_2state_analytical_mdp_solution() -> None:
    """Verifies that Value Iteration and Policy Iteration match hand-derived closed-form solutions."""
    mdp = _build_2state_analytical_mdp()

    # Optimal policy is: S0 -> switch to S1 (action 1), S1 -> stay at S1 (action 0)
    # V*(S1) = 2.0 / (1 - 0.9) = 20.0
    # V*(S0) = 0.0 + 0.9 * 20.0 = 18.0
    sol_vi = value_iteration(mdp, theta=1e-12)
    sol_pi = policy_iteration(mdp)

    assert abs(sol_vi.v_star[1] - 20.0) < 1e-6
    assert abs(sol_vi.v_star[0] - 18.0) < 1e-6

    assert abs(sol_pi.v_star[1] - 20.0) < 1e-10
    assert abs(sol_pi.v_star[0] - 18.0) < 1e-10

    # Optimal policy: S0 -> a1 (switch = index 1), S1 -> a0 (stay = index 0)
    assert sol_vi.policy[0] == 1
    assert sol_vi.policy[1] == 0
    assert (sol_vi.policy == sol_pi.policy).all()


def test_linear_solve_vs_iterative_policy_evaluation_oracle() -> None:
    """Oracle test: Iterative policy evaluation matches exact linear matrix solve (I - gamma*P_pi)^-1 * R_pi."""
    mdp = _build_2state_analytical_mdp()
    # Test across all 2^2 = 4 deterministic policies
    for p0 in (0, 1):
        for p1 in (0, 1):
            policy = np.array([p0, p1], dtype=np.int32)
            v_linear = linear_policy_evaluation(mdp, policy)
            v_iterative = iterative_policy_evaluation(mdp, policy, theta=1e-10)

            diff = np.max(np.abs(v_linear - v_iterative))
            assert diff < 1e-6, f"Mismatch for policy {policy}: diff={diff}"


def test_value_iteration_policy_iteration_equivalence() -> None:
    """Oracle test: Value Iteration and Policy Iteration reach identical V* and Q* on 3-state chain MDP."""
    # 3-state chain: S0 -> S1 -> S2 (terminal +10)
    T = np.zeros((3, 2, 3), dtype=np.float64)
    R = np.zeros((3, 2, 3), dtype=np.float64)

    # S0
    T[0, 0, 0] = 1.0
    R[0, 0, 0] = 0.0
    T[0, 1, 1] = 1.0
    R[0, 1, 1] = 0.0

    # S1
    T[1, 0, 0] = 1.0
    R[1, 0, 0] = 0.0
    T[1, 1, 2] = 1.0
    R[1, 1, 2] = 10.0

    # S2 (terminal)
    T[2, 0, 2] = 1.0
    T[2, 1, 2] = 1.0

    mdp = TabularMDP(
        num_states=3,
        num_actions=2,
        states=("S0", "S1", "S2"),
        actions=("back", "forward"),
        transition_tensor=T,
        reward_tensor=R,
        gamma=0.9,
        terminal_states=frozenset({2}),
    )

    sol_vi = value_iteration(mdp, theta=1e-10)
    sol_pi = policy_iteration(mdp)

    # Values must match to high numerical precision
    assert np.max(np.abs(sol_vi.v_star - sol_pi.v_star)) < 1e-8
    assert (sol_vi.policy == sol_pi.policy).all()
