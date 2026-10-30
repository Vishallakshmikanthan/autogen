"""Unit tests for GridWorld to Tabular MDP compilation and Russell & Norvig 4x3 reproduction."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.decision.gridworld_mdp import classic_russell_norvig_grid
from autogen.intelligence.decision.solvers import policy_iteration, value_iteration


def test_russell_norvig_4x3_transition_probabilities_valid() -> None:
    """Verifies that the compiled transition tensor satisfies row normalization."""
    mdp = classic_russell_norvig_grid(step_cost=-0.04, slip_prob=0.2, gamma=1.0)
    assert mdp.num_states == 11  # 4*3 - 1 wall at (1, 1)
    assert mdp.num_actions == 4

    row_sums = np.sum(mdp.transition_tensor, axis=2)
    for s in range(mdp.num_states):
        for a in range(mdp.num_actions):
            assert abs(row_sums[s, a] - 1.0) < 1e-10


def test_russell_norvig_4x3_textbook_values() -> None:
    """Verifies that Value Iteration and Policy Iteration reproduce classic AIMA textbook values.

    AIMA Reference Values (Section 17.1, Fig 17.3 for R(s)=-0.04, gamma=1.0):
      State (2, 2) approx 0.812
      State (0, 2) approx 0.645
      State (0, 0) approx 0.705
      State (3, 0) approx 0.388
    """
    mdp = classic_russell_norvig_grid(step_cost=-0.04, slip_prob=0.2, gamma=1.0)
    sol_vi = value_iteration(mdp, theta=1e-6)
    sol_pi = policy_iteration(mdp)

    # Values between VI and PI must match
    assert np.max(np.abs(sol_vi.v_star - sol_pi.v_star)) < 1e-5

    state_map = {name: i for i, name in enumerate(mdp.states)}

    # Cell (2, 2) is neighbor to (+1) at (3, 2): approx 0.918
    v_2_2 = sol_vi.v_star[state_map["(2,2)"]]
    assert 0.91 < v_2_2 < 0.93

    # Cell (0, 2) top-left: approx 0.812
    v_0_2 = sol_vi.v_star[state_map["(0,2)"]]
    assert 0.80 < v_0_2 < 0.83

    # Cell (0, 0) bottom-left start: approx 0.705
    v_0_0 = sol_vi.v_star[state_map["(0,0)"]]
    assert 0.69 < v_0_0 < 0.72

    # Cell (3, 0) bottom-right: approx 0.388
    v_3_0 = sol_vi.v_star[state_map["(3,0)"]]
    assert 0.37 < v_3_0 < 0.40
