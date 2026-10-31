"""Explicit test for strict terminal state target handling (zero bootstrap) (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.rl.tabular import (
    DoubleQLearning,
    ExpectedSARSA,
    QLearning,
    SARSA,
)


def test_q_learning_terminal_target_has_zero_bootstrap() -> None:
    """Verifies invariant: On terminal transition, target == reward exactly, with zero future bootstrap."""
    ql = QLearning(num_actions=4, alpha=1.0, gamma=0.99)
    # Set high Q-value on terminal state s_term to verify it is NOT bootstrapped
    ql.q_table.set_q("s_term", 0, 999.0)
    ql.q_table.set_q("s_start", 0, 0.0)

    # Execute transition into terminal state
    td_error = ql.update(
        state="s_start",
        action=0,
        reward=5.0,
        next_state="s_term",
        terminated=True,
    )

    # With alpha=1.0, new Q(s_start, 0) must equal target exactly
    new_q = ql.q_table.get_q("s_start")[0]
    assert new_q == 5.0, f"Expected Q=5.0 (pure reward), got {new_q} (bootstrapped from s_term!)"
    assert td_error == 5.0


def test_sarsa_terminal_target_has_zero_bootstrap() -> None:
    """Verifies invariant: SARSA terminal target equals reward exactly."""
    sarsa = SARSA(num_actions=4, alpha=1.0, gamma=0.95)
    sarsa.q_table.set_q("s_term", 1, 500.0)
    sarsa.q_table.set_q("s_0", 2, 0.0)

    td_err = sarsa.update(
        state="s_0",
        action=2,
        reward=-10.0,
        next_state="s_term",
        terminated=True,
        next_action=1,
    )

    assert sarsa.q_table.get_q("s_0")[2] == -10.0
    assert td_err == -10.0


def test_double_q_learning_terminal_target_has_zero_bootstrap() -> None:
    """Verifies invariant: Double Q-learning terminal target equals reward exactly."""
    dql = DoubleQLearning(num_actions=4, alpha=1.0, gamma=0.90)
    dql.q1_table.set_q("s_term", 0, 888.0)
    dql.q2_table.set_q("s_term", 0, 777.0)
    dql.q1_table.set_q("s_curr", 0, 0.0)
    dql.q2_table.set_q("s_curr", 0, 0.0)

    dql.update(
        state="s_curr",
        action=0,
        reward=7.5,
        next_state="s_term",
        terminated=True,
    )

    # Regardless of which table was updated, the updated value must be 7.5
    updated_val = max(dql.q1_table.get_q("s_curr")[0], dql.q2_table.get_q("s_curr")[0])
    assert updated_val == 7.5

