"""Matrix game sanity and game theory tests for Module M15 (SYS-14)."""

from __future__ import annotations

import pytest

from autogen.intelligence.multiagent.iql import IndependentQLearning
from autogen.intelligence.multiagent.matrix_games import MatrixGame


@pytest.mark.mathematical
def test_prisoners_dilemma_defection_dominance():
    """Verifies that Independent Q-Learning converges to mutual Defection in Prisoner's Dilemma."""
    game = MatrixGame.prisoners_dilemma()
    # 0 = Cooperate, 1 = Defect
    iql = IndependentQLearning(agent_ids=["p1", "p2"], num_actions=2, alpha=0.1, gamma=0.9, epsilon=0.05, seed=42)

    # State is fixed (single stage game)
    state = "matrix_stage"
    n_rounds = 400

    for _ in range(n_rounds):
        actions = iql.select_actions({"p1": state, "p2": state}, explore=True)
        a1, a2 = actions["p1"], actions["p2"]
        r1, r2 = game.get_payoffs(a1, a2)

        iql.update(
            prev_obs={"p1": state, "p2": state},
            actions=actions,
            rewards={"p1": r1, "p2": r2},
            next_obs={"p1": state, "p2": state},
            terminated=False,
        )

    # In Prisoner's Dilemma, Defect (action 1) strictly dominates Cooperate (action 0)
    q1 = iql.learners["p1"].q_table.get_q(state)
    q2 = iql.learners["p2"].q_table.get_q(state)

    assert q1[1] > q1[0], f"Agent 1 did not learn Defect: Q={q1}"
    assert q2[1] > q2[0], f"Agent 2 did not learn Defect: Q={q2}"


@pytest.mark.mathematical
def test_stag_hunt_coordination_convergence():
    """Verifies that IQL converges to Pareto-optimal mutual Stag when seeded towards cooperation."""
    game = MatrixGame.stag_hunt()
    # 0 = Stag (Payoff 4 if mutual), 1 = Hare (Payoff 2 safe)
    iql = IndependentQLearning(agent_ids=["p1", "p2"], num_actions=2, alpha=0.1, gamma=0.9, epsilon=0.05, seed=42)
    state = "stag_hunt_stage"

    # Seed initial Q-table with slight optimistic exploration towards Stag
    iql.learners["p1"].q_table.set_q(state, 0, 5.0)
    iql.learners["p2"].q_table.set_q(state, 0, 5.0)

    for _ in range(250):
        actions = iql.select_actions({"p1": state, "p2": state}, explore=True)
        a1, a2 = actions["p1"], actions["p2"]
        r1, r2 = game.get_payoffs(a1, a2)

        iql.update(
            prev_obs={"p1": state, "p2": state},
            actions=actions,
            rewards={"p1": r1, "p2": r2},
            next_obs={"p1": state, "p2": state},
            terminated=False,
        )

    q1 = iql.learners["p1"].q_table.get_q(state)
    q2 = iql.learners["p2"].q_table.get_q(state)

    # Must coordinate on Stag (action 0) yielding optimal payoff (4, 4)
    assert q1[0] > q1[1]
    assert q2[0] > q2[1]


@pytest.mark.unit
def test_matching_pennies_payoff_balance():
    """Verifies zero-sum symmetry in Matching Pennies payoffs."""
    game = MatrixGame.matching_pennies()
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            r1, r2 = game.get_payoffs(a1, a2)
            assert r1 + r2 == 0.0, f"Payoff is not zero sum: {r1} + {r2}"
