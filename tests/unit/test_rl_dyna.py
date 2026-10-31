"""Unit tests for model-based Dyna-Q learning (SYS-12, Module M13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.rl.adapter import CliffWalkingEnv
from autogen.intelligence.rl.dyna import DynaQ
from autogen.intelligence.rl.tabular import QLearning


def test_dyna_q_learns_environment_model() -> None:
    """Verifies that DynaQ accurately records transitions in its internal tabular model."""
    dyna = DynaQ(num_actions=4, n_planning_steps=5, seed=42)

    dyna.update(state="s1", action=0, reward=2.5, next_state="s2", terminated=False)
    assert ("s1", 0) in dyna.model
    reward, next_s, term = dyna.model[("s1", 0)]
    assert reward == 2.5
    assert next_s == "s2"
    assert not term
    assert "s1" in dyna.observed_states


def test_dyna_q_sample_efficiency_outperforms_pure_q_learning() -> None:
    """Verifies that Dyna-Q with planning steps (n=10) converges faster than pure Q-learning (n=0)."""
    # 4-state deterministic line: 0 -> 1 -> 2 -> 3 (term, r=10)
    def train_steps_until_solved(n_planning: int) -> int:
        agent = DynaQ(num_actions=2, alpha=0.5, gamma=0.9, n_planning_steps=n_planning, seed=42)
        episodes = 0
        solved = False

        while not solved and episodes < 50:
            episodes += 1
            s = 0
            while s < 3:
                a = agent.select_action(s, explore=True)
                next_s = s + 1 if a == 0 else s
                r = 10.0 if next_s == 3 else 0.0
                term = next_s == 3
                agent.update(s, a, r, next_s, term)
                s = next_s

            # Check if optimal path 0->1->2 is recognized
            q0 = agent.q_table.get_q(0)[0]
            if q0 > 2.0:
                solved = True

        return episodes

    episodes_dyna = train_steps_until_solved(n_planning=10)
    episodes_pure = train_steps_until_solved(n_planning=0)

    assert episodes_dyna <= episodes_pure

