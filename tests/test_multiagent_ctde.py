"""Unit and mathematical tests for CTDE Centralized Critic and Self-Play (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.multiagent.centralized_critic import CentralizedCriticMARL
from autogen.intelligence.multiagent.self_play import OpponentPool


@pytest.mark.mathematical
def test_centralized_critic_marl_update():
    """Verifies that CentralizedCriticMARL trains centralized critic and updates decentralized actors."""
    marl = CentralizedCriticMARL(
        agent_ids=["agent_1", "agent_2"],
        local_obs_dim=3,
        num_actions=2,
        global_state_dim=6,
        gamma=0.9,
        seed=42,
    )

    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    s_next = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    obs = {"agent_1": np.array([0.1, 0.2, 0.3]), "agent_2": np.array([0.4, 0.5, 0.6])}
    actions = {"agent_1": 0, "agent_2": 1}

    # Perform CTDE update with team reward
    metrics = marl.update(
        global_state=s,
        local_obs_map=obs,
        actions_map=actions,
        team_reward=10.0,
        next_global_state=s_next,
        terminated=False,
    )

    assert "critic_loss" in metrics
    assert "advantage" in metrics
    assert not np.isnan(metrics["critic_loss"])
    assert not np.isnan(metrics["advantage"])


@pytest.mark.unit
def test_opponent_pool_elo_tracking():
    """Verifies Elo updates after competitive matches in OpponentPool."""
    pool = OpponentPool(k_factor=32.0, seed=42)

    snap_a = pool.add_snapshot("v1", iteration=10, parameters={"w": 1.0}, initial_elo=1200.0)
    snap_b = pool.add_snapshot("v2", iteration=20, parameters={"w": 2.0}, initial_elo=1200.0)

    # A defeats B
    new_ra, new_rb = pool.update_elo(snap_a, snap_b, score_a=1.0)
    assert new_ra > 1200.0
    assert new_rb < 1200.0
    assert snap_a.wins == 1
    assert snap_b.losses == 1

