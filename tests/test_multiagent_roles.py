"""Unit tests for emergent role detection and Gini inequality metrics (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.multiagent.emergent_roles import RoleDetector
from autogen.intelligence.multiagent.metrics import MultiAgentMetrics


@pytest.mark.mathematical
def test_gini_coefficient_bounds_and_symmetry():
    """Verifies that Gini coefficient is 0.0 under perfect equality and approaches 1.0 under extreme concentration."""
    # Perfect equality: all agents hold 10.0 resources
    equal_wealth = [10.0, 10.0, 10.0, 10.0]
    assert np.isclose(MultiAgentMetrics.gini_coefficient(equal_wealth), 0.0)

    # Extreme inequality: one agent holds 1000.0, others hold 0.0
    skewed_wealth = [1000.0, 0.0, 0.0, 0.0, 0.0]
    gini_skewed = MultiAgentMetrics.gini_coefficient(skewed_wealth)
    # Theoretical Gini for 1 rich and 4 poor out of 5 is (5 - 1) / 5 = 0.8
    assert np.isclose(gini_skewed, 0.8, atol=1e-5)


@pytest.mark.mathematical
def test_emergent_role_detector_clustering():
    """Verifies that RoleDetector clusters agent behavioral feature profiles into roles."""
    detector = RoleDetector(n_roles=3, seed=42)

    # 6 agents: 2 scouts (high exploration), 2 gatherers (high foraging), 2 free-riders (low both)
    agent_features = {
        "scout_1": np.array([0.1, 0.9, 5.0, 0.1]),
        "scout_2": np.array([0.15, 0.85, 4.0, 0.05]),
        "gather_1": np.array([0.9, 0.2, 1.0, 0.2]),
        "gather_2": np.array([0.85, 0.25, 2.0, 0.15]),
        "free_1": np.array([0.05, 0.05, 0.0, 0.0]),
        "free_2": np.array([0.02, 0.08, 0.0, 0.0]),
    }

    roles = detector.fit_predict(agent_features)
    assert len(roles) == 6
    # Same types should cluster into identical role labels
    assert roles["scout_1"] == roles["scout_2"]
    assert roles["gather_1"] == roles["gather_2"]
    assert roles["free_1"] == roles["free_2"]
    # Distinct behavioral types should not all share one role
    distinct_roles = set(roles.values())
    assert len(distinct_roles) == 3
