"""Unit tests for MDP exact solution serialization (ground truth for M13 RL) (Module M11)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.decision.gridworld_mdp import classic_russell_norvig_grid
from autogen.intelligence.decision.serialization import (
    export_exact_mdp_solution,
    load_exact_mdp_solution,
)
from autogen.intelligence.decision.solvers import value_iteration


def test_mdp_solution_serialization_round_trip(tmp_path) -> None:
    """Verifies that an exact MDP solution exports and loads back identically."""
    mdp = classic_russell_norvig_grid()
    sol_original = value_iteration(mdp)

    out_file = tmp_path / "exact_rn_solution.json"
    export_exact_mdp_solution(sol_original, out_file)
    assert out_file.exists()

    sol_loaded = load_exact_mdp_solution(out_file)

    assert np.allclose(sol_original.v_star, sol_loaded.v_star, atol=1e-12)
    assert np.allclose(sol_original.q_star, sol_loaded.q_star, atol=1e-12)
    assert (sol_original.policy == sol_loaded.policy).all()
    assert sol_original.iterations == sol_loaded.iterations
    assert sol_original.state_labels == sol_loaded.state_labels
    assert sol_original.action_labels == sol_loaded.action_labels

