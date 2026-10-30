"""Smoke test: validates importability of packages, config files, and core types."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from autogen.core import (
    DecisionTrace,
    SimulationSettings,
    load_settings,
)


@pytest.mark.integration
def test_all_modules_importable():
    packages = [
        "autogen",
        "autogen.core",
        "autogen.simulation",
        "autogen.agents",
        "autogen.intelligence",
        "autogen.memory",
        "autogen.experiments",
        "autogen.evaluation",
        "autogen.analytics",
        "autogen.visualization",
        "apps.api",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None


@pytest.mark.integration
def test_base_yaml_configuration():
    repo_root = Path(__file__).resolve().parent.parent.parent
    base_config_path = repo_root / "configs" / "base.yaml"
    assert base_config_path.exists(), f"Missing config file: {base_config_path}"

    settings = load_settings(base_config_path)
    assert isinstance(settings, SimulationSettings)
    assert settings.world.width >= 4
    assert settings.seeds.master_seed >= 0


@pytest.mark.integration
def test_simulation_default_yaml_configuration():
    repo_root = Path(__file__).resolve().parent.parent.parent
    sim_config_path = repo_root / "configs" / "simulation_default.yaml"
    assert sim_config_path.exists(), f"Missing config file: {sim_config_path}"

    settings = load_settings(sim_config_path)
    assert isinstance(settings, SimulationSettings)
    assert settings.logging.format.value == "json"


@pytest.mark.integration
def test_decision_trace_immutability():
    trace = DecisionTrace(
        agent_id="agent_alpha",
        step_id=1,
        algorithm_name="bfs_pathfinder",
        decision_type="action_selection",
        metrics={"nodes_expanded": 12, "depth": 3},
        rationale={"chosen_action": "MOVE_NORTH"},
    )
    assert trace.agent_id == "agent_alpha"
    assert trace.step_id == 1

    # Invariant: DecisionTrace must be frozen/immutable
    with pytest.raises(ValidationError):
        trace.agent_id = "mutated_agent"
