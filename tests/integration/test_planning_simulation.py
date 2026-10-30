"""Integration test: PlanningAgent executing in SimulationEngine with strict determinism."""

from __future__ import annotations

import pytest

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.planning.planning_agent import PlanningAgent
from autogen.intelligence.planning.domains.survival import create_survival_crafting_problem
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position


def _build_planning_simulation(seed: int = 42) -> tuple[SimulationEngine, PlanningAgent]:
    config = EnvironmentConfig(width=15, height=15, seed=seed)
    world = GridWorld(config)
    world.reset()

    planning_agent = PlanningAgent(agent_id="planner_1")
    problem = create_survival_crafting_problem()
    planning_agent.set_problem(problem)

    # Associate high-level landmarks with grid positions
    planning_agent.set_target_location("Camp", Position(5, 5))
    planning_agent.set_target_location("Forest", Position(8, 5))
    planning_agent.set_target_location("Meadow", Position(5, 8))

    world.spawn_agent("planner_1")

    engine = SimulationEngine(environment=world, seed=seed)
    p_cfg = PerceptionConfig(radius=6)
    engine.register_agent(planning_agent, p_cfg)

    return engine, planning_agent


def test_planning_agent_simulation_determinism() -> None:
    """Verifies that running a simulation with PlanningAgent is strictly deterministic."""
    engine1, _ = _build_planning_simulation(seed=123)
    engine2, _ = _build_planning_simulation(seed=123)

    hashes1: list[str] = []
    hashes2: list[str] = []

    for _ in range(25):
        s1 = engine1.step()
        s2 = engine2.step()
        hashes1.append(canonical_world_hash(s1))
        hashes2.append(canonical_world_hash(s2))

    assert hashes1 == hashes2, "Simulation diverged non-deterministically with PlanningAgent!"


def test_planning_agent_produces_authentic_traces() -> None:
    """Verifies that PlanningAgent generates real DecisionTraces with plan metadata."""
    engine, agent = _build_planning_simulation(seed=999)

    for step_num in range(10):
        engine.step()
        trace = agent.latest_trace
        assert trace is not None
        assert trace.agent_id == "planner_1"
        assert trace.algorithm_name == "ClassicalPlanning"
        assert "selected_action" in trace.rationale

