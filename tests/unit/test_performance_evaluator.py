"""Unit and mathematical tests for PerformanceEvaluator against exact test oracles."""

import pytest

from autogen.evaluation.performance import PerformanceEvaluator
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import Position


def test_discounted_return_exact_oracle():
    """Verify J = sum gamma^t * r_t against analytical calculation."""
    gamma = 0.5
    evaluator = PerformanceEvaluator(gamma=gamma)

    # Synthetic setup with mock GridWorld states
    cfg = EnvironmentConfig(width=5, height=5, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)
    world.spawn_agent("a1", Position(2, 2))

    # 3 ticks of simulation trajectory
    state_0 = world.state
    state_1 = state_0.model_copy(update={"step": 1})
    state_2 = state_0.model_copy(update={"step": 2})

    trajectory = [state_0, state_1, state_2]
    # Rewards at t=0, 1, 2: [10.0, 4.0, 8.0]
    # J = 10.0 * 1.0 + 4.0 * 0.5 + 8.0 * 0.25 = 10.0 + 2.0 + 2.0 = 14.0
    rewards = {"a1": [10.0, 4.0, 8.0]}

    pop_perf = evaluator.evaluate_trajectories(trajectory, rewards)
    a1_perf = pop_perf.per_agent["a1"]

    assert a1_perf.discounted_return == pytest.approx(14.0, abs=1e-4)
    assert a1_perf.survival_time == 2
    assert a1_perf.total_steps == 2
    assert a1_perf.is_alive_at_end is True


def test_energy_efficiency_and_coverage_oracles():
    evaluator = PerformanceEvaluator(gamma=0.9)
    cfg = EnvironmentConfig(width=5, height=5, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)
    world.spawn_agent("a1", Position(1, 1), energy=100.0)

    state_0 = world.state

    # Step 1: Agent moves to (1, 2), spent 10 energy, collected 5 food
    agent_s1 = state_0.agents["a1"].model_copy(
        update={"position": Position(1, 2), "energy": 90.0, "carried_food": 5.0}
    )
    state_1 = state_0.model_copy(update={"step": 1, "agents": {"a1": agent_s1}})

    # Step 2: Agent moves to (1, 3), spent 10 energy, collected 15 more food (total 20)
    agent_s2 = agent_s1.model_copy(
        update={"position": Position(1, 3), "energy": 80.0, "carried_food": 20.0}
    )
    state_2 = state_1.model_copy(update={"step": 2, "agents": {"a1": agent_s2}})

    trajectory = [state_0, state_1, state_2]
    rewards = {"a1": [1.0, 1.0]}

    pop_perf = evaluator.evaluate_trajectories(trajectory, rewards)
    m = pop_perf.per_agent["a1"]

    # Total food collected = 20.0
    assert m.resources_collected == pytest.approx(20.0)
    # Energy spent = (100 - 90) + (90 - 80) = 20.0
    # Efficiency = 20.0 / 20.0 = 1.0
    assert m.energy_efficiency == pytest.approx(1.0)
    # Visited: (1, 1), (1, 2), (1, 3) = 3 unique positions
    # Coverage: 3 / walkable_cells_count
    assert 0.0 < m.exploration_coverage <= 1.0


def test_evaluator_validation_checks():
    with pytest.raises(ValueError, match="Discount factor gamma"):
        PerformanceEvaluator(gamma=-0.1)

    with pytest.raises(ValueError, match="Discount factor gamma"):
        PerformanceEvaluator(gamma=1.5)

    evaluator = PerformanceEvaluator()
    with pytest.raises(ValueError, match="Trajectory history cannot be empty"):
        evaluator.evaluate_trajectories([], {})
