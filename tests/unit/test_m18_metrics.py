"""Unit tests for M18 domain metrics computation (SYS-17)."""

import pytest

from autogen.analytics.metrics import (
    PathEfficiencyComputer,
    PopulationGrowthComputer,
    ResourceEfficiencyComputer,
    SimulationThroughputComputer,
    SurvivalRateComputer,
)
from autogen.simulation.state.types import Position, TerrainType
from autogen.simulation.state.world_state import CellState


def test_survival_rate_computation():
    assert SurvivalRateComputer.compute(0, 0) == 0.0
    assert SurvivalRateComputer.compute(10, 10) == 1.0
    assert SurvivalRateComputer.compute(10, 5) == 0.5
    assert SurvivalRateComputer.compute(10, 0) == 0.0
    assert SurvivalRateComputer.compute(10, 15) == 1.0  # Clamped to 1.0


def test_resource_efficiency_computation():
    # Efficiency = resources / max(1.0, energy)
    assert ResourceEfficiencyComputer.compute(0.0, 50.0) == 0.0
    assert ResourceEfficiencyComputer.compute(100.0, 0.0) == 100.0  # energy denom clamped to 1.0
    assert pytest.approx(ResourceEfficiencyComputer.compute(50.0, 100.0), abs=1e-6) == 0.5


def test_population_growth_computation():
    assert PopulationGrowthComputer.compute(0, 5) == 0.0
    assert pytest.approx(PopulationGrowthComputer.compute(10, 15), abs=1e-6) == 0.5
    assert pytest.approx(PopulationGrowthComputer.compute(10, 5), abs=1e-6) == -0.5
    assert PopulationGrowthComputer.compute(10, 10) == 0.0


def test_simulation_throughput_computation():
    assert SimulationThroughputComputer.compute(0, 10.0) == 0.0
    assert SimulationThroughputComputer.compute(100, 0.0) == 0.0
    assert pytest.approx(SimulationThroughputComputer.compute(1000, 2.0), abs=1e-6) == 500.0


def test_path_efficiency_computation():
    # Simple 5x5 grid with no obstacles
    grid = [
        [
            CellState(
                position=Position(x=x, y=y),
                terrain=TerrainType.GRASSLAND,
                food=0.0,
                movement_cost=1.0,
            )
            for x in range(5)
        ]
        for y in range(5)
    ]

    start = Position(x=0, y=0)
    goal = Position(x=2, y=2)  # Optimal distance = 4 steps

    opt_len = PathEfficiencyComputer.compute_optimal_path_length(grid, start, goal)
    assert opt_len == 4

    # Perfect path efficiency
    pe_perfect = PathEfficiencyComputer.compute(actual_path_length=4, optimal_length=4)
    assert pe_perfect == 1.0

    # Suboptimal path
    pe_sub = PathEfficiencyComputer.compute(actual_path_length=8, optimal_length=4)
    assert pytest.approx(pe_sub, abs=1e-6) == 0.5

    # Zero actual steps
    assert PathEfficiencyComputer.compute(actual_path_length=0, optimal_length=4) == 0.0
