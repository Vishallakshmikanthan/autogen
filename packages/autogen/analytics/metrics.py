"""Domain metrics computation for multi-agent simulation and benchmarking (SYS-17)."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.state.types import Position, TerrainType


class SimulationMetrics(BaseModel):
    """Container for aggregated simulation domain metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    survival_rate: float = Field(ge=0.0, le=1.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    average_return: float
    path_efficiency: float = Field(ge=0.0, le=1.0)
    resource_efficiency: float = Field(ge=0.0)
    population_growth_rate: float
    throughput_ticks_per_sec: float = Field(ge=0.0)
    total_nodes_expanded: int = Field(ge=0, default=0)
    total_model_updates: int = Field(ge=0, default=0)


class SurvivalRateComputer:
    """Computes survival rate SR = N_survived / N_initial."""

    @staticmethod
    def compute(initial_agents: int, surviving_agents: int) -> float:
        if initial_agents <= 0:
            return 0.0
        return float(min(1.0, max(0.0, surviving_agents / initial_agents)))


class SuccessRateComputer:
    """Computes task success rate across evaluation trials."""

    @staticmethod
    def compute(successful_trials: int, total_trials: int) -> float:
        if total_trials <= 0:
            return 0.0
        return float(min(1.0, max(0.0, successful_trials / total_trials)))


class PathEfficiencyComputer:
    """Computes path efficiency PE = L_optimal / L_actual.

    L_optimal is determined via Breadth-First Search (BFS) over walkable grid cells.
    """

    @staticmethod
    def compute_optimal_path_length(
        grid: Sequence[Sequence[Any]],
        start: tuple[int, int] | Position,
        goal: tuple[int, int] | Position,
    ) -> int | None:
        start_pt = (start.x, start.y) if isinstance(start, Position) else start
        goal_pt = (goal.x, goal.y) if isinstance(goal, Position) else goal

        if start_pt == goal_pt:
            return 0

        height = len(grid)
        if height == 0:
            return None
        width = len(grid[0])

        queue: deque[tuple[int, int, int]] = deque([(start_pt[0], start_pt[1], 0)])
        visited: set[tuple[int, int]] = {start_pt}

        while queue:
            cx, cy, dist = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    cell = grid[ny][nx]
                    terrain = getattr(cell, "terrain", None)
                    # Obstacle check
                    if terrain == TerrainType.OBSTACLE:
                        continue
                    if (nx, ny) == goal_pt:
                        return dist + 1
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
        return None

    @classmethod
    def compute(
        cls,
        actual_path_length: int,
        grid: Sequence[Sequence[Any]] | None = None,
        start: tuple[int, int] | Position | None = None,
        goal: tuple[int, int] | Position | None = None,
        optimal_length: int | None = None,
    ) -> float:
        if actual_path_length <= 0:
            return 0.0

        opt_len = optimal_length
        if opt_len is None and grid is not None and start is not None and goal is not None:
            opt_len = cls.compute_optimal_path_length(grid, start, goal)

        if opt_len is None:
            return 0.0

        if opt_len == 0:
            return 1.0 if actual_path_length == 0 else 0.0

        return float(min(1.0, max(0.0, opt_len / actual_path_length)))


class ResourceEfficiencyComputer:
    """Computes resource collection efficiency: resources collected / max(1.0, energy expended)."""

    @staticmethod
    def compute(resources_collected: float, energy_expended: float) -> float:
        energy_denom = max(1.0, float(energy_expended))
        return float(max(0.0, resources_collected / energy_denom))


class PopulationGrowthComputer:
    """Computes finite population growth rate lambda = (N_t - N_t0) / max(1, N_t0)."""

    @staticmethod
    def compute(initial_population: int, final_population: int) -> float:
        if initial_population <= 0:
            return 0.0
        return float((final_population - initial_population) / initial_population)


class SimulationThroughputComputer:
    """Computes simulation execution throughput in ticks per wall-clock second."""

    @staticmethod
    def compute(total_ticks: int, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 1e-9 or total_ticks <= 0:
            return 0.0
        return float(total_ticks / elapsed_seconds)
