"""Performance evaluator for agent trajectories and population metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.state.world_state import WorldState


class AgentPerformance(BaseModel):
    """Evaluation metrics for an individual agent across an episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    discounted_return: float
    survival_time: int = Field(ge=0)
    resources_collected: float = Field(ge=0.0)
    energy_efficiency: float = Field(ge=0.0)
    exploration_coverage: float = Field(ge=0.0, le=1.0)
    total_steps: int = Field(ge=0)
    is_alive_at_end: bool


class MetricDistribution(BaseModel):
    """Statistical summary for a population metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float
    std: float
    median: float
    min: float
    max: float


class PopulationPerformance(BaseModel):
    """Aggregated evaluation across all agents in a simulation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_count: int
    discounted_return: MetricDistribution
    survival_time: MetricDistribution
    resources_collected: MetricDistribution
    energy_efficiency: MetricDistribution
    exploration_coverage: MetricDistribution
    per_agent: dict[str, AgentPerformance]


class PerformanceEvaluator:
    """Evaluates agent performance metrics according to Phase 01 specifications."""

    def __init__(self, gamma: float = 0.99) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Discount factor gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    def evaluate_trajectories(
        self,
        trajectory: Sequence[WorldState],
        per_tick_rewards: dict[str, list[float]],
    ) -> PopulationPerformance:
        """Calculates per-agent and population aggregate metrics from trajectory history."""
        if not trajectory:
            raise ValueError("Trajectory history cannot be empty.")

        initial_state = trajectory[0]
        final_state = trajectory[-1]
        all_agent_ids = sorted(initial_state.agents.keys())

        # Calculate total walkable cells in world
        walkable_cells_count = 0
        for row in initial_state.grid:
            for cell in row:
                if TERRAIN_PROPERTIES[cell.terrain].walkable:
                    walkable_cells_count += 1
        walkable_cells_count = max(1, walkable_cells_count)

        per_agent_metrics: dict[str, AgentPerformance] = {}

        for aid in all_agent_ids:
            rewards = per_tick_rewards.get(aid, [])

            # 1. Discounted Return: J = sum_{t=0}^T gamma^t * r_t
            discounted_j = 0.0
            discount = 1.0
            for r in rewards:
                discounted_j += discount * r
                discount *= self.gamma

            # 2. Survival Time & 3. Resources Collected & 4. Energy Efficiency & 5. Coverage
            visited_positions: set[tuple[int, int]] = set()
            survival_ticks = 0
            energy_spent = 0.0
            prev_energy: float | None = None
            total_food_collected = 0.0
            prev_carried_food: float = 0.0

            for tick, state in enumerate(trajectory):
                agent = state.agents.get(aid)
                if agent is None:
                    continue

                if agent.is_alive:
                    survival_ticks = tick
                    visited_positions.add((agent.position.x, agent.position.y))

                    if prev_energy is not None and agent.energy < prev_energy:
                        energy_spent += prev_energy - agent.energy
                    prev_energy = agent.energy

                    # Food delta if collected
                    if agent.carried_food > prev_carried_food:
                        total_food_collected += agent.carried_food - prev_carried_food
                    prev_carried_food = agent.carried_food

            final_agent = final_state.agents.get(aid)
            is_alive_end = bool(final_agent.is_alive) if final_agent else False
            if is_alive_end:
                survival_ticks = len(trajectory) - 1

            # Energy efficiency = resources_collected / energy_spent
            efficiency = (total_food_collected / energy_spent) if energy_spent > 0.0 else 0.0

            # Exploration coverage = unique visited / walkable
            coverage = min(1.0, len(visited_positions) / walkable_cells_count)

            per_agent_metrics[aid] = AgentPerformance(
                agent_id=aid,
                discounted_return=round(discounted_j, 4),
                survival_time=survival_ticks,
                resources_collected=round(total_food_collected, 2),
                energy_efficiency=round(efficiency, 4),
                exploration_coverage=round(coverage, 4),
                total_steps=len(trajectory) - 1,
                is_alive_at_end=is_alive_end,
            )

        def summarize(values: list[float]) -> MetricDistribution:
            arr = np.array(values, dtype=np.float64)
            return MetricDistribution(
                mean=round(float(np.mean(arr)), 4),
                std=round(float(np.std(arr)), 4),
                median=round(float(np.median(arr)), 4),
                min=round(float(np.min(arr)), 4),
                max=round(float(np.max(arr)), 4),
            )

        pop_metrics = PopulationPerformance(
            agent_count=len(all_agent_ids),
            discounted_return=summarize([m.discounted_return for m in per_agent_metrics.values()]),
            survival_time=summarize([float(m.survival_time) for m in per_agent_metrics.values()]),
            resources_collected=summarize(
                [m.resources_collected for m in per_agent_metrics.values()]
            ),
            energy_efficiency=summarize([m.energy_efficiency for m in per_agent_metrics.values()]),
            exploration_coverage=summarize(
                [m.exploration_coverage for m in per_agent_metrics.values()]
            ),
            per_agent=per_agent_metrics,
        )
        return pop_metrics
