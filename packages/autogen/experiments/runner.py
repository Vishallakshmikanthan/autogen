"""Experiment runner orchestrating parallelized, reproducible simulation suites (SYS-16)."""

from __future__ import annotations

import concurrent.futures
import time
from datetime import UTC, datetime
from typing import Any

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.analytics.metrics import (
    PathEfficiencyComputer,
    PopulationGrowthComputer,
    ResourceEfficiencyComputer,
    SimulationThroughputComputer,
    SurvivalRateComputer,
)
from autogen.analytics.statistics import SampleStats, sample_stats
from autogen.experiments.config import ExperimentConfig
from autogen.experiments.record import (
    ExperimentSuiteRecord,
    RunRecord,
    capture_environment_versions,
    compute_config_hash,
    get_git_commit,
)
from autogen.experiments.seed import SeedSplitter
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import TerrainType


def _execute_single_run(payload: tuple[dict[str, Any], int, str, str]) -> dict[str, Any]:
    """Top-level picklable task worker for parallel or sequential simulation execution."""
    config_dict, seed, config_hash, git_commit = payload
    config = ExperimentConfig.model_validate(config_dict)

    start_time = time.perf_counter()

    # 1. Initialize environment
    terrain_weights = {
        TerrainType.GRASSLAND: max(0.1, 1.0 - config.environment.obstacle_density - 0.2),
        TerrainType.FOREST: 0.20,
        TerrainType.DESERT: 0.0,
        TerrainType.WATER: 0.0,
        TerrainType.MOUNTAIN: 0.0,
        TerrainType.OBSTACLE: config.environment.obstacle_density,
    }
    env_cfg = EnvironmentConfig(
        width=config.environment.width,
        height=config.environment.height,
        seed=seed,
        energy_max=config.environment.energy_max,
        terrain_weights=terrain_weights,
    )
    world = GridWorld(config=env_cfg)
    world.reset(seed=seed)

    # 2. Spawn and instantiate agents
    engine = SimulationEngine(environment=world, seed=seed)
    initial_agent_count = len(config.agents)
    rewards_per_agent: dict[str, list[float]] = {a.agent_id: [] for a in config.agents}

    for agent_spec in config.agents:
        world.spawn_agent(agent_id=agent_spec.agent_id, energy=config.environment.energy_max)
        try:
            agent_cfg = AgentConfig(
                agent_type=agent_spec.agent_type,
                agent_id=agent_spec.agent_id,
                seed=seed,
                custom_params=agent_spec.policy_params,
            )
            agent = AgentRegistry.create(agent_cfg)
        except Exception:
            # Fallback to simple reflex or random
            if "random" in agent_spec.agent_type.lower():
                agent = RandomAgent(agent_id=agent_spec.agent_id, seed=seed)
            else:
                agent = SimpleReflexAgent(agent_id=agent_spec.agent_id)

        engine.register_agent(agent)

    # 3. Step through simulation ticks
    max_steps = config.environment.max_steps
    step_count = 0
    start_positions = {aid: a.position for aid, a in world.state.agents.items()}
    total_resources_collected = 0.0
    total_energy_expended = 0.0

    for _step_idx in range(max_steps):
        prev_state = world.state
        new_state = engine.step()
        step_count += 1

        # Track tick rewards & metrics
        for aid in list(rewards_per_agent.keys()):
            if aid in new_state.agents:
                curr_ag = new_state.agents[aid]
                prev_ag = prev_state.agents.get(aid)
                food_diff = curr_ag.carried_food - (prev_ag.carried_food if prev_ag else 0.0)
                reward = 10.0 * max(0.0, food_diff) + (0.1 if curr_ag.is_alive else 0.0)
                rewards_per_agent[aid].append(reward)
                if prev_ag:
                    energy_loss = max(0.0, prev_ag.energy - curr_ag.energy)
                    total_energy_expended += energy_loss
                    if food_diff > 0:
                        total_resources_collected += food_diff

        # Stop early if all agents are dead
        all_dead = all(not ag.is_alive for ag in new_state.agents.values())
        if all_dead and initial_agent_count > 0:
            break

    elapsed = max(1e-6, time.perf_counter() - start_time)

    # 4. Compute metrics
    final_state = world.state
    surviving_count = sum(1 for ag in final_state.agents.values() if ag.is_alive)
    survival_rate = SurvivalRateComputer.compute(initial_agent_count, surviving_count)

    # Cumulative rewards
    all_agent_returns = [sum(r_list) for r_list in rewards_per_agent.values()]
    avg_return = float(sum(all_agent_returns) / len(all_agent_returns)) if all_agent_returns else 0.0

    # Path efficiency for first surviving/active agent
    pe_val = 1.0
    if config.agents:
        first_id = config.agents[0].agent_id
        if first_id in start_positions and first_id in final_state.agents:
            st = start_positions[first_id]
            fn = final_state.agents[first_id].position
            manhattan_dist = abs(st.x - fn.x) + abs(st.y - fn.y)
            pe_val = PathEfficiencyComputer.compute(
                actual_path_length=step_count,
                optimal_length=manhattan_dist,
            )

    re_val = ResourceEfficiencyComputer.compute(total_resources_collected, total_energy_expended)
    growth_val = PopulationGrowthComputer.compute(initial_agent_count, surviving_count)
    throughput = SimulationThroughputComputer.compute(step_count, elapsed)

    metrics: dict[str, float] = {
        "survival_rate": survival_rate,
        "average_return": avg_return,
        "path_efficiency": pe_val,
        "resource_efficiency": re_val,
        "population_growth_rate": growth_val,
        "throughput_ticks_per_sec": throughput,
        "total_ticks": float(step_count),
    }

    run_record = RunRecord(
        run_id=f"run_seed_{seed}",
        config_hash=config_hash,
        seed=seed,
        git_commit=git_commit,
        software_versions=capture_environment_versions(),
        metrics=metrics,
        timing_seconds=elapsed,
        ticks_per_second=throughput,
        timestamp=datetime.now(UTC).isoformat(),
    )
    return run_record.model_dump()


class ExperimentRunner:
    """Executes full experiment suites across deterministic seeds with parallel worker dispatch."""

    def __init__(self, use_parallel: bool = True) -> None:
        self.use_parallel = use_parallel

    def run(self, config: ExperimentConfig) -> ExperimentSuiteRecord:
        """Executes the experiment defined in config and returns complete reproducibility record."""
        config_hash = compute_config_hash(config)
        git_commit = get_git_commit()
        suite_id = f"suite_{config.name}_{config.seed}_{int(time.time())}"

        # Determine seed schedule
        if config.protocol.eval_seed_range:
            start_s, end_s = config.protocol.eval_seed_range
            available_seeds = list(range(start_s, end_s + 1))
            seeds = available_seeds[: config.episodes]
        else:
            splitter = SeedSplitter(root_seed=config.seed)
            seeds = splitter.spawn_seeds(config.episodes)

        payloads = [
            (config.model_dump(), seed, config_hash, git_commit)
            for seed in seeds
        ]

        run_records: list[RunRecord] = []

        # Execute runs
        num_workers = min(config.parallel_workers, len(seeds))
        if self.use_parallel and num_workers > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    raw_records = list(executor.map(_execute_single_run, payloads))
                    run_records = [RunRecord.model_validate(r) for r in raw_records]
            except Exception:
                # Fallback to sequential execution if ProcessPool encounters environment limitations
                raw_records = [_execute_single_run(p) for p in payloads]
                run_records = [RunRecord.model_validate(r) for r in raw_records]
        else:
            raw_records = [_execute_single_run(p) for p in payloads]
            run_records = [RunRecord.model_validate(r) for r in raw_records]

        # Aggregate metrics
        aggregated: dict[str, SampleStats] = {}
        metric_keys = list(run_records[0].metrics.keys()) if run_records else []
        for m_key in metric_keys:
            vals = [r.metrics[m_key] for r in run_records]
            aggregated[m_key] = sample_stats(vals, confidence=config.protocol.confidence_level)

        limitations: list[str] = []
        if len(run_records) < 30:
            limitations.append(
                f"Small sample size (N={len(run_records)}). Asymptotic normal approximations carry higher uncertainty."
            )

        record = ExperimentSuiteRecord(
            suite_id=suite_id,
            experiment_name=config.name,
            config_hash=config_hash,
            total_runs=len(run_records),
            git_commit=git_commit,
            software_versions=capture_environment_versions(),
            aggregated_metrics=aggregated,
            comparisons=[],
            ablation_summary=[],
            records=run_records,
            limitations=limitations,
            created_at=datetime.now(UTC).isoformat(),
        )
        return record
