"""M04 Memory Engine empirical benchmarks and ablation experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from autogen.agents.base.contracts import BaseAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.memory_augmented_agent import MemoryAugmentedAgent
from autogen.evaluation.performance import PerformanceEvaluator
from autogen.memory.contracts import MemoryConfig
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def run_single_eval(
    use_memory: bool,
    seed: int,
    memory_config: MemoryConfig | None = None,
    steps: int = 80,
    width: int = 25,
    height: int = 25,
) -> dict[str, float]:
    """Runs a single controlled simulation evaluating agent resource collection."""
    cfg = EnvironmentConfig(
        width=width,
        height=height,
        seed=seed,
        energy_max=120.0,
    )
    env = GridWorld(config=cfg)
    env.reset(seed=seed)

    agent_id = "agent_mem"
    env.spawn_agent(agent_id=agent_id, energy=120.0)

    agent: BaseAgent
    base_agent = SimpleReflexAgent(agent_id=agent_id)
    if use_memory:
        agent = MemoryAugmentedAgent(inner_agent=base_agent, config=memory_config)
    else:
        agent = base_agent

    engine = SimulationEngine(environment=env, seed=seed)
    engine.register_agent(agent)

    trajectory = [env.state]
    per_tick_rewards: dict[str, list[float]] = {agent_id: []}

    for _ in range(steps):
        state = engine.step()
        trajectory.append(state)
        food_gain = (
            10.0
            if state.agents[agent_id].carried_food > trajectory[-2].agents[agent_id].carried_food
            else 0.0
        )
        r = food_gain + 0.1
        per_tick_rewards[agent_id].append(r)
        if not state.agents[agent_id].is_alive:
            break

    evaluator = PerformanceEvaluator(gamma=0.99)
    pop_perf = evaluator.evaluate_trajectories(trajectory, per_tick_rewards)
    m = pop_perf.per_agent[agent_id]

    return {
        "discounted_return": m.discounted_return,
        "resources_collected": m.resources_collected,
        "survival_time": float(m.survival_time),
        "energy_efficiency": m.energy_efficiency,
        "exploration_coverage": m.exploration_coverage,
    }


def run_memory_benchmarks(num_seeds: int = 20, output_dir: Path | None = None) -> dict:
    """Executes memory vs no-memory comparisons, capacity sweeps, and decay rate sweeps."""
    out_dir = output_dir or Path("results/experiments/m04")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [4000 + i for i in range(num_seeds)]

    print(f"=== M04 Memory Engine Experiments (seeds={num_seeds}) ===")

    # 1. Memory vs No-Memory baseline comparison
    print("1. Running Memory vs No-Memory benchmark...")
    no_mem_returns: list[float] = []
    mem_returns: list[float] = []
    no_mem_resources: list[float] = []
    mem_resources: list[float] = []

    for s in seeds:
        res_no = run_single_eval(use_memory=False, seed=s)
        res_mem = run_single_eval(use_memory=True, seed=s)
        no_mem_returns.append(res_no["discounted_return"])
        mem_returns.append(res_mem["discounted_return"])
        no_mem_resources.append(res_no["resources_collected"])
        mem_resources.append(res_mem["resources_collected"])

    # Statistical evaluation
    diff = np.array(mem_returns) - np.array(no_mem_returns)
    std_diff = float(np.std(diff))
    if std_diff > 0:
        t_res = stats.ttest_rel(mem_returns, no_mem_returns)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
        cohen_d = float(np.mean(diff) / std_diff)
        ci_tuple = stats.t.interval(
            confidence=0.95,
            df=len(diff) - 1,
            loc=float(np.mean(diff)),
            scale=float(stats.sem(diff)),
        )
        ci_low, ci_high = float(ci_tuple[0]), float(ci_tuple[1])
    else:
        t_stat, p_val = 0.0, 1.0
        cohen_d = 0.0
        ci_low, ci_high = float(np.mean(diff)), float(np.mean(diff))

    print(f"   Memory Augmented Mean: {np.mean(mem_returns):.2f} +/- {np.std(mem_returns):.2f}")
    print(
        f"   No Memory Baseline Mean: {np.mean(no_mem_returns):.2f} +/- {np.std(no_mem_returns):.2f}"
    )
    print(f"   Mean Difference: {np.mean(diff):.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    print(f"   Cohen's d: {cohen_d:.2f}, paired t p-value: {p_val:.4e}")

    # 2. Capacity Sweep
    print("2. Running Spatial Memory Capacity sweep...")
    capacity_results = {}
    for cap in [5, 20, 50, 200]:
        cfg = MemoryConfig(spatial_memory_capacity=cap)
        rets = [
            run_single_eval(use_memory=True, seed=s, memory_config=cfg)["discounted_return"]
            for s in seeds[:10]
        ]
        capacity_results[f"cap_{cap}"] = {
            "mean": float(np.mean(rets)),
            "std": float(np.std(rets)),
        }

    # 3. Forgetting / Decay Rate Sweep
    print("3. Running Decay Rate sweep...")
    decay_results = {}
    for rate in [0.0, 0.01, 0.05, 0.1]:
        cfg = MemoryConfig(decay_rate=rate)
        rets = [
            run_single_eval(use_memory=True, seed=s, memory_config=cfg)["discounted_return"]
            for s in seeds[:10]
        ]
        decay_results[f"decay_{rate}"] = {
            "mean": float(np.mean(rets)),
            "std": float(np.std(rets)),
        }

    payload = {
        "memory_vs_no_memory": {
            "num_seeds": num_seeds,
            "no_memory": {
                "mean_return": float(np.mean(no_mem_returns)),
                "std_return": float(np.std(no_mem_returns)),
                "mean_resources": float(np.mean(no_mem_resources)),
            },
            "memory_augmented": {
                "mean_return": float(np.mean(mem_returns)),
                "std_return": float(np.std(mem_returns)),
                "mean_resources": float(np.mean(mem_resources)),
            },
            "effect_size": {
                "mean_diff": float(np.mean(diff)),
                "ci_95": [ci_low, ci_high],
                "cohens_d": cohen_d,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            },
        },
        "capacity_sweep": capacity_results,
        "decay_rate_sweep": decay_results,
    }

    out_file = out_dir / "memory_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Memory benchmark results written to {out_file.resolve()}")
    return payload


if __name__ == "__main__":
    run_memory_benchmarks(num_seeds=20)
