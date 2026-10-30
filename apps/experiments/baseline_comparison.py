"""M03 Baseline Agent Evaluation and Empirical Experiments runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from autogen.agents.base.contracts import BaseAgent
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.evaluation.performance import PerformanceEvaluator
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig, PerceptionMode


def run_single_simulation(
    agent_type: str,
    seed: int,
    steps: int = 100,
    width: int = 20,
    height: int = 20,
    food_scale: float = 1.0,
    perception_config: PerceptionConfig | None = None,
) -> dict[str, float]:
    """Runs a single controlled simulation for an agent type and returns performance metrics."""
    config = EnvironmentConfig(
        width=width,
        height=height,
        seed=seed,
        energy_max=100.0,
    )
    env = GridWorld(config=config)
    env.reset(seed=seed)

    if food_scale != 1.0:
        grid_copy = [list(r) for r in env.state.grid]
        for y in range(height):
            for x in range(width):
                c = grid_copy[y][x]
                if c.food > 0.0:
                    grid_copy[y][x] = c.model_copy(update={"food": round(c.food * food_scale, 2)})
        env._current_state = env.state.model_copy(
            update={"grid": tuple(tuple(r) for r in grid_copy)}
        )

    agent_id = "agent_0"
    env.spawn_agent(agent_id=agent_id, energy=100.0)

    agent: BaseAgent
    if agent_type == "random":
        agent = RandomAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "reflex":
        agent = SimpleReflexAgent(agent_id=agent_id)
    else:
        raise ValueError(f"Unknown agent type {agent_type}")

    p_configs = {agent_id: perception_config} if perception_config else None
    engine = SimulationEngine(
        environment=env,
        seed=seed,
        perception_configs=p_configs,
    )
    engine.register_agent(agent, perception_config=perception_config)

    trajectory = [env.state]
    per_tick_rewards: dict[str, list[float]] = {agent_id: []}

    for _ in range(steps):
        state = engine.step()
        trajectory.append(state)
        # Reward is food collected + small survival step reward
        collected = (
            10.0
            if state.agents[agent_id].carried_food > trajectory[-2].agents[agent_id].carried_food
            else 0.0
        )
        r = collected + (0.1 if state.agents[agent_id].is_alive else -10.0)
        per_tick_rewards[agent_id].append(r)
        if not state.agents[agent_id].is_alive:
            break

    evaluator = PerformanceEvaluator(gamma=0.99)
    pop_perf = evaluator.evaluate_trajectories(trajectory, per_tick_rewards)
    agent_m = pop_perf.per_agent[agent_id]

    return {
        "discounted_return": agent_m.discounted_return,
        "resources_collected": agent_m.resources_collected,
        "survival_time": agent_m.survival_time,
        "energy_efficiency": agent_m.energy_efficiency,
        "exploration_coverage": agent_m.exploration_coverage,
    }


def run_all_experiments(num_seeds: int = 30, output_dir: Path | None = None) -> dict:
    """Executes the complete M03 experiment suite and serializes results to disk."""
    out_dir = output_dir or Path("results/experiments/m03")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [1000 + i for i in range(num_seeds)]

    print(f"=== M03 Baseline Experiments (seeds={num_seeds}) ===")

    # 1. Random vs Reflex
    print("1. Running Random vs SimpleReflex benchmark...")
    random_returns: list[float] = []
    reflex_returns: list[float] = []
    random_resources: list[float] = []
    reflex_resources: list[float] = []

    for s in seeds:
        m_rand = run_single_simulation("random", seed=s)
        m_refl = run_single_simulation("reflex", seed=s)
        random_returns.append(m_rand["discounted_return"])
        reflex_returns.append(m_refl["discounted_return"])
        random_resources.append(m_rand["resources_collected"])
        reflex_resources.append(m_refl["resources_collected"])

    # Statistical hypothesis test (paired t-test)
    t_stat, p_val = stats.ttest_rel(reflex_returns, random_returns)
    print(f"   Reflex Mean Return: {np.mean(reflex_returns):.2f} +/- {np.std(reflex_returns):.2f}")
    print(f"   Random Mean Return: {np.mean(random_returns):.2f} +/- {np.std(random_returns):.2f}")
    print(f"   Paired t-test t={t_stat:.4f}, p-value={p_val:.4e}")

    # 2. Full vs Partial Observability
    print("2. Running Full vs Partial Observability sweep...")
    obs_results = {}
    for mode in [PerceptionMode.FULL, PerceptionMode.LOCAL_WINDOW]:
        cfg = PerceptionConfig(mode=mode, radius=3)
        returns = [
            run_single_simulation("reflex", seed=s, perception_config=cfg)["discounted_return"]
            for s in seeds[:15]
        ]
        obs_results[mode.value] = {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
        }

    # 3. Sensor Radius Sweep
    print("3. Running Sensor Radius sweep...")
    radius_results = {}
    for r in [1, 2, 3, 5]:
        cfg = PerceptionConfig(mode=PerceptionMode.LOCAL_WINDOW, radius=r)
        returns = [
            run_single_simulation("reflex", seed=s, perception_config=cfg)["discounted_return"]
            for s in seeds[:15]
        ]
        radius_results[f"r_{r}"] = {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
        }

    # 4. Resource Density Sweep
    print("4. Running Resource Density sweep...")
    density_results = {}
    for scale in [0.25, 0.75, 1.5]:
        returns = [
            run_single_simulation("reflex", seed=s, food_scale=scale)["discounted_return"]
            for s in seeds[:15]
        ]
        density_results[f"scale_{scale}"] = {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
        }

    payload = {
        "random_vs_reflex": {
            "num_seeds": num_seeds,
            "random": {
                "mean_return": float(np.mean(random_returns)),
                "std_return": float(np.std(random_returns)),
                "mean_resources": float(np.mean(random_resources)),
            },
            "reflex": {
                "mean_return": float(np.mean(reflex_returns)),
                "std_return": float(np.std(reflex_returns)),
                "mean_resources": float(np.mean(reflex_resources)),
            },
            "statistical_test": {
                "test": "paired_t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant_at_0_01": bool(p_val < 0.01 and t_stat > 0),
            },
        },
        "full_vs_partial_observability": obs_results,
        "sensor_radius_sweep": radius_results,
        "resource_density_sweep": density_results,
    }

    out_file = out_dir / "baseline_comparison_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Results successfully written to {out_file.resolve()}")
    return payload


if __name__ == "__main__":
    run_all_experiments(num_seeds=30)
