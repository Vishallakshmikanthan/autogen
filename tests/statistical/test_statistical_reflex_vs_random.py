"""Statistical hypothesis test: Assert SimpleReflexAgent dominates RandomAgent across >= 30 seeds."""

import numpy as np
from scipy import stats

from autogen.agents.base.contracts import BaseAgent
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.evaluation.performance import PerformanceEvaluator
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def _run_agent_eval(agent_type: str, seed: int, steps: int = 50) -> float:
    cfg = EnvironmentConfig(
        width=15,
        height=15,
        seed=seed,
        energy_max=100.0,
    )
    world = GridWorld(config=cfg)
    world.reset(seed=seed)

    agent_id = "test_agent"
    world.spawn_agent(agent_id=agent_id, energy=100.0)

    agent: BaseAgent
    if agent_type == "random":
        agent = RandomAgent(agent_id=agent_id, seed=seed)
    else:
        agent = SimpleReflexAgent(agent_id=agent_id)

    engine = SimulationEngine(environment=world, seed=seed)
    engine.register_agent(agent)

    trajectory = [world.state]
    rewards: dict[str, list[float]] = {agent_id: []}

    for _ in range(steps):
        state = engine.step()
        trajectory.append(state)
        # Reward function: food collected * 10.0 + alive_step * 0.1
        food_gain = (
            10.0
            if state.agents[agent_id].carried_food > trajectory[-2].agents[agent_id].carried_food
            else 0.0
        )
        r = food_gain + 0.1
        rewards[agent_id].append(r)
        if not state.agents[agent_id].is_alive:
            break

    evaluator = PerformanceEvaluator(gamma=0.99)
    perf = evaluator.evaluate_trajectories(trajectory, rewards)
    return perf.per_agent[agent_id].discounted_return


def test_statistical_reflex_beats_random_over_30_seeds():
    """Hypothesis test across 30 seeds asserting Reflex agent significantly outperforms Random."""
    num_seeds = 30
    seeds = [3000 + i for i in range(num_seeds)]

    random_scores: list[float] = []
    reflex_scores: list[float] = []

    for s in seeds:
        rand_ret = _run_agent_eval("random", s)
        refl_ret = _run_agent_eval("reflex", s)
        random_scores.append(rand_ret)
        reflex_scores.append(refl_ret)

    mean_rand = float(np.mean(random_scores))
    mean_refl = float(np.mean(reflex_scores))

    # Perform paired t-test
    t_stat, p_val = stats.ttest_rel(reflex_scores, random_scores)

    # SimpleReflexAgent must achieve significantly higher discounted return with p < 0.01
    assert mean_refl > mean_rand, f"Reflex mean {mean_refl:.2f} <= Random mean {mean_rand:.2f}"
    assert p_val < 0.01, f"Expected p < 0.01, got p = {p_val:.4e} (t = {t_stat:.2f})"
