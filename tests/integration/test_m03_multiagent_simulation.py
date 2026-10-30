"""Integration test: 20 mixed agents on a 50x50 grid world running deterministically."""

from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.evaluation.performance import PerformanceEvaluator
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig, PerceptionMode
from autogen.simulation.state.world_state import canonical_world_hash


def _run_multiagent_sim(seed: int, steps: int = 50) -> tuple[list[str], list[float]]:
    width, height = 50, 50
    cfg = EnvironmentConfig(
        width=width,
        height=height,
        seed=seed,
        energy_max=150.0,
    )
    env = GridWorld(config=cfg)
    env.reset(seed=seed)

    agent_types = ["random", "simple_reflex", "model_based_reflex"]
    engine = SimulationEngine(environment=env, seed=seed)

    # Spawn 20 mixed agents via AgentRegistry
    agents = []
    for i in range(20):
        aid = f"agent_{i:02d}"
        atype = agent_types[i % len(agent_types)]
        env.spawn_agent(aid)
        p_cfg = PerceptionConfig(
            mode=PerceptionMode.LOCAL_WINDOW if i % 2 == 0 else PerceptionMode.LINE_OF_SIGHT,
            radius=3,
        )
        agent = AgentRegistry.create(
            AgentConfig(agent_type=atype, agent_id=aid, seed=seed + i, perception=p_cfg)
        )
        engine.register_agent(agent, perception_config=p_cfg)
        agents.append(aid)

    state_hashes: list[str] = []
    trajectory = [env.state]
    per_tick_rewards: dict[str, list[float]] = {aid: [] for aid in agents}

    for _ in range(steps):
        state = engine.step()
        trajectory.append(state)
        state_hashes.append(canonical_world_hash(state))
        for aid in agents:
            ag = state.agents.get(aid)
            r = 1.0 if (ag and ag.is_alive) else 0.0
            per_tick_rewards[aid].append(r)

    evaluator = PerformanceEvaluator(gamma=0.95)
    pop_perf = evaluator.evaluate_trajectories(trajectory, per_tick_rewards)

    assert pop_perf.agent_count == 20
    assert len(state_hashes) == steps
    return state_hashes, [pop_perf.discounted_return.mean, pop_perf.exploration_coverage.mean]


def test_20_mixed_agents_50x50_determinism():
    """Confirms determinism and throughput with 20 mixed agents on a 50x50 world."""
    seed = 424242
    steps = 40

    hashes_run1, metrics_1 = _run_multiagent_sim(seed, steps=steps)
    hashes_run2, metrics_2 = _run_multiagent_sim(seed, steps=steps)

    # 1. Bit-identical hashes for every single tick
    assert hashes_run1 == hashes_run2, "Multi-agent simulation exhibited nondeterminism!"

    # 2. Metrics exactly match
    assert metrics_1 == metrics_2
