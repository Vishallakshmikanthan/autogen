"""Integration test: MemoryAugmentedAgents in deterministic simulation."""

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.memory_augmented_agent import MemoryAugmentedAgent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.world_state import canonical_world_hash


def _run_memory_sim(seed: int, steps: int = 30) -> list[str]:
    cfg = EnvironmentConfig(width=20, height=20, seed=seed, energy_max=100.0)
    world = GridWorld(config=cfg)
    world.reset(seed=seed)

    world.spawn_agent("mem_agent")
    world.spawn_agent("baseline_agent")

    inner_agent = SimpleReflexAgent("mem_agent")
    mem_agent = MemoryAugmentedAgent(inner_agent=inner_agent)
    rand_agent = RandomAgent("baseline_agent", seed=seed)

    engine = SimulationEngine(environment=world, seed=seed)
    engine.register_agent(mem_agent)
    engine.register_agent(rand_agent)

    hashes: list[str] = []
    for _ in range(steps):
        state = engine.step()
        hashes.append(canonical_world_hash(state))

    # Assert memory explorer returns populated snapshot
    snapshot = mem_agent.explorer.export_snapshot("mem_agent", current_step=steps)
    assert snapshot.spatial_cells_count > 0
    assert snapshot.working_memory_size > 0

    return hashes


def test_memory_simulation_determinism():
    """Verify identical hash replay for memory-augmented simulation."""
    seed = 88888
    hashes_1 = _run_memory_sim(seed)
    hashes_2 = _run_memory_sim(seed)

    assert hashes_1 == hashes_2, "Memory simulation exhibited nondeterminism!"
