"""Unit tests for MemoryAugmentedAgent wrapper and decision trace attribution."""

from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.memory_augmented_agent import MemoryAugmentedAgent
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.engine import PerceptionEngine
from autogen.simulation.state.types import Position, TerrainType


def test_memory_augmented_agent_decoration_and_trace():
    base = SimpleReflexAgent("agent_1")
    agent = MemoryAugmentedAgent(inner_agent=base)

    assert agent.agent_id == "agent_1"
    assert agent.capabilities.memory is True

    world = GridWorld(config=EnvironmentConfig(width=10, height=10, seed=42))
    world.reset(seed=42)
    grid = [list(r) for r in world.state.grid]
    grid[5][5] = grid[5][5].model_copy(update={"terrain": TerrainType.GRASSLAND, "food": 5.0})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("agent_1", Position(5, 5))

    engine = PerceptionEngine()
    obs = engine.generate_observation(world.state, "agent_1")

    agent.observe(obs)
    assert len(agent.working_memory) == 1
    assert len(agent.spatial_memory) > 0

    agent.update_belief()
    action, trace = agent.decide()

    assert action.agent_id == "agent_1"
    assert "memory_used" in trace.rationale
    assert "working_memory_len" in trace.rationale
    assert "spatial_memory_len" in trace.rationale

    # Calling learn() commits transition to episodic memory
    agent.learn(reward=5.0, terminal=False)
    assert len(agent.episodic_memory) == 1
    latest_trans = agent.episodic_memory.get_latest()
    assert latest_trans is not None
    assert latest_trans.reward == 5.0


def test_memory_augmented_agent_factory_instantiation():
    cfg = AgentConfig(
        agent_type="memory_augmented",
        agent_id="mem_1",
        custom_params={
            "inner_agent_type": "simple_reflex",
            "memory_config": {"working_memory_capacity": 15},
        },
    )

    agent = AgentRegistry.create(cfg)
    assert isinstance(agent, MemoryAugmentedAgent)
    assert agent.agent_id == "mem_1"
    assert agent.capabilities.memory is True
    assert agent.working_memory.capacity == 15
