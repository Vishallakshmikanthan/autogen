"""Unit tests for ViewerDataAdapter and ViewerState DTO serialization (Module M05)."""

from __future__ import annotations

from autogen.agents import (
    MemoryAugmentedAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
)
from autogen.memory.contracts import MemoryConfig
from autogen.simulation.core.contracts import Action, InvalidActionEvent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType
from autogen.visualization.adapter import ViewerDataAdapter
from autogen.visualization.contracts import ViewerState, ViewerTimeMachineState


def _setup_test_environment(width: int = 10, height: int = 10, seed: int = 42):
    config = EnvironmentConfig(width=width, height=height, seed=seed)
    world = GridWorld(config)
    world.reset(seed=seed)
    engine = SimulationEngine(world)

    # Agent 0: Memory-augmented agent
    world.spawn_agent("agent_00", energy=100.0)
    base = SimpleReflexAgent("agent_00")
    mem_agent = MemoryAugmentedAgent(
        base,
        config=MemoryConfig(spatial_memory_capacity=50, episodic_memory_capacity=50),
    )
    engine.register_agent(mem_agent)

    # Agent 1: Model-based reflex agent
    world.spawn_agent("agent_01", energy=100.0)
    mb_agent = ModelBasedReflexAgent("agent_01")
    engine.register_agent(mb_agent)

    adapter = ViewerDataAdapter(engine)
    return engine, adapter


def test_viewer_state_generation():
    """Verifies complete, valid ViewerState generation from live simulation."""
    engine, adapter = _setup_test_environment()

    # Step engine once so agents perceive and deliberate
    engine.step()
    adapter.update_trails()

    # Record trace if available
    agent_00 = engine.agents["agent_00"]
    if hasattr(agent_00, "latest_trace") and agent_00.latest_trace:
        adapter.record_decision_trace(agent_00.latest_trace)

    tm_state = ViewerTimeMachineState(
        current_step=1,
        total_steps=1,
        available_snapshots=[0],
        is_paused=True,
        playback_speed=1.0,
        can_fork=True,
    )

    state = adapter.generate_viewer_state(
        selected_agent_id="agent_00",
        time_machine_state=tm_state,
    )

    assert isinstance(state, ViewerState)
    assert state.step == 1
    assert state.width == 10
    assert state.height == 10
    assert len(state.grid_cells) == 100
    assert len(state.agents) == 2
    assert state.selected_agent_id == "agent_00"
    assert state.time_machine is not None
    assert state.time_machine.current_step == 1

    # Verify agent mind was populated for agent_00
    assert state.agent_mind is not None
    assert len(state.agent_mind) > 0


def test_viewer_state_json_roundtrip():
    """Verifies lossless JSON serialization/deserialization for M20 React Studio consumption."""
    engine, adapter = _setup_test_environment()
    engine.step()
    adapter.update_trails()

    state = adapter.generate_viewer_state(selected_agent_id="agent_01")

    # Serialize to JSON string
    json_payload = state.model_dump_json()
    assert isinstance(json_payload, str)

    # Reconstruct from JSON
    restored = ViewerState.model_validate_json(json_payload)

    assert restored.step == state.step
    assert restored.width == state.width
    assert restored.height == state.height
    assert restored.state_hash == state.state_hash
    assert len(restored.grid_cells) == len(state.grid_cells)
    assert len(restored.agents) == len(state.agents)
    assert restored.selected_agent_id == state.selected_agent_id
    assert len(restored.overlays.fog_of_war_visible) == len(state.overlays.fog_of_war_visible)


def test_viewer_overlays_and_collision_telemetry():
    """Verifies trail tracking and collision marker events."""
    engine, adapter = _setup_test_environment()

    # Step multiple times to build trails
    for _ in range(5):
        engine.step()
        adapter.update_trails()

    state = adapter.generate_viewer_state(selected_agent_id="agent_00")
    assert "agent_00" in state.overlays.movement_trails
    assert len(state.overlays.movement_trails["agent_00"]) == 5

    # Emit invalid action collision event
    pos = engine.environment.state.agents["agent_00"].position
    attempted = Action(agent_id="agent_00", action_type=ActionType.UP)
    substituted = Action(agent_id="agent_00", action_type=ActionType.STAY)
    adapter._on_invalid_action(
        InvalidActionEvent(
            agent_id="agent_00",
            attempted_action=attempted,
            substituted_action=substituted,
            reason="Blocked by obstacle",
            step=5,
        )
    )

    state_with_collision = adapter.generate_viewer_state(selected_agent_id="agent_00")
    assert (pos.x, pos.y) in state_with_collision.overlays.collision_markers
