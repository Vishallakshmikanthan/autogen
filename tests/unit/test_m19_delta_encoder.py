"""Unit tests for ViewerState delta-encoding and reconstruction exactness (M19, SYS-18)."""

from apps.api.worker.delta_encoder import DeltaEncoder
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.visualization.adapter import ViewerDataAdapter


def test_delta_encoding_roundtrip_exactness():
    # Set up world
    cfg = EnvironmentConfig(width=10, height=10, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)
    world.spawn_agent(agent_id="test_agent", energy=100.0)

    engine = SimulationEngine(environment=world, seed=42)
    adapter = ViewerDataAdapter(engine)

    state_0 = adapter.generate_viewer_state()

    # Step engine
    engine.step()
    state_1 = adapter.generate_viewer_state()

    # 1. Delta encode transition from state_0 to state_1
    delta = DeltaEncoder.encode(state_0, state_1, is_keyframe=False)
    assert delta["is_keyframe"] is False
    assert delta["step"] == 1

    # 2. Reconstruct state_1 from state_0 + delta
    reconstructed_state_1 = DeltaEncoder.decode_and_apply(state_0, delta)

    # Assert bit-identical reconstruction
    assert reconstructed_state_1.step == state_1.step
    assert reconstructed_state_1.state_hash == state_1.state_hash
    assert len(reconstructed_state_1.agents) == len(state_1.agents)
    assert reconstructed_state_1.agents[0].x == state_1.agents[0].x
    assert reconstructed_state_1.agents[0].y == state_1.agents[0].y
    assert reconstructed_state_1.agents[0].energy == state_1.agents[0].energy


def test_keyframe_encoding():
    cfg = EnvironmentConfig(width=6, height=6, seed=99)
    world = GridWorld(config=cfg)
    world.reset(seed=99)
    engine = SimulationEngine(environment=world, seed=99)
    adapter = ViewerDataAdapter(engine)
    state = adapter.generate_viewer_state()

    keyframe = DeltaEncoder.encode(None, state, is_keyframe=True)
    assert keyframe["is_keyframe"] is True
    assert "full_state" in keyframe

    reconstructed = DeltaEncoder.decode_and_apply(None, keyframe)
    assert reconstructed.state_hash == state.state_hash
