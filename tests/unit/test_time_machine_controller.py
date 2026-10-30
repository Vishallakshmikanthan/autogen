"""Unit tests for TimeMachineController: scrubbing, snapshots, and replay navigation (Module M05)."""

from __future__ import annotations

import pytest

from autogen.agents.baseline.random_agent import RandomAgent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.world_state import canonical_world_hash
from autogen.visualization.time_machine import TimeMachineController


def _setup_time_machine(width: int = 8, height: int = 8, seed: int = 123):
    config = EnvironmentConfig(width=width, height=height, seed=seed)
    world = GridWorld(config)
    world.reset(seed=seed)
    world.spawn_agent("agent_0")

    engine = SimulationEngine(world, seed=seed)
    engine.register_agent(RandomAgent("agent_0", seed=seed))

    controller = TimeMachineController(engine, snapshot_cadence=5)
    return engine, controller


def test_time_machine_initialization_and_controls():
    """Verifies default time machine state, play/pause toggles, and playback speeds."""
    _, controller = _setup_time_machine()

    state = controller.get_state()
    assert state.current_step == 0
    assert state.total_steps == 0
    assert state.available_snapshots == [0]
    assert state.is_paused is True
    assert state.playback_speed == 1.0

    # Controls
    controller.play()
    assert controller.is_paused is False
    controller.pause()
    assert controller.is_paused is True
    controller.toggle_play_pause()
    assert controller.is_paused is False

    controller.set_speed(2.5)
    assert controller.playback_speed == 2.5

    with pytest.raises(ValueError, match="Playback speed must be > 0"):
        controller.set_speed(0.0)


def test_snapshot_cadence_and_scrubbing():
    """Verifies periodic snapshot indexing and exact historical restoration via scrubbing."""
    engine, controller = _setup_time_machine()

    # Step forward 12 ticks
    hashes: dict[int, str] = {0: canonical_world_hash(engine.environment.state)}
    for _ in range(12):
        step = controller.step_forward()
        hashes[step] = canonical_world_hash(engine.environment.state)

    state = controller.get_state()
    assert state.current_step == 12
    assert state.total_steps == 12
    # Snapshots should be recorded at 0, 5, 10
    assert sorted(state.available_snapshots) == [0, 5, 10]

    # Scrub backwards to exact snapshot keyframe (step 5)
    step = controller.scrub_to(5)
    assert step == 5
    assert canonical_world_hash(engine.environment.state) == hashes[5]

    # Scrub backwards to intermediate step between keyframes (step 7)
    step = controller.scrub_to(7)
    assert step == 7
    assert canonical_world_hash(engine.environment.state) == hashes[7]

    # Single-step backward
    step = controller.step_backward()
    assert step == 6
    assert canonical_world_hash(engine.environment.state) == hashes[6]


def test_forking_branch_independence():
    """Verifies counterfactual forking produces an independent, valid time machine branch."""
    engine, controller = _setup_time_machine()

    for _ in range(15):
        controller.step_forward()

    # Fork at tick 10
    forked_controller = controller.fork(target_step=10)
    assert forked_controller.engine.environment.state.step == 10
    # Original should be at step 10 after scrub
    assert controller.engine.environment.state.step == 10

    # Advance original to 15 again
    controller.scrub_to(15)
    assert controller.engine.environment.state.step == 15

    # Forked engine should remain at 10 until stepped
    assert forked_controller.engine.environment.state.step == 10
    forked_controller.step_forward()
    assert forked_controller.engine.environment.state.step == 11
