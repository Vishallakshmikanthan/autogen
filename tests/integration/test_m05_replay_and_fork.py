"""Integration tests for M05 visualization, exact replay reproducibility, and counterfactual forking."""

from __future__ import annotations

import pytest

from autogen.agents.baseline.model_based_reflex_agent import ModelBasedReflexAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.simulation.core.contracts import Action
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType
from autogen.simulation.state.world_state import canonical_world_hash
from autogen.visualization.adapter import ViewerDataAdapter
from autogen.visualization.pygame_viewer import PygameViewer
from autogen.visualization.time_machine import TimeMachineController


@pytest.fixture
def multiagent_setup():
    """Initializes a deterministic 2-agent simulation with adapter and time machine."""
    config = EnvironmentConfig(width=12, height=12, seed=999)
    world = GridWorld(config)
    world.reset(seed=999)

    world.spawn_agent("agent_alpha", energy=100.0)
    world.spawn_agent("agent_beta", energy=100.0)

    engine = SimulationEngine(world, seed=999)
    engine.register_agent(SimpleReflexAgent("agent_alpha"))
    engine.register_agent(ModelBasedReflexAgent("agent_beta"))

    adapter = ViewerDataAdapter(engine)
    time_machine = TimeMachineController(engine, snapshot_cadence=5)
    return engine, adapter, time_machine


@pytest.mark.integration
def test_m05_full_replay_hash_equality(multiagent_setup):
    """Invariant: Scrubbing backwards and stepping forward produces bit-identical world hashes."""
    engine, adapter, time_machine = multiagent_setup

    original_hashes: list[str] = [canonical_world_hash(engine.environment.state)]

    # Run for 25 ticks
    for _ in range(25):
        time_machine.step_forward()
        original_hashes.append(canonical_world_hash(engine.environment.state))

    assert len(original_hashes) == 26

    # Scrub back to tick 0
    time_machine.scrub_to(0)
    assert canonical_world_hash(engine.environment.state) == original_hashes[0]

    # Replay forward tick-by-tick and verify exact hash match
    replay_hashes: list[str] = [canonical_world_hash(engine.environment.state)]
    for _ in range(25):
        time_machine.step_forward()
        replay_hashes.append(canonical_world_hash(engine.environment.state))

    assert replay_hashes == original_hashes


@pytest.mark.integration
def test_m05_fork_divergence_boundary(multiagent_setup):
    """Invariant: Forking without overrides produces identical history; overridden fork diverges."""
    engine, adapter, time_machine = multiagent_setup

    original_hashes: list[str] = [canonical_world_hash(engine.environment.state)]
    for _ in range(20):
        time_machine.step_forward()
        original_hashes.append(canonical_world_hash(engine.environment.state))

    fork_tick = 10

    # 1. Fork without overrides: state at fork_tick matches original run exactly
    unmodified_fork = time_machine.fork(target_step=fork_tick)
    assert unmodified_fork.engine.environment.state.step == fork_tick
    assert (
        canonical_world_hash(unmodified_fork.engine.environment.state) == original_hashes[fork_tick]
    )

    # Stepping the unmodified fork matches original history step-for-step
    unmod_next = unmodified_fork.step_forward()
    assert (
        canonical_world_hash(unmodified_fork.engine.environment.state)
        == original_hashes[unmod_next]
    )

    # 2. Fork with an overridden action forcing agent_alpha to move RIGHT
    override_action = Action(
        agent_id="agent_alpha",
        action_type=ActionType.RIGHT,
    )

    forked_tm = time_machine.fork(
        target_step=fork_tick,
        overridden_actions={"agent_alpha": override_action},
    )

    # Overridden fork executes the action override immediately, advancing to fork_tick + 1
    assert forked_tm.engine.environment.state.step == fork_tick + 1

    # Overridden move at fork_tick causes state at fork_tick + 1 to diverge from original run
    assert (
        canonical_world_hash(forked_tm.engine.environment.state) != original_hashes[fork_tick + 1]
    )

    # Step forked engine forward to tick 20 and verify continued divergence
    forked_hashes: list[str] = [canonical_world_hash(forked_tm.engine.environment.state)]
    for _ in range(20 - (fork_tick + 1)):
        forked_tm.step_forward()
        forked_hashes.append(canonical_world_hash(forked_tm.engine.environment.state))

    assert forked_hashes != original_hashes[fork_tick + 1 :]


@pytest.mark.integration
def test_m05_pygame_viewer_headless_execution(multiagent_setup):
    """Verifies PygameViewer executes in headless mode without crashing or requiring display."""
    engine, adapter, time_machine = multiagent_setup
    viewer = PygameViewer(adapter, time_machine, headless=True)

    time_machine.play()
    for _ in range(10):
        state = viewer.step()
        assert state is not None
        assert state.step == time_machine.engine.environment.state.step
        assert len(state.agents) == 2
