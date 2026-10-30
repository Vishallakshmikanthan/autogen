"""Integration tests for deterministic simulation, replay reproduction, snapshots, and forking."""

from __future__ import annotations

import pytest

from autogen.agents.base.dummy_agents import RandomWalkAgent, ScriptedAgent
from autogen.simulation.core.contracts import Action
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType
from autogen.simulation.state.world_state import canonical_world_hash


@pytest.mark.integration
def test_500_ticks_identical_hash_determinism():
    """Invariant: Running 500 ticks twice with identical seed produces bit-identical hashes every tick."""

    def create_run():
        world = GridWorld(EnvironmentConfig(width=10, height=10, seed=777))
        world.reset()
        world.spawn_agent("agent_1")
        world.spawn_agent("agent_2")

        engine = SimulationEngine(environment=world, seed=777)
        engine.register_agent(RandomWalkAgent("agent_1", seed=101))
        engine.register_agent(RandomWalkAgent("agent_2", seed=202))
        return engine

    engine_a = create_run()
    engine_b = create_run()

    hashes_a: list[str] = []
    hashes_b: list[str] = []

    for _ in range(500):
        s_a = engine_a.step()
        s_b = engine_b.step()
        hashes_a.append(canonical_world_hash(s_a))
        hashes_b.append(canonical_world_hash(s_b))

    assert hashes_a == hashes_b


@pytest.mark.integration
def test_snapshot_restore_exactness():
    """Snapshot at tick 200, continue to 300; restore to 200 and continue => identical state sequence."""
    world = GridWorld(EnvironmentConfig(width=8, height=8, seed=42))
    world.reset()
    world.spawn_agent("agent_time")

    engine = SimulationEngine(environment=world, seed=42)
    engine.register_agent(RandomWalkAgent("agent_time", seed=42))

    # Run to tick 200
    for _ in range(200):
        engine.step()

    snapshot_200 = engine.snapshot()
    hash_at_200 = canonical_world_hash(engine.environment.state)

    # Continue from 200 to 250 and collect hashes
    original_hashes: list[str] = []
    for _ in range(50):
        s = engine.step()
        original_hashes.append(canonical_world_hash(s))

    # Restore back to tick 200
    engine.restore(snapshot_200)
    assert canonical_world_hash(engine.environment.state) == hash_at_200

    # Step again from 200 to 250
    restored_hashes: list[str] = []
    for _ in range(50):
        s = engine.step()
        restored_hashes.append(canonical_world_hash(s))

    assert restored_hashes == original_hashes


@pytest.mark.integration
def test_fork_branching_and_divergence():
    """Fork with identical actions equals original; fork with altered action diverges."""
    world = GridWorld(EnvironmentConfig(width=8, height=8, seed=99))
    world.reset()
    world.spawn_agent("agent_fork")

    engine = SimulationEngine(environment=world, seed=99)
    engine.register_agent(ScriptedAgent("agent_fork", script=[ActionType.STAY]))

    for _ in range(50):
        engine.step()

    snap = engine.snapshot()

    # Fork 1: Continues with unchanged action (STAY)
    fork_unchanged = engine.fork(snap)
    s_unchanged = fork_unchanged.step()

    # Step original engine
    s_orig = engine.step()
    assert canonical_world_hash(s_unchanged) == canonical_world_hash(s_orig)

    # Fork 2: Continues with overridden action (COLLECT)
    override = {"agent_fork": Action(agent_id="agent_fork", action_type=ActionType.COLLECT)}
    fork_diverged = engine.fork(snap, overridden_actions=override)

    # The diverged fork should have a different state hash due to food collection / energy diff
    hash_diverged = canonical_world_hash(fork_diverged.environment.state)
    assert hash_diverged != canonical_world_hash(s_orig)


@pytest.mark.integration
def test_1000_tick_integration_run():
    """Integration hook: Runs M01 GridWorld + SimulationEngine + 2 Scripted agents for 1000 ticks."""
    world = GridWorld(EnvironmentConfig(width=16, height=16, seed=12345))
    world.reset()
    world.spawn_agent("agent_alpha")
    world.spawn_agent("agent_beta")

    engine = SimulationEngine(environment=world, seed=12345, snapshot_cadence=100)

    script_alpha = [ActionType.UP, ActionType.RIGHT, ActionType.COLLECT, ActionType.STAY]
    script_beta = [ActionType.DOWN, ActionType.LEFT, ActionType.COLLECT, ActionType.STAY]

    engine.register_agent(ScriptedAgent("agent_alpha", script=script_alpha))
    engine.register_agent(ScriptedAgent("agent_beta", script=script_beta))

    final_state = engine.run(max_ticks=1000)

    assert final_state.step == 1000
    assert len(engine.snapshots) >= 10
    assert len(engine.replay_steps) == 1000
