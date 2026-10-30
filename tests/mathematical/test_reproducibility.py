"""Mathematical tests verifying PRNG determinism and canonical world state hashing."""

from __future__ import annotations

import pytest

from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType
from autogen.simulation.state.world_state import canonical_world_hash


@pytest.mark.mathematical
def test_identical_seed_produces_identical_state_hash():
    """Invariant: Re-running identical seed with identical actions yields bit-identical hashes."""
    config1 = EnvironmentConfig(width=12, height=12, seed=42)
    world1 = GridWorld(config1)
    state1 = world1.reset()
    world1.spawn_agent("agent_1")

    config2 = EnvironmentConfig(width=12, height=12, seed=42)
    world2 = GridWorld(config2)
    state2 = world2.reset()
    world2.spawn_agent("agent_1")

    assert canonical_world_hash(state1) == canonical_world_hash(state2)

    # Step both environments synchronously for 25 steps with identical actions
    actions = [
        ActionType.STAY,
        ActionType.UP,
        ActionType.RIGHT,
        ActionType.COLLECT,
        ActionType.DOWN,
    ]
    for step_i in range(25):
        act = actions[step_i % len(actions)]
        s1 = world1.apply_transitions({"agent_1": act})
        s2 = world2.apply_transitions({"agent_1": act})
        assert canonical_world_hash(s1) == canonical_world_hash(s2)


@pytest.mark.mathematical
def test_different_seeds_produce_distinct_state_hashes():
    """Invariant: Distinct seeds produce distinct world topologies and hashes."""
    hashes = set()
    for seed in [1, 2, 3, 4, 5]:
        world = GridWorld(EnvironmentConfig(width=10, height=10, seed=seed))
        state = world.reset()
        h = canonical_world_hash(state)
        hashes.add(h)

    assert len(hashes) == 5
