"""Unit tests for Gym-style Environment Adapters and CliffWalkingEnv (SYS-12, Module M13)."""

from __future__ import annotations

import pytest

from autogen.intelligence.rl.adapter import CliffWalkingEnv, GymEnvAdapter
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def test_gym_env_adapter_reset_and_step() -> None:
    """Verifies that GymEnvAdapter provides standard Gym step tuples while preserving simulation state."""
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    adapter = GymEnvAdapter(world=world, agent_id="agent_1", max_steps=10)

    obs = adapter.reset(seed=42)
    assert obs.agent_id == "agent_1"
    assert obs.step == 0

    step_res = adapter.step(action=0)  # UP
    assert not step_res.terminated
    assert not step_res.truncated
    assert "position" in step_res.info
    assert isinstance(step_res.reward, float)


def test_cliff_walking_geometry_and_penalties() -> None:
    """Verifies that CliffWalkingEnv penalizes cliff steps by -100 and teleports back to (0,0)."""
    env = CliffWalkingEnv(max_steps=50)
    start_pos = env.reset()
    assert start_pos == (0, 0)

    # Move RIGHT into the cliff (1, 0)
    res = env.step(3)  # RIGHT
    assert res.reward == -100.0
    assert not res.terminated
    assert res.observation == (0, 0)  # Teleported back to Start

    # Move UP to (0, 1) -> Safe
    res_up = env.step(0)  # UP
    assert res_up.reward == -1.0
    assert res_up.observation == (0, 1)


def test_cliff_walking_goal_termination() -> None:
    """Verifies that reaching Goal (11,0) terminates episode with reward -1."""
    env = CliffWalkingEnv()
    env.reset()
    # Teleport to (11, 1) directly for test
    env.pos = (11, 1)
    # Move DOWN into Goal (11, 0)
    res = env.step(1)  # DOWN
    assert res.reward == -1.0
    assert res.observation == (11, 0)
    assert res.terminated

