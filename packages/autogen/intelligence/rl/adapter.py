"""Gym-style Environment Adapters over SimulationEngine and canonical RL environments (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from autogen.intelligence.rl.contracts import StepResult
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType, Position


class GymEnvAdapter:
    """Thin Gym-style adapter over the SimulationEngine preserving simulation authority."""

    ACTIONS = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        world: GridWorld,
        agent_id: str = "rl_agent",
        max_steps: int = 200,
    ) -> None:
        self.world = world
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.current_step = 0
        self._prev_energy = 100.0

    def reset(self, seed: int | None = None) -> Observation:
        self.world.reset(seed=seed)
        self.world.spawn_agent(self.agent_id)
        self.current_step = 0
        obs = self.world.get_observation(self.agent_id)
        self._prev_energy = obs.energy
        return obs

    def step(self, action: int | ActionType) -> StepResult:
        self.current_step += 1
        if isinstance(action, int):
            action_type = self.ACTIONS[action % len(self.ACTIONS)]
        else:
            action_type = action

        # Apply transition via environment
        next_world_state = self.world.apply_transitions({self.agent_id: action_type})
        obs = self.world.get_observation(self.agent_id)

        # Biological reward: energy change + survival reward
        energy_delta = obs.energy - self._prev_energy
        self._prev_energy = obs.energy
        reward = float(energy_delta + 0.1)

        is_alive = self.agent_id in next_world_state.agents and next_world_state.agents[self.agent_id].is_alive
        terminated = not is_alive
        truncated = self.current_step >= self.max_steps

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"step": self.current_step, "position": (obs.position.x, obs.position.y)},
        )


class CliffWalkingEnv:
    """The canonical 12x4 CliffWalking environment from Sutton & Barto (Example 6.6).

    Grid dimensions: 12 wide x 4 high.
    Row 0 (bottom): S (0,0), Cliff (1,0)..(10,0), G (11,0).
    Reward: -1 for every step; -100 if stepping into the cliff (resets to start).
    Terminates when reaching Goal (11,0).
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    """

    WIDTH = 12
    HEIGHT = 4
    START = (0, 0)
    GOAL = (11, 0)
    CLIFF = frozenset((x, 0) for x in range(1, 11))

    # (dx, dy)
    OFFSETS = ((0, 1), (0, -1), (-1, 0), (1, 0))

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = max_steps
        self.pos = self.START
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[int, int]:
        self.pos = self.START
        self.steps = 0
        return self.pos

    def step(self, action: int) -> StepResult:
        self.steps += 1
        dx, dy = self.OFFSETS[action % 4]
        new_x = max(0, min(self.WIDTH - 1, self.pos[0] + dx))
        new_y = max(0, min(self.HEIGHT - 1, self.pos[1] + dy))
        new_pos = (new_x, new_y)

        terminated = False
        if new_pos in self.CLIFF:
            reward = -100.0
            self.pos = self.START
        elif new_pos == self.GOAL:
            reward = -1.0
            self.pos = self.GOAL
            terminated = True
        else:
            reward = -1.0
            self.pos = new_pos

        truncated = self.steps >= self.max_steps

        return StepResult(
            observation=self.pos,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"position": self.pos, "steps": self.steps},
        )

