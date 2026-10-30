"""Perception Engine: projects true WorldState into permitted agent Observations."""

from __future__ import annotations

import numpy as np

from autogen.core.rng import SeedManager
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.perception.contracts import PerceptionConfig, PerceptionMode
from autogen.simulation.state.types import Position
from autogen.simulation.state.world_state import CellState, WorldState


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Generates the 2D grid coordinates connecting (x0, y0) to (x1, y1) via Bresenham's algorithm."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    curr_x, curr_y = x0, y0
    while True:
        points.append((curr_x, curr_y))
        if curr_x == x1 and curr_y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            curr_x += sx
        if e2 < dx:
            err += dx
            curr_y += sy
    return points


class PerceptionEngine:
    """Converts the true ground-truth WorldState into permitted Observations per agent."""

    def __init__(self, seed_manager: SeedManager | None = None) -> None:
        self.seed_manager = seed_manager or SeedManager(master_seed=42)
        self._agent_streams: dict[str, np.random.Generator] = {}

    def _get_agent_rng(self, agent_id: str) -> np.random.Generator:
        if agent_id not in self._agent_streams:
            self._agent_streams[agent_id] = self.seed_manager.spawn_stream(f"perception_{agent_id}")
        return self._agent_streams[agent_id]

    def generate_observation(
        self,
        world_state: WorldState,
        agent_id: str,
        config: PerceptionConfig | None = None,
    ) -> Observation:
        """Projects true WorldState into an agent's strictly permitted Observation."""
        agent = world_state.agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' does not exist in world state.")

        cfg = config or PerceptionConfig()
        pos = agent.position
        w, h = world_state.width, world_state.height
        visible_coords: set[tuple[int, int]] = set()

        if cfg.mode == PerceptionMode.FULL:
            for y in range(h):
                for x in range(w):
                    visible_coords.add((x, y))
        elif cfg.mode == PerceptionMode.LOCAL_WINDOW:
            r = cfg.radius
            min_x = max(0, pos.x - r)
            max_x = min(w - 1, pos.x + r)
            min_y = max(0, pos.y - r)
            max_y = min(h - 1, pos.y + r)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    visible_coords.add((x, y))
        elif cfg.mode == PerceptionMode.LINE_OF_SIGHT:
            r = cfg.radius
            visible_coords.add((pos.x, pos.y))
            min_x = max(0, pos.x - r)
            max_x = min(w - 1, pos.x + r)
            min_y = max(0, pos.y - r)
            max_y = min(h - 1, pos.y + r)

            # Generate perimeter points for rays
            perimeter_points: set[tuple[int, int]] = set()
            for x in range(min_x, max_x + 1):
                perimeter_points.add((x, min_y))
                perimeter_points.add((x, max_y))
            for y in range(min_y, max_y + 1):
                perimeter_points.add((min_x, y))
                perimeter_points.add((max_x, y))

            for tx, ty in perimeter_points:
                line = bresenham_line(pos.x, pos.y, tx, ty)
                for cx, cy in line:
                    if not (0 <= cx < w and 0 <= cy < h):
                        break
                    visible_coords.add((cx, cy))
                    cell = world_state.grid[cy][cx]
                    # Non-walkable terrain (walls, mountains) blocks line of sight beyond itself
                    if not TERRAIN_PROPERTIES[cell.terrain].walkable and (
                        cx != pos.x or cy != pos.y
                    ):
                        break

        # Extract visible cells and optionally apply sensor noise
        rng = self._get_agent_rng(agent_id) if cfg.noise.enabled else None
        visible_cells: list[CellState] = []
        for x, y in sorted(visible_coords):
            raw_cell = world_state.grid[y][x]
            if cfg.noise.enabled and rng is not None and raw_cell.food > 0.0:
                noise_offset = float(rng.normal(0.0, cfg.noise.resource_noise_std))
                noisy_food = max(0.0, round(raw_cell.food + noise_offset, 2))
                visible_cells.append(raw_cell.model_copy(update={"food": noisy_food}))
            else:
                visible_cells.append(raw_cell)

        # Detect other agents subject to agent sensing configuration
        visible_agents: list[tuple[str, Position]] = []
        as_cfg = cfg.agent_sensing
        agent_r = as_cfg.sensing_radius

        for other_id, other_agent in world_state.agents.items():
            if other_id == agent_id or not other_agent.is_alive:
                continue
            other_pos = other_agent.position
            # Must be in visible cell or within sensing radius
            dist = max(abs(other_pos.x - pos.x), abs(other_pos.y - pos.y))
            if dist <= agent_r and (other_pos.x, other_pos.y) in visible_coords:
                if cfg.noise.enabled and rng is not None:
                    if rng.uniform(0.0, 1.0) > cfg.noise.agent_detection_probability:
                        continue  # Missed detection due to noise
                reported_id = "anonymous" if as_cfg.anonymous else other_id
                visible_agents.append((reported_id, other_pos))

        return Observation(
            agent_id=agent_id,
            step=world_state.step,
            position=pos,
            energy=agent.energy,
            health=agent.health,
            carried_food=agent.carried_food,
            is_alive=agent.is_alive,
            visible_cells=tuple(visible_cells),
            visible_agents=tuple(visible_agents),
        )
