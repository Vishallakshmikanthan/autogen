"""Procedural generation of the 2D grid world with guaranteed connectivity."""

from __future__ import annotations

from collections import deque

import numpy as np

from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.state.types import (
    TERRAIN_PROPERTIES,
    Position,
    TerrainType,
)
from autogen.simulation.state.world_state import CellState


class ProceduralWorldGenerator:
    """Generates procedurally structured 2D grid worlds.

    Guarantees that all walkable cells form a single connected component
    via BFS flood-fill and corridor carving.
    """

    def __init__(self, config: EnvironmentConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def generate(self) -> tuple[tuple[CellState, ...], ...]:
        w, h = self.config.width, self.config.height
        terrains = list(self.config.terrain_weights.keys())
        raw_weights = np.array([self.config.terrain_weights[t] for t in terrains], dtype=np.float64)
        probabilities = raw_weights / np.sum(raw_weights)

        # 1. Sample initial terrain grid
        chosen_indices = self.rng.choice(len(terrains), size=(h, w), p=probabilities)
        grid_terrain: list[list[TerrainType]] = [
            [terrains[chosen_indices[y, x]] for x in range(w)] for y in range(h)
        ]

        # 2. Ensure perimeter has at least one access and connect all walkable components
        self._ensure_connectivity(grid_terrain, w, h)

        # 3. Create immutable CellState grid with initialized food
        cell_rows: list[tuple[CellState, ...]] = []
        for y in range(h):
            row_cells: list[CellState] = []
            for x in range(w):
                t_type = grid_terrain[y][x]
                t_props = TERRAIN_PROPERTIES[t_type]
                # Initialize food between 50% and 100% of capacity
                initial_food = (
                    float(t_props.food_capacity * self.rng.uniform(0.5, 1.0))
                    if t_props.walkable
                    else 0.0
                )
                cell = CellState(
                    position=Position(x, y),
                    terrain=t_type,
                    food=initial_food,
                    danger=t_props.danger,
                    movement_cost=t_props.move_cost if t_props.walkable else 0.0,
                )
                row_cells.append(cell)
            cell_rows.append(tuple(row_cells))

        return tuple(cell_rows)

    def _ensure_connectivity(self, grid: list[list[TerrainType]], w: int, h: int) -> None:
        """Partitions walkable cells and carves corridors until a single component remains."""
        components = self._find_connected_components(grid, w, h)

        if not components:
            # If completely impassable, convert center cross to GRASSLAND
            for x in range(w):
                grid[h // 2][x] = TerrainType.GRASSLAND
            for y in range(h):
                grid[y][w // 2] = TerrainType.GRASSLAND
            components = self._find_connected_components(grid, w, h)

        # Find largest component
        components.sort(key=len, reverse=True)
        largest = components[0]
        largest_set = set(largest)

        # Connect every other component to the largest component
        for comp in components[1:]:
            source_pos = comp[0]
            # Find closest cell in largest component
            target_pos = min(
                largest_set,
                key=lambda p: abs(p[0] - source_pos[0]) + abs(p[1] - source_pos[1]),
            )
            # Carve Manhattan corridor between source and target
            self._carve_corridor(grid, source_pos, target_pos, largest_set)

        # Ensure minimum walkable density
        walkable_count = sum(
            1 for y in range(h) for x in range(w) if TERRAIN_PROPERTIES[grid[y][x]].walkable
        )
        total_cells = w * h
        min_required = int(total_cells * self.config.min_walkable_ratio)

        if walkable_count < min_required:
            for y in range(h):
                for x in range(w):
                    if not TERRAIN_PROPERTIES[grid[y][x]].walkable:
                        grid[y][x] = TerrainType.GRASSLAND
                        walkable_count += 1
                        if walkable_count >= min_required:
                            break
                if walkable_count >= min_required:
                    break

    def _find_connected_components(
        self, grid: list[list[TerrainType]], w: int, h: int
    ) -> list[list[tuple[int, int]]]:
        visited: set[tuple[int, int]] = set()
        components: list[list[tuple[int, int]]] = []

        for y in range(h):
            for x in range(w):
                if (x, y) not in visited and TERRAIN_PROPERTIES[grid[y][x]].walkable:
                    comp: list[tuple[int, int]] = []
                    queue = deque([(x, y)])
                    visited.add((x, y))

                    while queue:
                        cx, cy = queue.popleft()
                        comp.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 <= nx < w
                                and 0 <= ny < h
                                and (nx, ny) not in visited
                                and TERRAIN_PROPERTIES[grid[ny][nx]].walkable
                            ):
                                visited.add((nx, ny))
                                queue.append((nx, ny))
                    components.append(comp)

        return components

    def _carve_corridor(
        self,
        grid: list[list[TerrainType]],
        src: tuple[int, int],
        dst: tuple[int, int],
        largest_set: set[tuple[int, int]],
    ) -> None:
        cx, cy = src
        tx, ty = dst

        # Move horizontally first, then vertically
        step_x = 1 if tx > cx else -1
        while cx != tx:
            grid[cy][cx] = TerrainType.GRASSLAND
            largest_set.add((cx, cy))
            cx += step_x

        step_y = 1 if ty > cy else -1
        while cy != ty:
            grid[cy][cx] = TerrainType.GRASSLAND
            largest_set.add((cx, cy))
            cy += step_y

        grid[cy][cx] = TerrainType.GRASSLAND
        largest_set.add((cx, cy))
