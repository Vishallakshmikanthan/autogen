"""Spatial memory store with confidence decay and capacity eviction."""

from __future__ import annotations

import math

from autogen.memory.contracts import (
    DecayPolicy,
    EvictionPolicy,
    MemoryConfig,
    SpatialMemoryCell,
)
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.state.types import Position


class SpatialMemory:
    """Spatial memory mapping (x, y) coordinates to remembered cells with decaying certainty."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._cells: dict[tuple[int, int], SpatialMemoryCell] = {}

    def update_from_observation(self, observation: Observation) -> None:
        """Ingests new observation cells and decays certainty on unobserved cells."""
        step = observation.step
        observed_positions: set[tuple[int, int]] = set()

        # 1. Ingest freshly observed cells (confidence = 1.0)
        for c in observation.visible_cells:
            pos_key = (c.position.x, c.position.y)
            observed_positions.add(pos_key)
            self._cells[pos_key] = SpatialMemoryCell(
                position=c.position,
                terrain=c.terrain,
                walkable=TERRAIN_PROPERTIES[c.terrain].walkable,
                last_seen_food=c.food,
                last_observed_tick=step,
                confidence=1.0,
            )

        # 2. Decay confidence of previously observed cells that were not seen this tick
        decay_rate = self.config.decay_rate
        decay_policy = self.config.decay_policy

        for pos_key, cell in list(self._cells.items()):
            if pos_key in observed_positions:
                continue

            delta_t = step - cell.last_observed_tick
            if delta_t <= 0:
                continue

            if decay_policy == DecayPolicy.EXPONENTIAL:
                new_conf = math.exp(-decay_rate * delta_t)
            elif decay_policy == DecayPolicy.LINEAR:
                new_conf = max(0.0, 1.0 - (decay_rate * delta_t))
            elif decay_policy == DecayPolicy.STEP:
                step_limit = int(1.0 / decay_rate) if decay_rate > 0 else 100
                new_conf = 1.0 if delta_t <= step_limit else 0.0
            else:
                new_conf = math.exp(-decay_rate * delta_t)

            self._cells[pos_key] = cell.model_copy(update={"confidence": round(new_conf, 4)})

        # 3. Enforce capacity bounds and prune negligible certainty
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        """Evicts cells if capacity limit is exceeded or certainty is below floor."""
        max_cap = self.config.spatial_memory_capacity
        floor = self.config.confidence_floor
        eviction = self.config.eviction_policy

        # Prune cells below confidence floor if over 80% capacity
        if len(self._cells) > int(max_cap * 0.8):
            for k, cell in list(self._cells.items()):
                if cell.confidence < floor:
                    del self._cells[k]

        # If still over capacity, evict excess based on eviction policy
        if len(self._cells) > max_cap:
            excess = len(self._cells) - max_cap
            if eviction == EvictionPolicy.LEAST_CONFIDENT:
                # Sort ascending by confidence
                sorted_keys = sorted(self._cells.keys(), key=lambda k: self._cells[k].confidence)
            elif eviction == EvictionPolicy.FIFO:
                # Sort ascending by last_observed_tick
                sorted_keys = sorted(
                    self._cells.keys(), key=lambda k: self._cells[k].last_observed_tick
                )
            else:
                sorted_keys = sorted(self._cells.keys(), key=lambda k: self._cells[k].confidence)

            for k in sorted_keys[:excess]:
                del self._cells[k]

    def get_cell(self, position: Position | tuple[int, int]) -> SpatialMemoryCell | None:
        """Retrieves a single remembered cell by coordinate."""
        pos_key = (position.x, position.y) if isinstance(position, Position) else position
        return self._cells.get(pos_key)

    def get_all_cells(self) -> dict[tuple[int, int], SpatialMemoryCell]:
        """Returns all currently retained spatial memory cells."""
        return dict(self._cells)

    def get_known_food(self, min_confidence: float = 0.1) -> list[SpatialMemoryCell]:
        """Returns cells known to have food with confidence above threshold."""
        return [
            c
            for c in self._cells.values()
            if c.last_seen_food > 0.0 and c.confidence >= min_confidence
        ]

    def clear(self) -> None:
        """Clears all stored spatial memory cells."""
        self._cells.clear()

    def __len__(self) -> int:
        return len(self._cells)
