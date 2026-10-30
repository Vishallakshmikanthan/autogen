"""Unit tests for SpatialMemory, confidence decay monotonicity, and anti-leakage invariants."""

import math

import pytest

from autogen.memory.contracts import DecayPolicy, EvictionPolicy, MemoryConfig
from autogen.memory.spatial_memory import SpatialMemory
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position, TerrainType
from autogen.simulation.state.world_state import CellState


def _make_cell(x: int, y: int, food: float = 0.0) -> CellState:
    return CellState(
        position=Position(x, y),
        terrain=TerrainType.GRASSLAND,
        food=food,
        danger=0.0,
        movement_cost=1.0,
    )


def _make_obs(step: int, cells: list[CellState]) -> Observation:
    return Observation(
        agent_id="test",
        step=step,
        position=Position(0, 0),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        is_alive=True,
        visible_cells=tuple(cells),
        visible_agents=(),
    )


def test_spatial_memory_ingest_and_confidence_decay_monotonicity():
    """Verify confidence is 1.0 upon observation and decays monotonically with staleness."""
    cfg = MemoryConfig(decay_policy=DecayPolicy.EXPONENTIAL, decay_rate=0.05)
    sm = SpatialMemory(config=cfg)

    # Tick 0: Observe cell at (2, 2)
    obs_0 = _make_obs(step=0, cells=[_make_cell(2, 2, food=10.0)])
    sm.update_from_observation(obs_0)

    cell_t0 = sm.get_cell((2, 2))
    assert cell_t0 is not None
    assert cell_t0.confidence == 1.0
    assert cell_t0.last_seen_food == 10.0

    # Ticks 1 through 5: Observe elsewhere (cell (2, 2) is now unobserved and decaying)
    prev_conf = 1.0
    for tick in range(1, 6):
        obs_other = _make_obs(step=tick, cells=[_make_cell(9, 9)])
        sm.update_from_observation(obs_other)

        cell_decayed = sm.get_cell((2, 2))
        assert cell_decayed is not None
        # Monotonic decay invariant: c_t <= c_{t-1}
        assert cell_decayed.confidence <= prev_conf
        expected_conf = math.exp(-0.05 * tick)
        assert cell_decayed.confidence == pytest.approx(expected_conf, abs=0.01)
        prev_conf = cell_decayed.confidence


def test_spatial_memory_linear_decay():
    cfg = MemoryConfig(decay_policy=DecayPolicy.LINEAR, decay_rate=0.1)
    sm = SpatialMemory(config=cfg)

    sm.update_from_observation(_make_obs(step=0, cells=[_make_cell(1, 1)]))
    cell_0 = sm.get_cell((1, 1))
    assert cell_0 is not None
    assert cell_0.confidence == 1.0

    # At delta_t = 4: conf = 1.0 - 0.4 = 0.6
    sm.update_from_observation(_make_obs(step=4, cells=[]))
    cell_4 = sm.get_cell((1, 1))
    assert cell_4 is not None
    assert cell_4.confidence == pytest.approx(0.6, abs=0.01)

    # At delta_t = 12: conf floors at 0.0
    sm.update_from_observation(_make_obs(step=12, cells=[]))
    # Cell may be pruned or confidence is 0.0
    cell = sm.get_cell((1, 1))
    if cell is not None:
        assert cell.confidence == 0.0


def test_spatial_memory_capacity_and_eviction():
    cfg = MemoryConfig(
        spatial_memory_capacity=5,
        eviction_policy=EvictionPolicy.LEAST_CONFIDENT,
        decay_rate=0.05,
    )
    sm = SpatialMemory(config=cfg)

    # Ingest 5 cells at tick 0
    cells_t0 = [_make_cell(i, 0) for i in range(5)]
    sm.update_from_observation(_make_obs(step=0, cells=cells_t0))
    assert len(sm) == 5

    # Ingest 3 new cells at tick 10 (causing capacity overflow)
    new_cells = [_make_cell(10, i) for i in range(3)]
    sm.update_from_observation(_make_obs(step=10, cells=new_cells))

    # Total stored cells must not exceed capacity
    assert len(sm) <= 5
    # Freshly observed cells must be preserved (confidence 1.0)
    for c in new_cells:
        assert sm.get_cell((c.position.x, c.position.y)) is not None


def test_spatial_memory_anti_leakage_invariant():
    """Security invariant: memory never contains cells that were never in an observation."""
    sm = SpatialMemory()

    observed_coords = {(1, 2), (3, 4), (5, 6)}
    cells = [_make_cell(x, y) for x, y in observed_coords]
    sm.update_from_observation(_make_obs(step=0, cells=cells))

    all_stored = set(sm.get_all_cells().keys())
    assert all_stored == observed_coords

    # Any unobserved coordinate is strictly absent
    for unobserved in [(0, 0), (10, 10), (2, 1), (99, 99)]:
        assert unobserved not in all_stored
        assert sm.get_cell(unobserved) is None
