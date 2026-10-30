"""Unit tests for MemoryExplorerAPI snapshot export."""

from autogen.memory.contracts import MemorySnapshot
from autogen.memory.episodic_memory import EpisodicMemory
from autogen.memory.explorer import MemoryExplorerAPI
from autogen.memory.spatial_memory import SpatialMemory
from autogen.memory.working_memory import WorkingMemory
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position, TerrainType
from autogen.simulation.state.world_state import CellState


def test_memory_explorer_snapshot():
    wm = WorkingMemory(capacity=5)
    em = EpisodicMemory(capacity=10)
    sm = SpatialMemory()

    obs = Observation(
        agent_id="agent_1",
        step=5,
        position=Position(3, 3),
        energy=95.0,
        health=100.0,
        carried_food=0.0,
        is_alive=True,
        visible_cells=(
            CellState(
                position=Position(3, 3),
                terrain=TerrainType.GRASSLAND,
                food=2.0,
                danger=0.0,
                movement_cost=1.0,
            ),
        ),
        visible_agents=(),
    )

    wm.push(obs)
    sm.update_from_observation(obs)

    explorer = MemoryExplorerAPI(wm, em, sm)
    snapshot = explorer.export_snapshot(agent_id="agent_1", current_step=5)

    assert isinstance(snapshot, MemorySnapshot)
    assert snapshot.agent_id == "agent_1"
    assert snapshot.current_step == 5
    assert snapshot.working_memory_size == 1
    assert snapshot.spatial_cells_count == 1
    assert snapshot.mean_spatial_confidence == 1.0
    assert len(snapshot.spatial_cells) == 1
    assert snapshot.spatial_cells[0].last_seen_food == 2.0
