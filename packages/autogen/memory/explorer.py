"""Memory Explorer API: read-only inspection interface for UI and debugging."""

from __future__ import annotations

import numpy as np

from autogen.memory.contracts import EpisodicTransition, MemorySnapshot, SpatialMemoryCell
from autogen.memory.episodic_memory import EpisodicMemory
from autogen.memory.spatial_memory import SpatialMemory
from autogen.memory.working_memory import WorkingMemory
from autogen.simulation.core.contracts import Observation


class MemoryExplorerAPI:
    """Read-only view of agent memory contents for UI inspection and Algorithm Lab."""

    def __init__(
        self,
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
        spatial_memory: SpatialMemory,
    ) -> None:
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = spatial_memory

    def get_working_memory(self) -> list[Observation]:
        """Returns all currently buffered working memory observations."""
        return self._wm.get_all()

    def get_episodic_memory(self, limit: int = 50) -> list[EpisodicTransition]:
        """Returns the most recent episodic transitions up to limit."""
        return self._em.get_recent(limit=limit)

    def get_spatial_memory(self) -> dict[tuple[int, int], SpatialMemoryCell]:
        """Returns all remembered spatial cells with their current confidence levels."""
        return self._sm.get_all_cells()

    def export_snapshot(self, agent_id: str, current_step: int) -> MemorySnapshot:
        """Constructs an immutable MemorySnapshot for serializing to UI or diagnostics."""
        spatial_cells = list(self._sm.get_all_cells().values())
        confidences = [c.confidence for c in spatial_cells]
        mean_conf = float(np.mean(confidences)) if confidences else 1.0

        wm_steps = [o.step for o in self._wm.get_all()]

        return MemorySnapshot(
            agent_id=agent_id,
            current_step=current_step,
            working_memory_size=len(self._wm),
            episodic_memory_size=len(self._em),
            spatial_cells_count=len(spatial_cells),
            mean_spatial_confidence=round(mean_conf, 4),
            working_memory_steps=wm_steps,
            spatial_cells=spatial_cells,
        )
