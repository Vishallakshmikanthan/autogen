"""ViewerState delta-encoding for bandwidth-efficient WebSocket streaming (M19, SYS-18)."""

from __future__ import annotations

from typing import Any

from autogen.visualization.contracts import (
    ViewerState,
)


class DeltaEncoder:
    """Encodes transitions between ViewerState frames to reduce WebSocket bandwidth."""

    @staticmethod
    def encode(
        prev_state: ViewerState | None,
        new_state: ViewerState,
        is_keyframe: bool = False,
    ) -> dict[str, Any]:
        """Calculates minimal delta dictionary between two ViewerStates."""
        if prev_state is None or is_keyframe:
            return {
                "step": new_state.step,
                "is_keyframe": True,
                "state_hash": new_state.state_hash,
                "full_state": new_state.model_dump(),
            }

        # 1. Delta for agents
        prev_agents = {a.agent_id: a for a in prev_state.agents}
        updated_agents: list[dict[str, Any]] = []
        for ag in new_state.agents:
            prev_ag = prev_agents.get(ag.agent_id)
            if prev_ag is None or (
                ag.x != prev_ag.x
                or ag.y != prev_ag.y
                or abs(ag.energy - prev_ag.energy) > 1e-6
                or abs(ag.health - prev_ag.health) > 1e-6
                or abs(ag.carried_food - prev_ag.carried_food) > 1e-6
                or ag.is_alive != prev_ag.is_alive
            ):
                updated_agents.append(ag.model_dump())

        # 2. Delta for modified cells
        prev_cells = {(c.x, c.y): c for c in prev_state.grid_cells}
        modified_cells: list[dict[str, Any]] = []
        for c in new_state.grid_cells:
            prev_c = prev_cells.get((c.x, c.y))
            if prev_c is None or abs(c.food - prev_c.food) > 1e-6 or c.terrain != prev_c.terrain:
                modified_cells.append(c.model_dump())

        return {
            "step": new_state.step,
            "is_keyframe": False,
            "state_hash": new_state.state_hash,
            "updated_agents": updated_agents,
            "modified_cells": modified_cells,
            "decision_trace": new_state.decision_trace.model_dump() if new_state.decision_trace else None,
            "overlays": new_state.overlays.model_dump(),
            "time_machine": new_state.time_machine.model_dump(),
        }

    @staticmethod
    def decode_and_apply(
        prev_state: ViewerState | None,
        delta: dict[str, Any],
    ) -> ViewerState:
        """Applies a delta dictionary to reproduce the exact target ViewerState."""
        if delta.get("is_keyframe", False) or prev_state is None:
            full = delta["full_state"]
            return ViewerState.model_validate(full)

        state_dict = prev_state.model_dump()
        state_dict["step"] = delta["step"]
        state_dict["state_hash"] = delta["state_hash"]

        # Apply updated agents
        current_agents = {a["agent_id"]: a for a in state_dict["agents"]}
        for ag_delta in delta.get("updated_agents", []):
            current_agents[ag_delta["agent_id"]] = ag_delta
        state_dict["agents"] = list(current_agents.values())

        # Apply modified cells
        cell_map = {(c["x"], c["y"]): c for c in state_dict["grid_cells"]}
        for c_delta in delta.get("modified_cells", []):
            cell_map[(c_delta["x"], c_delta["y"])] = c_delta
        state_dict["grid_cells"] = list(cell_map.values())

        state_dict["decision_trace"] = delta.get("decision_trace")
        if "overlays" in delta:
            state_dict["overlays"] = delta["overlays"]
        if "time_machine" in delta:
            state_dict["time_machine"] = delta["time_machine"]

        return ViewerState.model_validate(state_dict)
