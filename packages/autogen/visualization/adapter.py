"""Render-free consumer data adapter translating simulation state to ViewerState DTO."""

from __future__ import annotations

from collections import deque
from typing import Any

from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import InvalidActionEvent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.state.world_state import canonical_world_hash
from autogen.visualization.contracts import (
    ViewerAgentMindCell,
    ViewerAgentState,
    ViewerCellState,
    ViewerOverlayData,
    ViewerState,
    ViewerTimeMachineState,
)


class ViewerDataAdapter:
    """Consumer adapter converting engine state and event telemetry into serializable ViewerState."""

    def __init__(self, engine: SimulationEngine, max_trail_len: int = 15) -> None:
        self.engine = engine
        self.max_trail_len = max_trail_len
        self._trails: dict[str, deque[tuple[int, int]]] = {}
        self._collision_markers: list[tuple[int, int]] = []
        self._latest_traces: dict[str, DecisionTrace] = {}
        self._last_trail_step: int = -1

        # Subscribe to event bus for telemetry tracking
        self.engine.event_bus.subscribe(InvalidActionEvent, self._on_invalid_action)

    def _on_invalid_action(self, event: InvalidActionEvent) -> None:
        agent = self.engine.environment.state.agents.get(event.agent_id)
        if agent is not None:
            self._collision_markers.append((agent.position.x, agent.position.y))
            # Keep only the 10 most recent collision markers
            if len(self._collision_markers) > 10:
                self._collision_markers.pop(0)

    def record_decision_trace(self, trace: DecisionTrace) -> None:
        """Records a deliberation trace for agent inspector display."""
        self._latest_traces[trace.agent_id] = trace

    def update_trails(self) -> None:
        """Appends current agent positions to historical movement trails."""
        current_state = self.engine.environment.state
        if current_state.step == self._last_trail_step:
            return
        self._last_trail_step = current_state.step
        for aid, agent in current_state.agents.items():
            if aid not in self._trails:
                self._trails[aid] = deque(maxlen=self.max_trail_len)
            if agent.is_alive:
                self._trails[aid].append((agent.position.x, agent.position.y))

    def generate_viewer_state(
        self,
        selected_agent_id: str | None = None,
        time_machine_state: ViewerTimeMachineState | None = None,
    ) -> ViewerState:
        """Transforms current world state and sensory context into an immutable ViewerState DTO."""
        world_state = self.engine.environment.state
        self.update_trails()

        # 1. Transform grid cells
        cells: list[ViewerCellState] = []
        resource_intensity: dict[str, float] = {}
        for row in world_state.grid:
            for c in row:
                cells.append(
                    ViewerCellState(
                        x=c.position.x,
                        y=c.position.y,
                        terrain=c.terrain.value,
                        food=round(c.food, 2),
                        movement_cost=round(c.movement_cost, 2),
                    )
                )
                if c.food > 0.0:
                    resource_intensity[f"{c.position.x},{c.position.y}"] = round(c.food, 2)

        # 2. Transform agents
        agents: list[ViewerAgentState] = []
        for aid, a in world_state.agents.items():
            agents.append(
                ViewerAgentState(
                    agent_id=aid,
                    x=a.position.x,
                    y=a.position.y,
                    energy=round(a.energy, 2),
                    health=round(a.health, 2),
                    carried_food=round(a.carried_food, 2),
                    is_alive=a.is_alive,
                )
            )

        # 3. Extract subjective agent mind, fog of war, and lookahead tree for selected agent
        fog_visible: list[tuple[int, int]] = []
        agent_mind_cells: list[ViewerAgentMindCell] | None = None
        decision_trace: DecisionTrace | None = None
        lookahead_tree_dict: dict[str, Any] | None = None

        if selected_agent_id and selected_agent_id in world_state.agents:
            agent_obj = self.engine.agents.get(selected_agent_id)
            decision_trace = self._latest_traces.get(selected_agent_id) or getattr(agent_obj, "latest_trace", None)

            # Generate permitted observation to determine fog of war
            p_cfg = self.engine.perception_configs.get(selected_agent_id)
            obs = self.engine.perception_engine.generate_observation(
                world_state, selected_agent_id, p_cfg
            )
            fog_visible = [(c.position.x, c.position.y) for c in obs.visible_cells]

            # Extract internal spatial memory or lookahead tree if present
            agent_obj = self.engine.agents.get(selected_agent_id)
            if agent_obj is not None:
                # Check for adversarial search lookahead tree
                tree_root = getattr(agent_obj, "last_search_tree", None)
                if tree_root is not None:
                    lookahead_tree_dict = self._serialize_game_tree(tree_root)

                # Check for direct or wrapped spatial memory
                sm = getattr(
                    agent_obj,
                    "spatial_memory",
                    getattr(agent_obj, "internal_map", None),
                )
                if sm is not None:
                    agent_mind_cells = []
                    # SpatialMemory store
                    if hasattr(sm, "get_all_cells"):
                        for (mx, my), sc in sm.get_all_cells().items():
                            agent_mind_cells.append(
                                ViewerAgentMindCell(
                                    x=mx,
                                    y=my,
                                    terrain=sc.terrain.value,
                                    food=sc.last_seen_food,
                                    confidence=sc.confidence,
                                    last_observed_tick=sc.last_observed_tick,
                                )
                            )
                    # ModelBasedReflexAgent internal_map dict
                    elif isinstance(sm, dict):
                        for (mx, my), imc in sm.items():
                            agent_mind_cells.append(
                                ViewerAgentMindCell(
                                    x=mx,
                                    y=my,
                                    terrain=imc.terrain.value,
                                    food=imc.food,
                                    confidence=1.0,
                                    last_observed_tick=imc.last_observed_tick,
                                )
                            )

        # 4. Compile overlays
        overlays = ViewerOverlayData(
            fog_of_war_visible=fog_visible,
            movement_trails={aid: list(coords) for aid, coords in self._trails.items()},
            collision_markers=list(self._collision_markers),
            resource_intensity=resource_intensity,
        )

        # 5. Default time machine state if not provided
        tm_state = time_machine_state or ViewerTimeMachineState(
            current_step=world_state.step,
            total_steps=world_state.step,
            available_snapshots=sorted(self.engine.snapshots.keys()),
            is_paused=True,
            playback_speed=1.0,
            can_fork=True,
        )

        return ViewerState(
            step=world_state.step,
            width=world_state.width,
            height=world_state.height,
            seed=self.engine.seed,
            state_hash=canonical_world_hash(world_state),
            grid_cells=cells,
            agents=agents,
            selected_agent_id=selected_agent_id,
            decision_trace=decision_trace,
            agent_mind=agent_mind_cells,
            lookahead_tree=lookahead_tree_dict,
            overlays=overlays,
            time_machine=tm_state,
        )

    def _serialize_game_tree(self, node: Any, max_depth: int = 4) -> dict[str, Any]:
        """Serializes a GameTreeNode into a JSON-compatible dictionary."""
        children_serialized = []
        if getattr(node, "depth", 0) < max_depth and hasattr(node, "children"):
            for child in node.children:
                children_serialized.append(self._serialize_game_tree(child, max_depth))

        return {
            "node_id": str(getattr(node, "node_id", "")),
            "player": getattr(node, "player", 0),
            "depth": getattr(node, "depth", 0),
            "value": round(float(getattr(node, "value", 0.0)), 2),
            "best_action": str(getattr(node, "best_action", "")),
            "alpha": getattr(node, "alpha", None),
            "beta": getattr(node, "beta", None),
            "pruned": getattr(node, "pruned", False),
            "children": children_serialized,
        }

