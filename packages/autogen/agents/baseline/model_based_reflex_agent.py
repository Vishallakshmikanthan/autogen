"""Model-based reflex agent maintaining an internal spatial map."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.state.types import ActionType, TerrainType

DIRECTION_DELTAS: dict[ActionType, tuple[int, int]] = {
    ActionType.UP: (0, -1),
    ActionType.DOWN: (0, 1),
    ActionType.LEFT: (-1, 0),
    ActionType.RIGHT: (1, 0),
}


@dataclass(slots=True)
class InternalMapCell:
    """Internal representation of a cell known to the agent."""

    terrain: TerrainType
    walkable: bool
    food: float
    last_observed_tick: int


class ModelBasedReflexAgent(BaseAgent):
    """Reflex agent with internal state maintaining an internal map of observed world cells."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(memory=True))
        self.internal_map: dict[tuple[int, int], InternalMapCell] = {}
        self.known_food: dict[tuple[int, int], float] = {}
        self._step_count = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        """Updates internal spatial map strictly from permitted observations."""
        if self._last_observation is None:
            return

        step = self._last_observation.step
        for cell in self._last_observation.visible_cells:
            cpos = (cell.position.x, cell.position.y)
            is_walkable = TERRAIN_PROPERTIES[cell.terrain].walkable
            self.internal_map[cpos] = InternalMapCell(
                terrain=cell.terrain,
                walkable=is_walkable,
                food=cell.food,
                last_observed_tick=step,
            )

            if cell.food > 0.0:
                self.known_food[cpos] = cell.food
            elif cpos in self.known_food:
                del self.known_food[cpos]

    def _bfs_path(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Finds shortest path to goal using internal map of known walkable cells."""
        if start == goal:
            return []

        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque([(start, [])])
        visited: set[tuple[int, int]] = {start}

        while queue:
            curr, path = queue.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nxt = (curr[0] + dx, curr[1] + dy)
                if nxt == goal:
                    return path + [nxt]
                if nxt not in visited:
                    cell = self.internal_map.get(nxt)
                    if cell is not None and cell.walkable:
                        visited.add(nxt)
                        queue.append((nxt, path + [nxt]))
        return None

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            raise RuntimeError("Agent must observe before deciding.")

        obs = self._last_observation
        self._step_count += 1
        pos = (obs.position.x, obs.position.y)
        current_cell = self.internal_map.get(pos)

        builder = (
            DecisionTraceBuilder(self.agent_id, obs.step, "internal_map_deliberation")
            .set_mechanism("model_based_memory_search")
            .set_observation_summary(obs)
            .set_belief_summary(
                {
                    "mapped_cells_count": len(self.internal_map),
                    "known_food_sources": len(self.known_food),
                }
            )
            .add_metric("step_count", self._step_count)
            .add_metric("mapped_cells", len(self.internal_map))
        )

        all_actions = [
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
            ActionType.STAY,
            ActionType.COLLECT,
        ]
        scores: dict[ActionType, float] = dict.fromkeys(all_actions, 0.0)

        selected_action: ActionType
        decision_reason: str

        # RULE 1: Harvest food present at feet
        if current_cell is not None and current_cell.food > 0.0:
            scores[ActionType.COLLECT] = 100.0
            selected_action = ActionType.COLLECT
            decision_reason = (
                f"Harvesting known food ({current_cell.food:.1f}) at current location."
            )
        else:
            # RULE 2: Target known food anywhere in internal map
            target_food_pos: tuple[int, int] | None = None
            best_food_path: list[tuple[int, int]] | None = None

            if self.known_food:
                # Sort food positions by Manhattan distance heuristic
                sorted_food = sorted(
                    self.known_food.keys(),
                    key=lambda f: abs(f[0] - pos[0]) + abs(f[1] - pos[1]),
                )
                for fpos in sorted_food:
                    path = self._bfs_path(pos, fpos)
                    if path:
                        best_food_path = path
                        target_food_pos = fpos
                        break

            if best_food_path and target_food_pos is not None:
                next_step = best_food_path[0]
                dx = next_step[0] - pos[0]
                dy = next_step[1] - pos[1]
                for act, (adx, ady) in DIRECTION_DELTAS.items():
                    if (dx, dy) == (adx, ady):
                        selected_action = act
                        scores[act] = 80.0
                    else:
                        scores[act] = 10.0
                decision_reason = (
                    f"Navigating along internal map to known food at {target_food_pos} "
                    f"({len(best_food_path)} steps away)."
                )
            else:
                # RULE 3: Frontier exploration (navigate toward least recently observed walkable neighbor)
                best_frontier_action: ActionType | None = None
                oldest_tick = float("inf")

                for act, (adx, ady) in DIRECTION_DELTAS.items():
                    nxt = (pos[0] + adx, pos[1] + ady)
                    cell = self.internal_map.get(nxt)
                    if cell is not None and cell.walkable:
                        score = 30.0 - min(20.0, float(cell.last_observed_tick))
                        scores[act] = score
                        if cell.last_observed_tick < oldest_tick:
                            oldest_tick = float(cell.last_observed_tick)
                            best_frontier_action = act
                    elif cell is None:
                        # Unknown cell adjacent! High exploration value
                        scores[act] = 50.0
                        best_frontier_action = act
                        oldest_tick = -1.0
                    else:
                        scores[act] = 0.0

                if best_frontier_action is not None:
                    selected_action = best_frontier_action
                    decision_reason = (
                        f"Exploring map frontier along heading {best_frontier_action.value}."
                    )
                else:
                    selected_action = ActionType.STAY
                    decision_reason = "No viable walkable path found in internal map; staying."

        for act in all_actions:
            builder.add_candidate_action(act, scores[act])

        builder.set_selected_action(selected_action, decision_reason)
        trace = builder.build()
        action = Action(agent_id=self.agent_id, action_type=selected_action)
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.internal_map.clear()
        self.known_food.clear()
        self._step_count = 0
