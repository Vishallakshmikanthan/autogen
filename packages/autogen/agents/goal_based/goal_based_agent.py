"""Goal-based autonomous agent with pluggable search and path replanning (SYS-06, Module M06)."""

from __future__ import annotations

from collections import deque
from typing import Any

from autogen.agents.base import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.search.contracts import SearchProblem, SearchResult
from autogen.intelligence.search.grid_problem import GridSearchProblem
from autogen.intelligence.search.informed import AStarSearch
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import (
    TERRAIN_PROPERTIES,
    ActionType,
    Position,
)


class GoalBasedAgent(BaseAgent):
    """Agent using state-space search to navigate toward specific goals with path caching and replanning."""

    def __init__(
        self,
        agent_id: str,
        search_strategy: Any = None,
        capabilities: AgentCapabilities | None = None,
        fixed_goal: Position | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities())
        self.search_strategy = search_strategy or AStarSearch[Position, ActionType]()
        self.fixed_goal = fixed_goal
        self.current_goal: Position | None = fixed_goal

        # Mental map of observed cells: Position -> (walkable, food)
        self.known_cells: dict[Position, tuple[bool, float]] = {}
        self.known_food: set[Position] = set()

        # Path cache: sequence of actions planned to reach current_goal
        self._cached_path: deque[ActionType] = deque()
        self.latest_trace: DecisionTrace | None = None
        self._last_search_result: SearchResult[Position, ActionType] | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        """Phase 1: Ingests observation and updates known terrain and food."""
        super().observe(observation)
        self._step_counter = observation.step

        # Ingest observed sensory cells
        for cell in observation.visible_cells:
            pos = cell.position
            walkable = TERRAIN_PROPERTIES[cell.terrain].walkable
            food = cell.food
            self.known_cells[pos] = (walkable, food)

            if food > 0.0:
                self.known_food.add(pos)
            elif pos in self.known_food:
                self.known_food.remove(pos)

        # Invalidation check: if the next scheduled action leads into a newly discovered obstacle
        if self._cached_path and self._last_observation is not None:
            next_act = self._cached_path[0]
            cur_pos = self._last_observation.position
            target_pos = self._predict_next_position(cur_pos, next_act)
            if target_pos in self.known_cells and not self.known_cells[target_pos][0]:
                # Next step is blocked! Invalidate remaining path
                self._cached_path.clear()

    def update_belief(self) -> None:
        """Phase 2: Selects or updates goal target based on internal state and known food."""
        if self._last_observation is None:
            return

        cur_pos = self._last_observation.position

        # If arrived at goal, clear current goal
        if self.current_goal is not None and cur_pos == self.current_goal:
            self._cached_path.clear()
            if cur_pos in self.known_food:
                self.known_food.remove(cur_pos)
            self.current_goal = None

        # If fixed goal is configured, always pursue it
        if self.fixed_goal is not None:
            self.current_goal = self.fixed_goal
            return

        # If no active goal, select closest known food
        if self.current_goal is None and self.known_food:
            best_dist = float("inf")
            best_target: Position | None = None
            for fpos in self.known_food:
                dist = cur_pos.manhattan_distance(fpos)
                if dist < best_dist:
                    best_dist = dist
                    best_target = fpos
            self.current_goal = best_target

    def decide(self) -> tuple[Action, DecisionTrace]:
        """Phase 3: Deliberates, runs search if path is empty, and returns action with trace."""
        if self._last_observation is None:
            raise RuntimeError("decide() called before observe().")

        cur_pos = self._last_observation.position
        energy = self._last_observation.energy
        step_id = self._last_observation.step
        algo_name = self.search_strategy.__class__.__name__

        trace_builder = DecisionTraceBuilder(
            agent_id=self.agent_id,
            step_id=step_id,
            algorithm_name=algo_name,
        )
        trace_builder.set_mechanism("GoalDirectedSearch")
        trace_builder.set_observation_summary(self._last_observation)
        trace_builder.set_belief_summary(
            {
                "current_goal": (self.current_goal.x, self.current_goal.y)
                if self.current_goal
                else None,
                "cached_path_len": len(self._cached_path),
                "known_cells_count": len(self.known_cells),
                "known_food_count": len(self.known_food),
            }
        )

        # 1. Harvest condition: if on goal or current tile has food and energy < 95
        cur_cell = self.known_cells.get(cur_pos)
        if cur_cell and cur_cell[1] > 0.0 and energy < 95.0:
            trace_builder.add_candidate_action(ActionType.COLLECT, score=1.0)
            trace_builder.set_selected_action(ActionType.COLLECT, reason="Harvesting food at goal")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.COLLECT), trace

        # 2. Plan / Replan if path is empty and a goal is defined
        if not self._cached_path and self.current_goal is not None and self.current_goal != cur_pos:
            problem = self._build_search_problem(start=cur_pos, goal=self.current_goal)
            search_res = self.search_strategy.solve(problem)
            self._last_search_result = search_res

            trace_builder.add_metric("search_success", int(search_res.success))
            trace_builder.add_metric("nodes_expanded", search_res.nodes_expanded)
            trace_builder.add_metric("nodes_generated", search_res.nodes_generated)
            trace_builder.add_metric("search_runtime_ms", search_res.runtime_seconds * 1000.0)
            trace_builder.add_metric("total_path_cost", search_res.total_cost)

            if search_res.success and search_res.path:
                self._cached_path = deque(search_res.path)

        # 3. Action selection from path or fallback
        chosen_action: ActionType
        if self._cached_path:
            chosen_action = self._cached_path.popleft()
            reason = f"Following search path towards goal {self.current_goal}"
        else:
            chosen_action = ActionType.STAY
            reason = "No path or goal reachable; remaining in place"

        for act in ActionType:
            score = 1.0 if act == chosen_action else 0.0
            trace_builder.add_candidate_action(act, score=score)

        trace_builder.set_selected_action(chosen_action, reason=reason)
        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def _build_search_problem(
        self, start: Position, goal: Position
    ) -> SearchProblem[Position, ActionType]:
        """Constructs a GridSearchProblem based on agent's known cells."""
        # Find bounds of known area, with minimum size
        all_x = [p.x for p in self.known_cells] + [start.x, goal.x]
        all_y = [p.y for p in self.known_cells] + [start.y, goal.y]
        max_w = max(all_x) + 5
        max_h = max(all_y) + 5

        def _is_walkable(p: Position) -> bool:
            if p in self.known_cells:
                return self.known_cells[p][0]
            # Optimism under uncertainty: unobserved tiles assumed walkable
            return True

        return GridSearchProblem(
            start=start,
            goal=goal,
            width=max_w,
            height=max_h,
            walkable_fn=_is_walkable,
            min_step_cost=1.0,
            heuristic_type="manhattan",
        )

    def _predict_next_position(self, pos: Position, action: ActionType) -> Position:
        if action == ActionType.UP:
            return Position(pos.x, pos.y - 1)
        if action == ActionType.DOWN:
            return Position(pos.x, pos.y + 1)
        if action == ActionType.LEFT:
            return Position(pos.x - 1, pos.y)
        if action == ActionType.RIGHT:
            return Position(pos.x + 1, pos.y)
        return pos

    def learn(self, reward: float, terminal: bool) -> None:
        """Phase 7: No online parameter learning in classical goal-directed search."""
        pass

    def reset(self) -> None:
        """Resets path and known memory for fresh episode."""
        self.known_cells.clear()
        self.known_food.clear()
        self._cached_path.clear()
        self.current_goal = self.fixed_goal
        self.latest_trace = None
