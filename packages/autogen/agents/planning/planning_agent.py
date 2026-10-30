"""Planning-driven autonomous agent integrating STRIPS planning with GridWorld execution (SYS-08, Module M09)."""

from __future__ import annotations

from collections import deque
from typing import Any

from autogen.agents.base import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.planning.contracts import Fact, GroundAction, Plan, PlanningProblem
from autogen.intelligence.planning.forward import ForwardStateSpacePlanner
from autogen.intelligence.planning.heuristics import h_max
from autogen.intelligence.search.grid_problem import GridSearchProblem
from autogen.intelligence.search.informed import AStarSearch
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import (
    DIRECTION_DISPLACEMENTS,
    TERRAIN_PROPERTIES,
    ActionType,
    Position,
)


class PlanningAgent(BaseAgent):
    """Autonomous agent that formulates high-level STRIPS plans and compiles steps into low-level grid actions."""

    def __init__(
        self,
        agent_id: str,
        planning_problem: PlanningProblem | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(planning=True, belief=True, memory=True))
        self.planner = ForwardStateSpacePlanner(heuristic_fn=h_max)
        self.navigation_search = AStarSearch[Position, ActionType]()
        self.planning_problem = planning_problem

        self.current_plan: Plan | None = None
        self.remaining_plan_actions: deque[GroundAction] = deque()
        self.current_primitive_actions: deque[ActionType] = deque()

        # Mental spatial map
        self.known_walkable: set[Position] = set()
        self.known_targets: dict[str, Position] = {}
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def set_problem(self, problem: PlanningProblem) -> None:
        """Sets or updates the high-level STRIPS planning problem."""
        self.planning_problem = problem
        self.current_plan = None
        self.remaining_plan_actions.clear()
        self.current_primitive_actions.clear()

    def set_target_location(self, landmark_name: str, position: Position) -> None:
        """Associates a high-level location landmark with a spatial GridWorld coordinate."""
        self.known_targets[landmark_name] = position

    def observe(self, observation: Observation) -> None:
        """Updates mental spatial knowledge and current position from observation."""
        super().observe(observation)
        self._step_counter = observation.step

        for cell in observation.visible_cells:
            if TERRAIN_PROPERTIES[cell.terrain].walkable:
                self.known_walkable.add(cell.position)
            else:
                self.known_walkable.discard(cell.position)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        """Executes next primitive action compiled from high-level STRIPS plan."""
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "ClassicalPlanning")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_pos = self._last_observation.position

        # 1. Formulate high-level plan if not yet planned
        if self.current_plan is None and self.planning_problem is not None:
            self.current_plan = self.planner.solve_astar(self.planning_problem)
            if self.current_plan.is_solved:
                self.remaining_plan_actions = deque(self.current_plan.actions)

        # 2. Check if we have primitive actions queued for current subtask
        if self.current_primitive_actions:
            next_act = self.current_primitive_actions.popleft()
            trace_builder.set_selected_action(next_act, "Executing compiled primitive action")
            trace_builder.add_metric("remaining_subplan_steps", len(self.current_primitive_actions))
            trace_builder.add_metric("remaining_strips_actions", len(self.remaining_plan_actions))
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=next_act), trace

        # 3. Advance to next high-level action in STRIPS plan
        if self.remaining_plan_actions:
            next_strips = self.remaining_plan_actions.popleft()

            # Compile STRIPS action to primitive grid actions
            compiled = self._compile_strips_action(next_strips, current_pos)
            if compiled:
                self.current_primitive_actions = deque(compiled)
                next_act = self.current_primitive_actions.popleft()
                trace_builder.set_selected_action(
                    next_act, f"Started STRIPS action '{next_strips.name}'"
                )
            else:
                next_act = ActionType.STAY
                trace_builder.set_selected_action(
                    ActionType.STAY, f"Completed in-place action '{next_strips.name}'"
                )

            trace_builder.add_metric("remaining_strips_actions", len(self.remaining_plan_actions))
            trace = trace_builder.build()
            trace.rationale["strips_action"] = next_strips.name
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=next_act), trace

        # Goal reached or plan empty
        trace_builder.set_selected_action(ActionType.STAY, "STRIPS plan completed or empty")
        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

    def _compile_strips_action(
        self,
        strips_action: GroundAction,
        current_pos: Position,
    ) -> list[ActionType]:
        """Translates a high-level STRIPS action into a sequence of low-level GridWorld actions."""
        # Check if action is Move(from, to)
        if strips_action.name.startswith("Move(") and strips_action.name.endswith(")"):
            inner = strips_action.name[5:-1]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) == 2:
                target_landmark = parts[1]
                target_pos = self.known_targets.get(target_landmark)
                if target_pos is not None and target_pos != current_pos:
                    all_x = [p.x for p in self.known_walkable] + [current_pos.x, target_pos.x]
                    all_y = [p.y for p in self.known_walkable] + [current_pos.y, target_pos.y]
                    max_w = max(all_x) + 5
                    max_h = max(all_y) + 5

                    def _is_walkable(p: Position) -> bool:
                        return p in self.known_walkable or p == current_pos or p == target_pos

                    # Run A* search on grid
                    grid_problem = GridSearchProblem(
                        start=current_pos,
                        goal=target_pos,
                        width=max_w,
                        height=max_h,
                        walkable_fn=_is_walkable,
                    )
                    res = self.navigation_search.solve(grid_problem)
                    if res.success and res.path:
                        return list(res.path)

        # For resource gathering actions, return COLLECT if at target location
        if strips_action.name.startswith("Gather") or strips_action.name.startswith("Collect"):
            return [ActionType.COLLECT]

        return []

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        """Resets agent state for a new simulation run or replay."""
        self._last_observation = None
        self.current_plan = None
        self.remaining_plan_actions.clear()
        self.current_primitive_actions.clear()
        self.known_walkable.clear()
        self.latest_trace = None
        self._step_counter = 0
