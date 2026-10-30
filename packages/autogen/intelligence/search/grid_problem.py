"""Grid search problem adapter conforming to SearchProblem protocol (SYS-06, Module M06)."""

from __future__ import annotations

from collections.abc import Callable

from autogen.intelligence.search.contracts import SearchProblem
from autogen.intelligence.search.heuristics import (
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
    null_heuristic,
    octile_distance,
)
from autogen.simulation.state.types import ActionType, Position


class GridSearchProblem(SearchProblem[Position, ActionType]):
    """Standard 4-connected grid navigation search problem."""

    # Direction offsets for 4-connected grid
    _DELTAS: list[tuple[ActionType, int, int]] = [
        (ActionType.UP, 0, -1),
        (ActionType.RIGHT, 1, 0),
        (ActionType.DOWN, 0, 1),
        (ActionType.LEFT, -1, 0),
    ]

    def __init__(
        self,
        start: Position,
        goal: Position | set[Position] | Callable[[Position], bool],
        width: int,
        height: int,
        walkable_fn: Callable[[Position], bool] | set[Position],
        step_cost_fn: Callable[[Position, ActionType, Position], float] | None = None,
        min_step_cost: float = 1.0,
        heuristic_type: str = "manhattan",
    ) -> None:
        self._start = start
        self._width = width
        self._height = height
        self._min_step_cost = min_step_cost
        self._heuristic_type = heuristic_type

        # Goal configuration
        if isinstance(goal, Position):
            self._target_pos: Position | None = goal
            self._is_goal_fn: Callable[[Position], bool] = lambda p: p == goal
        elif isinstance(goal, set):
            self._target_pos = next(iter(goal)) if goal else None
            self._is_goal_fn = lambda p: p in goal
        else:
            self._target_pos = None
            self._is_goal_fn = goal

        # Walkability configuration
        if isinstance(walkable_fn, set):
            self._walkable_fn: Callable[[Position], bool] = lambda p: p in walkable_fn
        else:
            self._walkable_fn = walkable_fn

        # Step cost configuration
        self._cost_fn = step_cost_fn

    @property
    def initial_state(self) -> Position:
        return self._start

    def is_goal(self, state: Position) -> bool:
        return self._is_goal_fn(state)

    def successors(self, state: Position) -> list[tuple[ActionType, Position, float]]:
        succs: list[tuple[ActionType, Position, float]] = []

        for action, dx, dy in self._DELTAS:
            nx = state.x + dx
            ny = state.y + dy

            # Bounds check
            if 0 <= nx < self._width and 0 <= ny < self._height:
                npos = Position(nx, ny)
                if self._walkable_fn(npos):
                    cost = (
                        self._cost_fn(state, action, npos)
                        if self._cost_fn is not None
                        else self._min_step_cost
                    )
                    succs.append((action, npos, cost))

        return succs

    def heuristic(self, state: Position) -> float:
        if self._target_pos is None:
            return 0.0

        if self._heuristic_type == "manhattan":
            return manhattan_distance(state, self._target_pos, min_step_cost=self._min_step_cost)
        if self._heuristic_type == "euclidean":
            return euclidean_distance(state, self._target_pos, min_step_cost=self._min_step_cost)
        if self._heuristic_type == "chebyshev":
            return chebyshev_distance(state, self._target_pos, min_step_cost=self._min_step_cost)
        if self._heuristic_type == "octile":
            return octile_distance(state, self._target_pos, min_step_cost=self._min_step_cost)
        return null_heuristic(state)

    @classmethod
    def from_cost_matrix(
        cls,
        grid_matrix: list[list[float]],  # float('inf') for obstacles
        start: Position,
        goal: Position,
        heuristic_type: str = "manhattan",
    ) -> GridSearchProblem:
        """Constructs a GridSearchProblem from a 2D matrix of terrain step costs."""
        height = len(grid_matrix)
        width = len(grid_matrix[0]) if height > 0 else 0

        # Determine non-obstacle minimum step cost
        costs = [
            c
            for row in grid_matrix
            for c in row
            if c != float("inf") and not (isinstance(c, float) and (c <= 0 or c != c))
        ]
        min_cost = min(costs) if costs else 1.0

        def _walkable(p: Position) -> bool:
            return grid_matrix[p.y][p.x] != float("inf")

        def _cost(p: Position, a: ActionType, np: Position) -> float:
            return grid_matrix[np.y][np.x]

        return cls(
            start=start,
            goal=goal,
            width=width,
            height=height,
            walkable_fn=_walkable,
            step_cost_fn=_cost,
            min_step_cost=min_cost,
            heuristic_type=heuristic_type,
        )
