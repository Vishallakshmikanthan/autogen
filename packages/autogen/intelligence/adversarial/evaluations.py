"""Evaluation heuristics and move-ordering strategies for adversarial subgames (SYS-10, Module M07)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from autogen.simulation.state.types import ActionType, Position


@dataclass(frozen=True, slots=True)
class PredatorPreyState:
    """Immutable state representation of a localized predator-prey pursuit subgame."""

    predator_pos: Position
    prey_pos: Position
    predator_energy: float
    prey_energy: float
    turn: int  # 0: Predator (MAX), 1: Prey (MIN)
    blocked_cells: frozenset[Position] = frozenset()
    grid_width: int = 50
    grid_height: int = 50
    is_captured: bool = False


@dataclass(slots=True)
class PredatorPreyEvaluator:
    """Parametrized linear evaluation function for predator-prey lookahead.

    Math model:
        V(s) = - w_dist * d_manhattan(pred, prey)
               - w_escape * degree_of_freedom(prey)
               + w_energy * (energy_pred - energy_prey)
               + capture_bonus (if captured)
    """

    w_distance: float = 10.0
    w_escape_routes: float = 5.0
    w_energy: float = 0.5
    capture_bonus: float = 1000.0

    def __call__(self, state: PredatorPreyState, player: int) -> float:
        if state.is_captured or state.predator_pos == state.prey_pos:
            return self.capture_bonus if player == 0 else -self.capture_bonus

        # Manhattan distance
        dist = state.predator_pos.manhattan_distance(state.prey_pos)

        # Degree of freedom for prey (open walkable non-obstacle cardinal neighbors)
        deg_free = self.count_escape_routes(state)

        # Energy differential
        d_energy = state.predator_energy - state.prey_energy

        # Base value from MAX (predator) perspective
        # Predator wants: min distance (hence -w_distance * dist), min escape routes, max energy
        val_max = (
            - self.w_distance * dist
            - self.w_escape_routes * deg_free
            + self.w_energy * d_energy
        )

        return val_max if player == 0 else -val_max

    def count_escape_routes(self, state: PredatorPreyState) -> int:
        """Counts unblocked adjacent cells surrounding prey."""
        routes = 0
        px, py = state.prey_pos.x, state.prey_pos.y
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < state.grid_width and 0 <= ny < state.grid_height:
                npos = Position(x=nx, y=ny)
                if npos not in state.blocked_cells and npos != state.predator_pos:
                    routes += 1
        return routes


def order_predator_moves(
    actions: Sequence[ActionType],
    state: PredatorPreyState,
    player: int,
) -> list[ActionType]:
    """Orders predator moves so that actions moving closer to the prey are evaluated first.

    This provides strong move-ordering in Alpha-Beta pruning, driving complexity towards O(b^(d/2)).
    """
    def action_priority(action: ActionType) -> float:
        dx, dy = 0, 0
        if action == ActionType.UP:
            dy = -1
        elif action == ActionType.DOWN:
            dy = 1
        elif action == ActionType.LEFT:
            dx = -1
        elif action == ActionType.RIGHT:
            dx = 1

        if player == 0:  # Predator (MAX) prefers minimizing distance to prey
            next_pos = Position(x=state.predator_pos.x + dx, y=state.predator_pos.y + dy)
            return float(next_pos.manhattan_distance(state.prey_pos))
        else:  # Prey (MIN) prefers maximizing distance from predator
            next_pos = Position(x=state.prey_pos.x + dx, y=state.prey_pos.y + dy)
            return -float(next_pos.manhattan_distance(state.predator_pos))

    return sorted(actions, key=action_priority)
