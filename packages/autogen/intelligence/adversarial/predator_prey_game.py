"""Predator-prey alternating turn-based subgame formulation (SYS-10, Module M07)."""

from __future__ import annotations

from autogen.intelligence.adversarial.evaluations import PredatorPreyState
from autogen.simulation.state.types import ActionType, Position


class PredatorPreySubGame:
    """Turn-based lookahead subgame abstraction approximating simultaneous predator-prey dynamics."""

    def __init__(self, initial_state: PredatorPreyState, energy_cost_per_move: float = 0.5) -> None:
        self._initial_state = initial_state
        self.energy_cost_per_move = energy_cost_per_move

    @property
    def initial_state(self) -> PredatorPreyState:
        return self._initial_state

    def player(self, state: PredatorPreyState) -> int:
        return state.turn

    def legal_actions(self, state: PredatorPreyState) -> list[ActionType]:
        if self.is_terminal(state):
            return []

        cur_pos = state.predator_pos if state.turn == 0 else state.prey_pos
        legal: list[ActionType] = [ActionType.STAY]

        deltas = [
            (ActionType.UP, 0, -1),
            (ActionType.DOWN, 0, 1),
            (ActionType.LEFT, -1, 0),
            (ActionType.RIGHT, 1, 0),
        ]

        for act, dx, dy in deltas:
            nx, ny = cur_pos.x + dx, cur_pos.y + dy
            if 0 <= nx < state.grid_width and 0 <= ny < state.grid_height:
                npos = Position(x=nx, y=ny)
                if npos not in state.blocked_cells:
                    legal.append(act)

        return legal

    def successor(self, state: PredatorPreyState, action: ActionType) -> PredatorPreyState:
        dx, dy = 0, 0
        if action == ActionType.UP:
            dy = -1
        elif action == ActionType.DOWN:
            dy = 1
        elif action == ActionType.LEFT:
            dx = -1
        elif action == ActionType.RIGHT:
            dx = 1

        if state.turn == 0:  # Predator moves
            new_pred_pos = Position(x=state.predator_pos.x + dx, y=state.predator_pos.y + dy)
            is_captured = new_pred_pos == state.prey_pos
            return PredatorPreyState(
                predator_pos=new_pred_pos,
                prey_pos=state.prey_pos,
                predator_energy=max(0.0, state.predator_energy - self.energy_cost_per_move),
                prey_energy=state.prey_energy,
                turn=1,
                blocked_cells=state.blocked_cells,
                grid_width=state.grid_width,
                grid_height=state.grid_height,
                is_captured=is_captured,
            )
        else:  # Prey moves
            new_prey_pos = Position(x=state.prey_pos.x + dx, y=state.prey_pos.y + dy)
            is_captured = state.predator_pos == new_prey_pos
            return PredatorPreyState(
                predator_pos=state.predator_pos,
                prey_pos=new_prey_pos,
                predator_energy=state.predator_energy,
                prey_energy=max(0.0, state.prey_energy - self.energy_cost_per_move),
                turn=0,
                blocked_cells=state.blocked_cells,
                grid_width=state.grid_width,
                grid_height=state.grid_height,
                is_captured=is_captured,
            )

    def is_terminal(self, state: PredatorPreyState) -> bool:
        return state.is_captured or state.predator_pos == state.prey_pos or state.prey_energy <= 0.0

    def utility(self, state: PredatorPreyState, player: int) -> float:
        if state.is_captured or state.predator_pos == state.prey_pos:
            return 1000.0 if player == 0 else -1000.0
        if state.prey_energy <= 0.0:
            return 500.0 if player == 0 else -500.0
        return 0.0

    def chance_outcomes(self, state: PredatorPreyState, action: ActionType) -> list[tuple[PredatorPreyState, float]]:
        # Deterministic subgame transition by default
        return [(self.successor(state, action), 1.0)]
