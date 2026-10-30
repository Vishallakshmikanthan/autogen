"""Tic-Tac-Toe zero-sum game oracle for minimax/alphabeta verification (SYS-10, Module M07)."""

from __future__ import annotations

from typing import Sequence

# 3x3 board winning lines
WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class TicTacToeGame:
    """Canonical 3x3 zero-sum turn-based game. Solved value with optimal play is 0 (draw)."""

    def __init__(self, initial_board: tuple[str, ...] | None = None) -> None:
        self._initial_state = initial_board or tuple(" " for _ in range(9))

    @property
    def initial_state(self) -> tuple[str, ...]:
        return self._initial_state

    def player(self, state: tuple[str, ...]) -> int:
        x_count = state.count("X")
        o_count = state.count("O")
        return 0 if x_count == o_count else 1

    def legal_actions(self, state: tuple[str, ...]) -> list[int]:
        if self.is_terminal(state):
            return []
        return [i for i, cell in enumerate(state) if cell == " "]

    def successor(self, state: tuple[str, ...], action: int) -> tuple[str, ...]:
        p = "X" if self.player(state) == 0 else "O"
        board_list = list(state)
        board_list[action] = p
        return tuple(board_list)

    def is_terminal(self, state: tuple[str, ...]) -> bool:
        if self._check_winner(state) is not None:
            return True
        return " " not in state

    def utility(self, state: tuple[str, ...], player: int) -> float:
        winner = self._check_winner(state)
        if winner == "X":
            return 1.0 if player == 0 else -1.0
        elif winner == "O":
            return -1.0 if player == 0 else 1.0
        return 0.0  # Draw

    def chance_outcomes(self, state: tuple[str, ...], action: int) -> list[tuple[tuple[str, ...], float]]:
        return [(self.successor(state, action), 1.0)]

    def _check_winner(self, state: tuple[str, ...]) -> str | None:
        for a, b, c in WINNING_LINES:
            if state[a] != " " and state[a] == state[b] == state[c]:
                return state[a]
        return None

