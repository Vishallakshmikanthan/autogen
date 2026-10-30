"""Tic-Tac-Toe solved value oracle and adversarial endgame verification (Module M07)."""

from __future__ import annotations

import pytest

from autogen.intelligence.adversarial.minimax import AlphaBetaSearch, MinimaxSearch
from autogen.intelligence.adversarial.tictactoe import TicTacToeGame


def test_tictactoe_solved_value_is_draw() -> None:
    """Oracle verification: The game-theoretic value of Tic-Tac-Toe with optimal play is 0 (draw)."""
    game = TicTacToeGame()
    alphabeta = AlphaBetaSearch[tuple[str, ...], int]()

    dummy_eval = lambda s, p: 0.0

    # Search full tree to terminal leaves (depth 9)
    result = alphabeta.search(game, game.initial_state, max_depth=9, eval_fn=dummy_eval)

    # Invariant: Value from start board must be exactly 0.0 (draw)
    assert result.expected_value == 0.0
    # Must pick an initial move (e.g. center or corner)
    assert result.best_action in range(9)


def test_tictactoe_immediate_win_detection() -> None:
    """Verifies that AlphaBeta takes an immediate winning action when available."""
    # Board where X can win immediately by playing index 2
    # X X .
    # O O .
    # . . .
    board = (
        "X", "X", " ",
        "O", "O", " ",
        " ", " ", " ",
    )
    game = TicTacToeGame(initial_board=board)
    alphabeta = AlphaBetaSearch[tuple[str, ...], int]()

    res = alphabeta.search(game, board, max_depth=5, eval_fn=lambda s, p: 0.0)

    # Must choose action 2 to win
    assert res.best_action == 2
    assert res.expected_value == 1.0


def test_tictactoe_immediate_block_detection() -> None:
    """Verifies that O blocks X's imminent win."""
    # Board where X has two in a row and it is O's turn (count X=2, count O=1)
    # X X .
    # O . .
    # . . .
    board = (
        "X", "X", " ",
        "O", " ", " ",
        " ", " ", " ",
    )
    game = TicTacToeGame(initial_board=board)
    alphabeta = AlphaBetaSearch[tuple[str, ...], int]()

    # O is MIN player (player 1)
    res = alphabeta.search(game, board, max_depth=5, eval_fn=lambda s, p: 0.0)

    # O must choose 2 to prevent X from winning
    assert res.best_action == 2

