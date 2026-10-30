"""Iterated game tournament and repeated interaction tests (Module M07)."""

from __future__ import annotations

import pytest

from autogen.intelligence.game_theory.iterated import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    TitForTat,
    play_iterated_match,
    run_tournament,
)
from autogen.intelligence.game_theory.matrix_games import make_prisoners_dilemma


def test_tit_for_tat_vs_always_defect() -> None:
    """Tit-for-Tat cooperates in round 0, gets suckered, then defects for all remaining rounds."""
    game = make_prisoners_dilemma()
    res = play_iterated_match(game, TitForTat(), AlwaysDefect(), num_rounds=10)

    # Round 0: (C, D) -> Payoff (0, 5)
    # Rounds 1..9: (D, D) -> Payoff (1, 1) * 9 = (9, 9)
    # Total: (9, 14)
    assert res.history_1 == [0] + [1] * 9
    assert res.history_2 == [1] * 10
    assert res.total_payoff_1 == 9.0
    assert res.total_payoff_2 == 14.0


def test_tit_for_tat_vs_tit_for_tat() -> None:
    """Two Tit-for-Tat strategies maintain perpetual mutual cooperation."""
    game = make_prisoners_dilemma()
    res = play_iterated_match(game, TitForTat(), TitForTat(), num_rounds=20)

    # All rounds: (C, C) -> Payoff (3, 3) * 20 = (60, 60)
    assert res.history_1 == [0] * 20
    assert res.history_2 == [0] * 20
    assert res.total_payoff_1 == 60.0
    assert res.total_payoff_2 == 60.0


def test_grim_trigger_behavior() -> None:
    """Grim Trigger cooperates until a single defection occurs, then defects permanently."""
    game = make_prisoners_dilemma()
    grim = GrimTrigger()
    defect = AlwaysDefect()

    res = play_iterated_match(game, grim, defect, num_rounds=5)
    assert res.history_1 == [0, 1, 1, 1, 1]


def test_iterated_prisoners_dilemma_tournament() -> None:
    """Verifies that running a tournament produces sensible average payoffs."""
    game = make_prisoners_dilemma()
    strats = [TitForTat(), GrimTrigger(), AlwaysCooperate(), AlwaysDefect()]

    scores = run_tournament(game, strats, num_rounds=50)

    # In a community with multiple cooperative strategies, TitForTat and GrimTrigger score highly
    assert "TitForTat" in scores
    assert "AlwaysDefect" in scores
    assert scores["TitForTat"] > scores["AlwaysCooperate"]

