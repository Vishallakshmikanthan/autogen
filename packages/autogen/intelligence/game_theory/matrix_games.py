"""Canonical normal-form matrix games for strategic interaction and resource competition (SYS-10, Module M07)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.game_theory.contracts import NormalFormGame


def make_prisoners_dilemma(
    reward: float = 3.0,
    sucker: float = 0.0,
    temptation: float = 5.0,
    penalty: float = 1.0,
) -> NormalFormGame:
    """Canonical Prisoner's Dilemma game.

    Standard condition: Temptation (5) > Reward (3) > Penalty (1) > Sucker (0).
    Unique pure Nash equilibrium: (Defect, Defect) with payoff (1, 1).
    """
    # Actions: [0: Cooperate, 1: Defect]
    u1 = np.array([
        [reward, sucker],
        [temptation, penalty],
    ], dtype=np.float64)

    u2 = np.array([
        [reward, temptation],
        [sucker, penalty],
    ], dtype=np.float64)

    return NormalFormGame(
        players=["Player 1", "Player 2"],
        actions=[["Cooperate", "Defect"], ["Cooperate", "Defect"]],
        payoff_matrices=[u1, u2],
    )


def make_hawk_dove(
    resource_value: float = 4.0,
    injury_cost: float = 6.0,
) -> NormalFormGame:
    """Hawk-Dove / Chicken game modeling evolutionary resource competition.

    Actions: [0: Hawk (Aggressive), 1: Dove (Passive)]
    Payoffs:
      Hawk vs Hawk: ((V-C)/2, (V-C)/2)
      Hawk vs Dove: (V, 0)
      Dove vs Hawk: (0, V)
      Dove vs Dove: (V/2, V/2)
    """
    v = resource_value
    c = injury_cost

    hh = (v - c) / 2.0
    hd = v
    dh = 0.0
    dd = v / 2.0

    u1 = np.array([
        [hh, hd],
        [dh, dd],
    ], dtype=np.float64)

    u2 = np.array([
        [hh, dh],
        [hd, dd],
    ], dtype=np.float64)

    return NormalFormGame(
        players=["Player 1", "Player 2"],
        actions=[["Hawk", "Dove"], ["Hawk", "Dove"]],
        payoff_matrices=[u1, u2],
    )


def make_stag_hunt(
    stag_reward: float = 5.0,
    hare_reward: float = 3.0,
    loss_when_solo_stag: float = 0.0,
) -> NormalFormGame:
    """Stag Hunt coordination game modeling trust and social cooperation.

    Payoffs:
      Stag vs Stag: (5, 5)  (Payoff-dominant Nash)
      Stag vs Hare: (0, 3)
      Hare vs Stag: (3, 0)
      Hare vs Hare: (3, 3)  (Risk-dominant Nash)
    """
    u1 = np.array([
        [stag_reward, loss_when_solo_stag],
        [hare_reward, hare_reward],
    ], dtype=np.float64)

    u2 = np.array([
        [stag_reward, hare_reward],
        [loss_when_solo_stag, hare_reward],
    ], dtype=np.float64)

    return NormalFormGame(
        players=["Player 1", "Player 2"],
        actions=[["Stag", "Hare"], ["Stag", "Hare"]],
        payoff_matrices=[u1, u2],
    )


def make_matching_pennies() -> NormalFormGame:
    """Matching Pennies zero-sum game.

    Actions: [0: Heads, 1: Tails]
    Player 1 wins +1 if both choose the same.
    Player 2 wins +1 if choices differ.
    Unique Nash equilibrium is strictly mixed: ((0.5, 0.5), (0.5, 0.5)).
    """
    u1 = np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ], dtype=np.float64)

    u2 = np.array([
        [-1.0, 1.0],
        [1.0, -1.0],
    ], dtype=np.float64)

    return NormalFormGame(
        players=["Player 1", "Player 2"],
        actions=[["Heads", "Tails"], ["Heads", "Tails"]],
        payoff_matrices=[u1, u2],
    )


def make_coordination_game() -> NormalFormGame:
    """Pure Coordination Game with two mutually beneficial pure Nash equilibria."""
    u1 = np.array([
        [4.0, 0.0],
        [0.0, 2.0],
    ], dtype=np.float64)

    u2 = np.array([
        [4.0, 0.0],
        [0.0, 2.0],
    ], dtype=np.float64)

    return NormalFormGame(
        players=["Player 1", "Player 2"],
        actions=[["Option A", "Option B"], ["Option A", "Option B"]],
        payoff_matrices=[u1, u2],
    )

