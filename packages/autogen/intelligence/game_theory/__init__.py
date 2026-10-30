"""Game theory models, matrix games, solvers, and iterated tournaments (SYS-10, Module M07)."""

from autogen.intelligence.game_theory.contracts import (
    MixedStrategyProfile,
    NormalFormGame,
)
from autogen.intelligence.game_theory.iterated import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    IteratedStrategy,
    MatchResult,
    RandomStrategy,
    TitForTat,
    play_iterated_match,
    run_tournament,
)
from autogen.intelligence.game_theory.matrix_games import (
    make_coordination_game,
    make_hawk_dove,
    make_matching_pennies,
    make_prisoners_dilemma,
    make_stag_hunt,
)
from autogen.intelligence.game_theory.solvers import (
    find_best_responses,
    find_dominant_strategies,
    find_pure_nash_equilibria,
    solve_2x2_mixed_nash,
    solve_bimatrix_support_enumeration,
)
from autogen.intelligence.game_theory.verifier import (
    is_nash_equilibrium,
    verify_equilibrium_inequality,
)

__all__ = [
    "AlwaysCooperate",
    "AlwaysDefect",
    "GrimTrigger",
    "IteratedStrategy",
    "MatchResult",
    "MixedStrategyProfile",
    "NormalFormGame",
    "RandomStrategy",
    "TitForTat",
    "find_best_responses",
    "find_dominant_strategies",
    "find_pure_nash_equilibria",
    "is_nash_equilibrium",
    "make_coordination_game",
    "make_hawk_dove",
    "make_matching_pennies",
    "make_prisoners_dilemma",
    "make_stag_hunt",
    "play_iterated_match",
    "run_tournament",
    "solve_2x2_mixed_nash",
    "solve_bimatrix_support_enumeration",
    "verify_equilibrium_inequality",
]

