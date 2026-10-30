"""Iterated matrix games, repeated interactions, and tournament simulations (SYS-10, Module M07)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Sequence

from autogen.intelligence.game_theory.contracts import NormalFormGame


class IteratedStrategy(ABC):
    """Abstract base class for repeated normal-form game decision strategies."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        """Resets strategy memory for a new repeated interaction episode."""
        ...

    @abstractmethod
    def choose_action(
        self,
        round_num: int,
        my_history: list[int],
        opp_history: list[int],
    ) -> int:
        """Selects action index for the current round based on past interaction history."""
        ...


class AlwaysCooperate(IteratedStrategy):
    """Unconditionally cooperates (action 0)."""

    def __init__(self) -> None:
        super().__init__("AlwaysCooperate")

    def reset(self) -> None:
        pass

    def choose_action(self, round_num: int, my_history: list[int], opp_history: list[int]) -> int:
        return 0


class AlwaysDefect(IteratedStrategy):
    """Unconditionally defects (action 1)."""

    def __init__(self) -> None:
        super().__init__("AlwaysDefect")

    def reset(self) -> None:
        pass

    def choose_action(self, round_num: int, my_history: list[int], opp_history: list[int]) -> int:
        return 1


class TitForTat(IteratedStrategy):
    """Cooperates on round 0, then duplicates the opponent's previous action."""

    def __init__(self) -> None:
        super().__init__("TitForTat")

    def reset(self) -> None:
        pass

    def choose_action(self, round_num: int, my_history: list[int], opp_history: list[int]) -> int:
        if round_num == 0 or not opp_history:
            return 0  # Cooperate
        return opp_history[-1]


class GrimTrigger(IteratedStrategy):
    """Cooperates until the opponent defects once, then defects permanently."""

    def __init__(self) -> None:
        super().__init__("GrimTrigger")
        self._triggered = False

    def reset(self) -> None:
        self._triggered = False

    def choose_action(self, round_num: int, my_history: list[int], opp_history: list[int]) -> int:
        if self._triggered:
            return 1
        if opp_history and opp_history[-1] != 0:
            self._triggered = True
            return 1
        return 0


class RandomStrategy(IteratedStrategy):
    """Plays an action drawn from a seeded random distribution."""

    def __init__(self, seed: int = 42, cooperate_prob: float = 0.5) -> None:
        super().__init__("RandomStrategy")
        self.cooperate_prob = cooperate_prob
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def choose_action(self, round_num: int, my_history: list[int], opp_history: list[int]) -> int:
        return 0 if self.rng.random() < self.cooperate_prob else 1


@dataclass(slots=True)
class MatchResult:
    """Outcome of an iterated matchup between two strategies."""

    strategy_1: str
    strategy_2: str
    rounds: int
    total_payoff_1: float
    total_payoff_2: float
    mean_payoff_1: float
    mean_payoff_2: float
    history_1: list[int] = field(default_factory=list)
    history_2: list[int] = field(default_factory=list)


def play_iterated_match(
    game: NormalFormGame,
    strat_1: IteratedStrategy,
    strat_2: IteratedStrategy,
    num_rounds: int = 100,
) -> MatchResult:
    """Executes a 2-player repeated normal-form game for num_rounds."""
    strat_1.reset()
    strat_2.reset()

    hist_1: list[int] = []
    hist_2: list[int] = []
    payoff_1 = 0.0
    payoff_2 = 0.0

    for r in range(num_rounds):
        a1 = strat_1.choose_action(r, hist_1, hist_2)
        a2 = strat_2.choose_action(r, hist_2, hist_1)

        hist_1.append(a1)
        hist_2.append(a2)

        payoff_1 += game.payoff(0, (a1, a2))
        payoff_2 += game.payoff(1, (a1, a2))

    return MatchResult(
        strategy_1=strat_1.name,
        strategy_2=strat_2.name,
        rounds=num_rounds,
        total_payoff_1=payoff_1,
        total_payoff_2=payoff_2,
        mean_payoff_1=payoff_1 / num_rounds,
        mean_payoff_2=payoff_2 / num_rounds,
        history_1=hist_1,
        history_2=hist_2,
    )


def run_tournament(
    game: NormalFormGame,
    strategies: Sequence[IteratedStrategy],
    num_rounds: int = 100,
) -> dict[str, float]:
    """Runs a round-robin tournament where every strategy plays against all others (and itself)."""
    scores: dict[str, float] = {s.name: 0.0 for s in strategies}

    for s1 in strategies:
        for s2 in strategies:
            res = play_iterated_match(game, s1, s2, num_rounds=num_rounds)
            scores[s1.name] += res.mean_payoff_1

    # Average score across all matchups
    total_matches = len(strategies)
    return {name: round(total / total_matches, 4) for name, total in scores.items()}

