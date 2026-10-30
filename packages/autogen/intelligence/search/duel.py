"""Algorithm Duel runner: lockstep multi-algorithm search execution and step telemetry (SYS-06, Module M06)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Generic, TypeVar, cast

from autogen.intelligence.search.contracts import (
    SearchProblem,
    SearchResult,
    SearchStepEvent,
)

S = TypeVar("S")
A = TypeVar("A")


class AlgorithmDuelRunner(Generic[S, A]):
    """Orchestrates side-by-side search algorithm duels on identical problems."""

    def __init__(self, algorithms: dict[str, Any]) -> None:
        if not algorithms:
            raise ValueError("AlgorithmDuelRunner requires at least one algorithm")
        self.algorithms = algorithms

    def step_duel(
        self, problem: SearchProblem[S, A]
    ) -> Generator[dict[str, SearchStepEvent[S, A]], None, dict[str, SearchResult[S, A]]]:
        """Runs all registered algorithms in lockstep, yielding per-tick step event snapshots."""
        generators = {}
        results: dict[str, SearchResult[S, A]] = {}
        active_algorithms = set(self.algorithms.keys())

        # Initialize generators
        for name, algo in self.algorithms.items():
            if hasattr(algo, "step_solve"):
                generators[name] = algo.step_solve(problem)
            else:
                # Synchronous fallback: solve and wrap
                res = algo.solve(problem)
                results[name] = res
                active_algorithms.remove(name)

        step_counter = 0

        while active_algorithms:
            step_counter += 1
            combined_events: dict[str, SearchStepEvent[S, A]] = {}
            finished_in_this_step: list[str] = []

            for name in list(active_algorithms):
                gen = generators[name]
                try:
                    event = next(gen)
                    combined_events[name] = event
                except StopIteration as stop:
                    # Generator yielded final SearchResult as return value
                    res = stop.value
                    if res is None and hasattr(self.algorithms[name], "_final_result"):
                        res = self.algorithms[name]._final_result
                    results[name] = res
                    finished_in_this_step.append(name)

            for name in finished_in_this_step:
                active_algorithms.remove(name)

            if combined_events:
                yield combined_events

        return results

    def run_duel(self, problem: SearchProblem[S, A]) -> dict[str, SearchResult[S, A]]:
        """Synchronously executes the duel until all algorithms complete."""
        gen = self.step_duel(problem)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return cast(dict[str, SearchResult[S, A]], stop.value)
