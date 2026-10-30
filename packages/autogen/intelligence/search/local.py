"""Local and stochastic search algorithms implemented from scratch (SYS-06, Module M06)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

from autogen.intelligence.search.contracts import (
    SearchNode,
    SearchProblem,
    SearchResult,
    reconstruct_path,
)

S = TypeVar("S")
A = TypeVar("A")


class HillClimbingSearch(Generic[S, A]):
    """Steepest-Ascent Hill Climbing with optional random restarts."""

    def __init__(self, max_restarts: int = 5, max_steps: int = 1000) -> None:
        self.max_restarts = max_restarts
        self.max_steps = max_steps

    def solve(
        self,
        problem: SearchProblem[S, A],
        restart_sampler: Callable[[], S] | None = None,
    ) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        nodes_expanded = 0
        nodes_generated = 1

        best_overall_node: SearchNode[S, A] | None = None
        best_overall_h = float("inf")

        for _restart in range(self.max_restarts + 1):
            if _restart == 0 or restart_sampler is None:
                current_state = problem.initial_state
            else:
                current_state = restart_sampler()

            h_curr = problem.heuristic(current_state)
            current_node: SearchNode[S, A] = SearchNode(
                state=current_state,
                g_cost=0.0,
                h_cost=h_curr,
                depth=0,
            )

            if h_curr < best_overall_h:
                best_overall_h = h_curr
                best_overall_node = current_node

            if problem.is_goal(current_state):
                actions, states = reconstruct_path(current_node)
                return SearchResult(
                    success=True,
                    path=actions,
                    states=states,
                    total_cost=current_node.g_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=1,
                    runtime_seconds=time.perf_counter() - start_time,
                )

            for _ in range(self.max_steps):
                nodes_expanded += 1
                successors = problem.successors(current_node.state)
                if not successors:
                    break

                best_succ: tuple[A, S, float, float] | None = None
                best_succ_h = h_curr

                for action, next_state, cost in successors:
                    nodes_generated += 1
                    h_next = problem.heuristic(next_state)
                    if h_next < best_succ_h:
                        best_succ_h = h_next
                        best_succ = (action, next_state, cost, h_next)

                if best_succ is None:
                    # Trapped on local minimum or plateau
                    break

                act, nxt_state, step_cost, nxt_h = best_succ
                current_node = SearchNode(
                    state=nxt_state,
                    parent=current_node,
                    action=act,
                    g_cost=current_node.g_cost + step_cost,
                    h_cost=nxt_h,
                    depth=current_node.depth + 1,
                )
                h_curr = nxt_h

                if h_curr < best_overall_h:
                    best_overall_h = h_curr
                    best_overall_node = current_node

                if problem.is_goal(nxt_state):
                    actions, states = reconstruct_path(current_node)
                    return SearchResult(
                        success=True,
                        path=actions,
                        states=states,
                        total_cost=current_node.g_cost,
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        max_frontier_size=1,
                        runtime_seconds=time.perf_counter() - start_time,
                    )

        # Return best attained state
        actions, states = reconstruct_path(best_overall_node) if best_overall_node else ([], [])
        return SearchResult(
            success=False,
            path=actions,
            states=states,
            total_cost=best_overall_node.g_cost if best_overall_node else float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=1,
            runtime_seconds=time.perf_counter() - start_time,
        )


class SimulatedAnnealingSearch(Generic[S, A]):
    """Simulated Annealing: accepts worse transitions with probability P = exp(-dE / T)."""

    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 1e-4,
        max_steps: int = 5000,
        schedule: str = "geometric",
        seed: int = 42,
    ) -> None:
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_steps = max_steps
        self.schedule = schedule
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _cool(self, temp: float, step: int) -> float:
        if self.schedule == "geometric":
            return temp * self.cooling_rate
        if self.schedule == "linear":
            return max(self.min_temperature, self.initial_temperature - 0.05 * step)
        if self.schedule == "logarithmic":
            return self.initial_temperature / (1.0 + math.log(1.0 + step))
        return temp * self.cooling_rate

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        start = problem.initial_state
        current_node: SearchNode[S, A] = SearchNode(
            state=start,
            g_cost=0.0,
            h_cost=problem.heuristic(start),
            depth=0,
        )

        nodes_expanded = 0
        nodes_generated = 1
        temp = self.initial_temperature

        if problem.is_goal(start):
            return SearchResult(
                success=True,
                path=[],
                states=[start],
                total_cost=0.0,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                runtime_seconds=time.perf_counter() - start_time,
            )

        for step in range(1, self.max_steps + 1):
            if temp <= self.min_temperature:
                break

            nodes_expanded += 1
            successors = problem.successors(current_node.state)
            if not successors:
                break

            # Pick a random successor
            idx = int(self.rng.integers(0, len(successors)))
            action, next_state, step_cost = successors[idx]
            nodes_generated += 1

            next_h = problem.heuristic(next_state)
            delta_e = next_h - current_node.h_cost  # Minimizing heuristic energy

            # Transition decision
            accept = False
            if delta_e < 0:
                accept = True
            else:
                p_accept = math.exp(-delta_e / max(temp, 1e-12))
                accept = bool(self.rng.random() < p_accept)

            if accept:
                current_node = SearchNode(
                    state=next_state,
                    parent=current_node,
                    action=action,
                    g_cost=current_node.g_cost + step_cost,
                    h_cost=next_h,
                    depth=current_node.depth + 1,
                )

                if problem.is_goal(next_state):
                    actions, states = reconstruct_path(current_node)
                    return SearchResult(
                        success=True,
                        path=actions,
                        states=states,
                        total_cost=current_node.g_cost,
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        max_frontier_size=1,
                        runtime_seconds=time.perf_counter() - start_time,
                    )

            temp = self._cool(temp, step)

        actions, states = reconstruct_path(current_node)
        return SearchResult(
            success=False,
            path=actions,
            states=states,
            total_cost=current_node.g_cost,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=1,
            runtime_seconds=time.perf_counter() - start_time,
        )


class LocalBeamSearch(Generic[S, A]):
    """Local Beam Search retaining k best parallel candidate nodes."""

    def __init__(self, beam_width: int = 3, max_steps: int = 500) -> None:
        if beam_width < 1:
            raise ValueError(f"Beam width must be >= 1, got {beam_width}")
        self.beam_width = beam_width
        self.max_steps = max_steps

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        start = problem.initial_state

        root: SearchNode[S, A] = SearchNode(
            state=start,
            g_cost=0.0,
            h_cost=problem.heuristic(start),
            depth=0,
        )
        if problem.is_goal(start):
            return SearchResult(
                success=True,
                path=[],
                states=[start],
                total_cost=0.0,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                runtime_seconds=time.perf_counter() - start_time,
            )

        beam: list[SearchNode[S, A]] = [root]
        nodes_expanded = 0
        nodes_generated = 1

        for _ in range(self.max_steps):
            candidates: list[SearchNode[S, A]] = []

            for node in beam:
                nodes_expanded += 1
                for action, next_state, cost in problem.successors(node.state):
                    nodes_generated += 1
                    child = SearchNode(
                        state=next_state,
                        parent=node,
                        action=action,
                        g_cost=node.g_cost + cost,
                        h_cost=problem.heuristic(next_state),
                        depth=node.depth + 1,
                    )
                    if problem.is_goal(next_state):
                        actions, states = reconstruct_path(child)
                        return SearchResult(
                            success=True,
                            path=actions,
                            states=states,
                            total_cost=child.g_cost,
                            nodes_expanded=nodes_expanded,
                            nodes_generated=nodes_generated,
                            max_frontier_size=self.beam_width,
                            runtime_seconds=time.perf_counter() - start_time,
                        )
                    candidates.append(child)

            if not candidates:
                break

            # Sort candidates by heuristic evaluation and retain top k
            candidates.sort(key=lambda n: n.h_cost)
            beam = candidates[: self.beam_width]

        best_node = min(beam, key=lambda n: n.h_cost) if beam else root
        actions, states = reconstruct_path(best_node)
        return SearchResult(
            success=False,
            path=actions,
            states=states,
            total_cost=best_node.g_cost,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=self.beam_width,
            runtime_seconds=time.perf_counter() - start_time,
        )
