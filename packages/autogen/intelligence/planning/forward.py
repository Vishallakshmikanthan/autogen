"""Forward state-space planning algorithms (BFS and A*) (SYS-08, Module M09)."""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

from autogen.intelligence.planning.contracts import (
    Fact,
    GroundAction,
    Plan,
    PlanningMetrics,
    PlanningProblem,
)
from autogen.intelligence.planning.heuristics import h_add, h_ff, h_max


@dataclass(order=True)
class _PriorityQueueEntry:
    priority: float
    sequence_id: int
    state: frozenset[Fact] = field(compare=False)
    actions: tuple[GroundAction, ...] = field(compare=False)
    g_cost: float = field(compare=False)


class ForwardStateSpacePlanner:
    """Forward state-space progression planner exploring states via BFS or A* search."""

    def __init__(
        self,
        heuristic_fn: Callable[[frozenset[Fact], PlanningProblem], float] | None = None,
    ) -> None:
        self.heuristic_fn = heuristic_fn or h_max

    def solve_bfs(self, problem: PlanningProblem) -> Plan:
        """Solves the planning problem via Breadth-First Search (guaranteeing shortest plan)."""
        t0 = time.perf_counter()

        if problem.is_goal(problem.initial_facts):
            return Plan(
                is_solved=True,
                actions=(),
                total_cost=0.0,
                metrics=PlanningMetrics(nodes_expanded=0, nodes_generated=1, runtime_sec=time.perf_counter() - t0),
            )

        queue: deque[tuple[frozenset[Fact], tuple[GroundAction, ...], float]] = deque(
            [(problem.initial_facts, (), 0.0)]
        )
        visited: set[frozenset[Fact]] = {problem.initial_facts}
        nodes_expanded = 0
        nodes_generated = 1

        while queue:
            current_state, path, g_cost = queue.popleft()
            nodes_expanded += 1

            for action in problem.actions:
                if action.is_applicable(current_state):
                    next_state = action.apply(current_state)
                    nodes_generated += 1

                    if problem.is_goal(next_state):
                        t1 = time.perf_counter()
                        return Plan(
                            is_solved=True,
                            actions=path + (action,),
                            total_cost=g_cost + action.cost,
                            metrics=PlanningMetrics(
                                nodes_expanded=nodes_expanded,
                                nodes_generated=nodes_generated,
                                runtime_sec=t1 - t0,
                            ),
                        )

                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + (action,), g_cost + action.cost))

        t1 = time.perf_counter()
        return Plan(
            is_solved=False,
            actions=(),
            total_cost=0.0,
            metrics=PlanningMetrics(
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                runtime_sec=t1 - t0,
            ),
            is_proven_unsolvable=True,
        )

    def solve_astar(
        self,
        problem: PlanningProblem,
        max_expansions: int = 10000,
    ) -> Plan:
        """Solves the planning problem using A* forward state-space search guided by heuristic."""
        t0 = time.perf_counter()

        initial_h = self.heuristic_fn(problem.initial_facts, problem)
        if math.isinf(initial_h):
            # Dead end certified by delete relaxation
            return Plan(
                is_solved=False,
                actions=(),
                total_cost=0.0,
                metrics=PlanningMetrics(nodes_expanded=0, nodes_generated=1, runtime_sec=time.perf_counter() - t0),
                is_proven_unsolvable=True,
            )

        if problem.is_goal(problem.initial_facts):
            return Plan(
                is_solved=True,
                actions=(),
                total_cost=0.0,
                metrics=PlanningMetrics(nodes_expanded=0, nodes_generated=1, runtime_sec=time.perf_counter() - t0),
            )

        sequence_id = 0
        frontier: list[_PriorityQueueEntry] = []
        heapq.heappush(
            frontier,
            _PriorityQueueEntry(
                priority=initial_h,
                sequence_id=sequence_id,
                state=problem.initial_facts,
                actions=(),
                g_cost=0.0,
            ),
        )

        best_g: dict[frozenset[Fact], float] = {problem.initial_facts: 0.0}
        nodes_expanded = 0
        nodes_generated = 1
        max_queue_size = 1

        while frontier and nodes_expanded < max_expansions:
            if len(frontier) > max_queue_size:
                max_queue_size = len(frontier)

            entry = heapq.heappop(frontier)
            state = entry.state
            path = entry.actions
            g = entry.g_cost

            if g > best_g.get(state, math.inf):
                continue

            if problem.is_goal(state):
                t1 = time.perf_counter()
                return Plan(
                    is_solved=True,
                    actions=path,
                    total_cost=g,
                    metrics=PlanningMetrics(
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        runtime_sec=t1 - t0,
                        max_queue_size=max_queue_size,
                    ),
                )

            nodes_expanded += 1

            for action in problem.actions:
                if action.is_applicable(state):
                    next_state = action.apply(state)
                    next_g = g + action.cost
                    nodes_generated += 1

                    if next_g < best_g.get(next_state, math.inf):
                        best_g[next_state] = next_g
                        h = self.heuristic_fn(next_state, problem)
                        if not math.isinf(h):
                            f = next_g + h
                            sequence_id += 1
                            heapq.heappush(
                                frontier,
                                _PriorityQueueEntry(
                                    priority=f,
                                    sequence_id=sequence_id,
                                    state=next_state,
                                    actions=path + (action,),
                                    g_cost=next_g,
                                ),
                            )

        t1 = time.perf_counter()
        return Plan(
            is_solved=False,
            actions=(),
            total_cost=0.0,
            metrics=PlanningMetrics(
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                runtime_sec=t1 - t0,
                max_queue_size=max_queue_size,
            ),
            is_proven_unsolvable=len(frontier) == 0,
        )
