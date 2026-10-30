"""Informed heuristic graph search algorithms implemented from scratch (SYS-06, Module M06)."""

from __future__ import annotations

import heapq
import time
from collections.abc import Generator
from typing import Generic, TypeVar

from autogen.intelligence.search.contracts import (
    SearchNode,
    SearchProblem,
    SearchResult,
    SearchStepEvent,
    reconstruct_path,
)

S = TypeVar("S")
A = TypeVar("A")


class GreedyBestFirstSearch(Generic[S, A]):
    """Greedy Best-First Search: expands node with lowest heuristic estimate f(n) = h(n)."""

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        for _ in self.step_solve(problem):
            pass
        return self._final_result

    def step_solve(
        self, problem: SearchProblem[S, A]
    ) -> Generator[SearchStepEvent[S, A], None, SearchResult[S, A]]:
        start_time = time.perf_counter()
        start = problem.initial_state

        if problem.is_goal(start):
            res: SearchResult[S, A] = SearchResult(
                success=True,
                path=[],
                states=[start],
                total_cost=0.0,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                runtime_seconds=time.perf_counter() - start_time,
            )
            self._final_result = res
            return res

        tie_breaker = 0
        h_start = problem.heuristic(start)
        frontier: list[tuple[float, int, SearchNode[S, A]]] = []
        root: SearchNode[S, A] = SearchNode(state=start, g_cost=0.0, h_cost=h_start, depth=0)
        heapq.heappush(frontier, (h_start, tie_breaker, root))

        visited: set[S] = set()
        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1
        step_idx = 0

        while frontier:
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

            _, _, current = heapq.heappop(frontier)

            if current.state in visited:
                continue
            visited.add(current.state)

            nodes_expanded += 1
            step_idx += 1

            yield SearchStepEvent(
                algorithm="GreedyBestFirst",
                step=step_idx,
                current_state=current.state,
                frontier_states=[item[2].state for item in frontier],
                explored_states=list(visited),
                f_cost=current.h_cost,
                g_cost=current.g_cost,
                nodes_expanded=nodes_expanded,
            )

            if problem.is_goal(current.state):
                actions, states = reconstruct_path(current)
                res = SearchResult(
                    success=True,
                    path=actions,
                    states=states,
                    total_cost=current.g_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier,
                    runtime_seconds=time.perf_counter() - start_time,
                )
                self._final_result = res
                return res

            for action, next_state, cost in problem.successors(current.state):
                if next_state not in visited:
                    nodes_generated += 1
                    tie_breaker += 1
                    h_next = problem.heuristic(next_state)
                    child = SearchNode(
                        state=next_state,
                        parent=current,
                        action=action,
                        g_cost=current.g_cost + cost,
                        h_cost=h_next,
                        depth=current.depth + 1,
                    )
                    heapq.heappush(frontier, (h_next, tie_breaker, child))

        res = SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier,
            runtime_seconds=time.perf_counter() - start_time,
        )
        self._final_result = res
        return res


class AStarSearch(Generic[S, A]):
    """A* Graph Search (f(n) = g(n) + h(n)) with reopening support for non-monotonic heuristics."""

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        for _ in self.step_solve(problem):
            pass
        return self._final_result

    def step_solve(
        self, problem: SearchProblem[S, A]
    ) -> Generator[SearchStepEvent[S, A], None, SearchResult[S, A]]:
        start_time = time.perf_counter()
        start = problem.initial_state

        if problem.is_goal(start):
            res: SearchResult[S, A] = SearchResult(
                success=True,
                path=[],
                states=[start],
                total_cost=0.0,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                runtime_seconds=time.perf_counter() - start_time,
            )
            self._final_result = res
            return res

        tie_breaker = 0
        h_start = problem.heuristic(start)
        # Priority tuple: (rounded_f, -g_cost (tie-break toward goal), tie_breaker, node)
        frontier: list[tuple[float, float, int, SearchNode[S, A]]] = []
        root: SearchNode[S, A] = SearchNode(state=start, g_cost=0.0, h_cost=h_start, depth=0)
        heapq.heappush(frontier, (round(root.f_cost, 10), -root.g_cost, tie_breaker, root))

        best_g: dict[S, float] = {start: 0.0}
        closed_set: set[S] = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1
        step_idx = 0

        while frontier:
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

            _, _, _, current = heapq.heappop(frontier)

            # Lazy deletion check
            if current.state in closed_set:
                continue
            if current.g_cost > best_g.get(current.state, float("inf")):
                continue

            closed_set.add(current.state)
            nodes_expanded += 1
            step_idx += 1

            yield SearchStepEvent(
                algorithm="A*",
                step=step_idx,
                current_state=current.state,
                frontier_states=[item[3].state for item in frontier],
                explored_states=list(closed_set),
                f_cost=current.f_cost,
                g_cost=current.g_cost,
                nodes_expanded=nodes_expanded,
            )

            # Goal check on pop guarantees optimality
            if problem.is_goal(current.state):
                actions, states = reconstruct_path(current)
                res = SearchResult(
                    success=True,
                    path=actions,
                    states=states,
                    total_cost=current.g_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier,
                    runtime_seconds=time.perf_counter() - start_time,
                )
                self._final_result = res
                return res

            for action, next_state, cost in problem.successors(current.state):
                new_g = current.g_cost + cost

                # If next_state is in closed_set but new_g is cheaper, reopen it
                if next_state in closed_set and new_g < best_g[next_state]:
                    closed_set.remove(next_state)

                if next_state not in best_g or new_g < best_g[next_state]:
                    best_g[next_state] = new_g
                    nodes_generated += 1
                    tie_breaker += 1
                    h_next = problem.heuristic(next_state)
                    child = SearchNode(
                        state=next_state,
                        parent=current,
                        action=action,
                        g_cost=new_g,
                        h_cost=h_next,
                        depth=current.depth + 1,
                    )
                    heapq.heappush(
                        frontier,
                        (round(child.f_cost, 10), -child.g_cost, tie_breaker, child),
                    )

        res = SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier,
            runtime_seconds=time.perf_counter() - start_time,
        )
        self._final_result = res
        return res


class WeightedAStarSearch(Generic[S, A]):
    """Weighted A* Search (f_w(n) = g(n) + w * h(n)) with bounded suboptimality C <= w * C*."""

    def __init__(self, weight: float = 1.5) -> None:
        if weight < 1.0:
            raise ValueError(f"Weighted A* requires weight w >= 1.0, got {weight}")
        self.weight = weight

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        for _ in self.step_solve(problem):
            pass
        return self._final_result

    def step_solve(
        self, problem: SearchProblem[S, A]
    ) -> Generator[SearchStepEvent[S, A], None, SearchResult[S, A]]:
        start_time = time.perf_counter()
        start = problem.initial_state

        if problem.is_goal(start):
            res: SearchResult[S, A] = SearchResult(
                success=True,
                path=[],
                states=[start],
                total_cost=0.0,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                runtime_seconds=time.perf_counter() - start_time,
            )
            self._final_result = res
            return res

        tie_breaker = 0
        h_start = problem.heuristic(start)
        frontier: list[tuple[float, float, int, SearchNode[S, A]]] = []
        root: SearchNode[S, A] = SearchNode(state=start, g_cost=0.0, h_cost=h_start, depth=0)
        f_init = root.g_cost + self.weight * root.h_cost
        heapq.heappush(frontier, (round(f_init, 10), -root.g_cost, tie_breaker, root))

        best_g: dict[S, float] = {start: 0.0}
        closed_set: set[S] = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1
        step_idx = 0

        while frontier:
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

            _, _, _, current = heapq.heappop(frontier)

            if current.state in closed_set:
                continue
            if current.g_cost > best_g.get(current.state, float("inf")):
                continue

            closed_set.add(current.state)
            nodes_expanded += 1
            step_idx += 1

            yield SearchStepEvent(
                algorithm=f"WeightedA*(w={self.weight})",
                step=step_idx,
                current_state=current.state,
                frontier_states=[item[3].state for item in frontier],
                explored_states=list(closed_set),
                f_cost=current.g_cost + self.weight * current.h_cost,
                g_cost=current.g_cost,
                nodes_expanded=nodes_expanded,
            )

            if problem.is_goal(current.state):
                actions, states = reconstruct_path(current)
                res = SearchResult(
                    success=True,
                    path=actions,
                    states=states,
                    total_cost=current.g_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier,
                    runtime_seconds=time.perf_counter() - start_time,
                )
                self._final_result = res
                return res

            for action, next_state, cost in problem.successors(current.state):
                new_g = current.g_cost + cost

                if next_state in closed_set and new_g < best_g[next_state]:
                    closed_set.remove(next_state)

                if next_state not in best_g or new_g < best_g[next_state]:
                    best_g[next_state] = new_g
                    nodes_generated += 1
                    tie_breaker += 1
                    h_next = problem.heuristic(next_state)
                    child = SearchNode(
                        state=next_state,
                        parent=current,
                        action=action,
                        g_cost=new_g,
                        h_cost=h_next,
                        depth=current.depth + 1,
                    )
                    f_child = new_g + self.weight * h_next
                    heapq.heappush(
                        frontier,
                        (round(f_child, 10), -child.g_cost, tie_breaker, child),
                    )

        res = SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier,
            runtime_seconds=time.perf_counter() - start_time,
        )
        self._final_result = res
        return res


class IterativeDeepeningAStar(Generic[S, A]):
    """Iterative Deepening A* (IDA*): low-memory heuristic depth search."""

    def __init__(self, max_iterations: int = 1000) -> None:
        self.max_iterations = max_iterations

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        start = problem.initial_state
        bound = problem.heuristic(start)

        nodes_expanded = 0
        nodes_generated = 1

        root: SearchNode[S, A] = SearchNode(state=start, g_cost=0.0, h_cost=bound, depth=0)
        path: list[SearchNode[S, A]] = [root]
        path_states: set[S] = {start}
        visited_g: dict[S, float] = {start: 0.0}

        def _search(g: float, threshold: float) -> tuple[SearchNode[S, A] | None, float]:
            nonlocal nodes_expanded, nodes_generated
            current = path[-1]
            f = g + current.h_cost

            if f > threshold:
                return None, f
            if problem.is_goal(current.state):
                return current, threshold

            min_excess = float("inf")
            nodes_expanded += 1

            for action, next_state, cost in problem.successors(current.state):
                if next_state in path_states:
                    continue

                next_g = g + cost
                if next_state in visited_g and visited_g[next_state] <= next_g:
                    continue
                visited_g[next_state] = next_g

                nodes_generated += 1
                h_next = problem.heuristic(next_state)
                child = SearchNode(
                    state=next_state,
                    parent=current,
                    action=action,
                    g_cost=next_g,
                    h_cost=h_next,
                    depth=current.depth + 1,
                )
                path.append(child)
                path_states.add(next_state)

                found_node, excess = _search(next_g, threshold)
                if found_node is not None:
                    return found_node, threshold
                if excess < min_excess:
                    min_excess = excess

                path_states.remove(next_state)
                path.pop()

            return None, min_excess

        for _ in range(self.max_iterations):
            visited_g = {start: 0.0}
            found_node, next_threshold = _search(0.0, bound)
            if found_node is not None:
                actions, states = reconstruct_path(found_node)
                return SearchResult(
                    success=True,
                    path=actions,
                    states=states,
                    total_cost=found_node.g_cost,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=len(path),
                    runtime_seconds=time.perf_counter() - start_time,
                )

            if next_threshold == float("inf"):
                # Unreachable
                break
            bound = next_threshold

        return SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=len(path),
            runtime_seconds=time.perf_counter() - start_time,
        )
