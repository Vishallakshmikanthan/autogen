"""Uninformed graph and tree search algorithms implemented from scratch (SYS-06, Module M06)."""

from __future__ import annotations

import heapq
import time
from collections import deque
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


class BreadthFirstSearch(Generic[S, A]):
    """Breadth-First Graph Search (FIFO queue, visited set, optimal on unweighted graphs)."""

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

        frontier: deque[SearchNode[S, A]] = deque([SearchNode(state=start, g_cost=0.0, depth=0)])
        visited: set[S] = {start}
        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1
        step_idx = 0

        while frontier:
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

            current = frontier.popleft()
            nodes_expanded += 1
            step_idx += 1

            yield SearchStepEvent(
                algorithm="BFS",
                step=step_idx,
                current_state=current.state,
                frontier_states=[n.state for n in frontier],
                explored_states=list(visited),
                f_cost=float(current.depth),
                g_cost=current.g_cost,
                nodes_expanded=nodes_expanded,
            )

            for action, next_state, cost in problem.successors(current.state):
                if next_state not in visited:
                    nodes_generated += 1
                    child = SearchNode(
                        state=next_state,
                        parent=current,
                        action=action,
                        g_cost=current.g_cost + cost,
                        depth=current.depth + 1,
                    )

                    if problem.is_goal(next_state):
                        actions, states = reconstruct_path(child)
                        res = SearchResult(
                            success=True,
                            path=actions,
                            states=states,
                            total_cost=child.g_cost,
                            nodes_expanded=nodes_expanded,
                            nodes_generated=nodes_generated,
                            max_frontier_size=max_frontier,
                            runtime_seconds=time.perf_counter() - start_time,
                        )
                        self._final_result = res
                        return res

                    visited.add(next_state)
                    frontier.append(child)

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


class DepthFirstSearch(Generic[S, A]):
    """Depth-First Search (iterative stack with cycle detection and depth limit)."""

    def __init__(self, max_depth: int = 10_000) -> None:
        self.max_depth = max_depth

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

        frontier: list[SearchNode[S, A]] = [SearchNode(state=start, g_cost=0.0, depth=0)]
        visited: set[S] = set()
        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1
        step_idx = 0

        while frontier:
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

            current = frontier.pop()

            if current.state in visited:
                continue
            visited.add(current.state)

            nodes_expanded += 1
            step_idx += 1

            yield SearchStepEvent(
                algorithm="DFS",
                step=step_idx,
                current_state=current.state,
                frontier_states=[n.state for n in frontier],
                explored_states=list(visited),
                f_cost=float(current.depth),
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

            if current.depth < self.max_depth:
                for action, next_state, cost in reversed(problem.successors(current.state)):
                    if next_state not in visited:
                        nodes_generated += 1
                        child = SearchNode(
                            state=next_state,
                            parent=current,
                            action=action,
                            g_cost=current.g_cost + cost,
                            depth=current.depth + 1,
                        )
                        frontier.append(child)

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


class DepthLimitedSearch(Generic[S, A]):
    """Depth-Limited Search returning cutoff status when depth limit is exhausted."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.cutoff_occurred = False

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        self.cutoff_occurred = False
        nodes_expanded = 0
        nodes_generated = 1
        visited_depth: dict[S, int] = {problem.initial_state: 0}

        def _recursive_dls(
            node: SearchNode[S, A], path_set: set[S]
        ) -> tuple[SearchNode[S, A] | None, bool]:
            nonlocal nodes_expanded, nodes_generated
            if problem.is_goal(node.state):
                return node, False

            if node.depth >= self.limit:
                return None, True  # Cutoff

            cutoff = False
            nodes_expanded += 1

            for action, next_state, cost in problem.successors(node.state):
                if next_state in path_set:
                    continue  # Cycle prevention on active branch
                if next_state in visited_depth and visited_depth[next_state] <= node.depth + 1:
                    continue
                visited_depth[next_state] = node.depth + 1

                nodes_generated += 1
                child = SearchNode(
                    state=next_state,
                    parent=node,
                    action=action,
                    g_cost=node.g_cost + cost,
                    depth=node.depth + 1,
                )
                path_set.add(next_state)
                result_node, child_cutoff = _recursive_dls(child, path_set)
                path_set.remove(next_state)

                if result_node is not None:
                    return result_node, False
                if child_cutoff:
                    cutoff = True

            return None, cutoff

        root: SearchNode[S, A] = SearchNode(state=problem.initial_state, g_cost=0.0, depth=0)
        found_node, cutoff = _recursive_dls(root, {root.state})
        self.cutoff_occurred = cutoff

        if found_node is not None:
            actions, states = reconstruct_path(found_node)
            return SearchResult(
                success=True,
                path=actions,
                states=states,
                total_cost=found_node.g_cost,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=self.limit,
                runtime_seconds=time.perf_counter() - start_time,
            )

        return SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=self.limit,
            runtime_seconds=time.perf_counter() - start_time,
        )


class IterativeDeepeningSearch(Generic[S, A]):
    """Iterative Deepening Search: successive DLS from depth 0 to max_depth."""

    def __init__(self, max_depth: int = 100) -> None:
        self.max_depth = max_depth

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        total_expanded = 0
        total_generated = 0
        max_frontier = 0

        for depth in range(self.max_depth + 1):
            dls = DepthLimitedSearch[S, A](limit=depth)
            result = dls.solve(problem)
            total_expanded += result.nodes_expanded
            total_generated += result.nodes_generated
            max_frontier = max(max_frontier, result.max_frontier_size)

            if result.success:
                return SearchResult(
                    success=True,
                    path=result.path,
                    states=result.states,
                    total_cost=result.total_cost,
                    nodes_expanded=total_expanded,
                    nodes_generated=total_generated,
                    max_frontier_size=max_frontier,
                    runtime_seconds=time.perf_counter() - start_time,
                )

            if not dls.cutoff_occurred:
                # Search space completely exhausted without cutoff
                break

        return SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=total_expanded,
            nodes_generated=total_generated,
            max_frontier_size=max_frontier,
            runtime_seconds=time.perf_counter() - start_time,
        )


class UniformCostSearch(Generic[S, A]):
    """Uniform-Cost Search (Dijkstra): explores nodes in order of cumulative cost g(n)."""

    def solve(self, problem: SearchProblem[S, A]) -> SearchResult[S, A]:
        for _ in self.step_solve(problem):
            pass
        return self._final_result

    def step_solve(
        self, problem: SearchProblem[S, A]
    ) -> Generator[SearchStepEvent[S, A], None, SearchResult[S, A]]:
        start_time = time.perf_counter()
        start = problem.initial_state

        tie_breaker = 0
        frontier: list[tuple[float, int, SearchNode[S, A]]] = []
        start_node: SearchNode[S, A] = SearchNode(state=start, g_cost=0.0, depth=0)
        heapq.heappush(frontier, (0.0, tie_breaker, start_node))

        best_g: dict[S, float] = {start: 0.0}
        closed_set: set[S] = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier = 1
        step_idx = 0

        while frontier:
            if len(frontier) > max_frontier:
                max_frontier = len(frontier)

            cost_priority, _, current = heapq.heappop(frontier)

            # Ignore stale entries
            if current.state in closed_set:
                continue
            if current.g_cost > best_g.get(current.state, float("inf")):
                continue

            closed_set.add(current.state)
            nodes_expanded += 1
            step_idx += 1

            yield SearchStepEvent(
                algorithm="UCS",
                step=step_idx,
                current_state=current.state,
                frontier_states=[item[2].state for item in frontier],
                explored_states=list(closed_set),
                f_cost=current.g_cost,
                g_cost=current.g_cost,
                nodes_expanded=nodes_expanded,
            )

            # Goal check on pop guarantees cost optimality
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

            for action, next_state, step_cost in problem.successors(current.state):
                new_g = current.g_cost + step_cost
                if next_state not in best_g or new_g < best_g[next_state]:
                    best_g[next_state] = new_g
                    nodes_generated += 1
                    tie_breaker += 1
                    child = SearchNode(
                        state=next_state,
                        parent=current,
                        action=action,
                        g_cost=new_g,
                        depth=current.depth + 1,
                    )
                    heapq.heappush(frontier, (new_g, tie_breaker, child))

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


class BidirectionalBFS(Generic[S, A]):
    """Bidirectional Breadth-First Search on reversible search graphs."""

    def solve(
        self,
        problem: SearchProblem[S, A],
        goal_state: S | None = None,
        reverse_successors_fn: object | None = None,
    ) -> SearchResult[S, A]:
        start_time = time.perf_counter()
        start = problem.initial_state

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

        # Fallback to single-directional BFS if goal_state is not explicitly provided
        if goal_state is None:
            return BreadthFirstSearch[S, A]().solve(problem)

        frontier_f: deque[SearchNode[S, A]] = deque([SearchNode(state=start, g_cost=0.0)])
        frontier_b: deque[SearchNode[S, A]] = deque([SearchNode(state=goal_state, g_cost=0.0)])

        visited_f: dict[S, SearchNode[S, A]] = {start: frontier_f[0]}
        visited_b: dict[S, SearchNode[S, A]] = {goal_state: frontier_b[0]}

        nodes_expanded = 0
        nodes_generated = 2
        max_frontier = 2

        while frontier_f and frontier_b:
            cur_frontier_len = len(frontier_f) + len(frontier_b)
            if cur_frontier_len > max_frontier:
                max_frontier = cur_frontier_len

            # 1. Expand forward step
            curr_f = frontier_f.popleft()
            nodes_expanded += 1

            if curr_f.state in visited_b:
                return self._join_paths(
                    curr_f,
                    visited_b[curr_f.state],
                    nodes_expanded,
                    nodes_generated,
                    max_frontier,
                    start_time,
                )

            for action, next_state, cost in problem.successors(curr_f.state):
                if next_state not in visited_f:
                    nodes_generated += 1
                    child_f = SearchNode(
                        state=next_state,
                        parent=curr_f,
                        action=action,
                        g_cost=curr_f.g_cost + cost,
                        depth=curr_f.depth + 1,
                    )
                    visited_f[next_state] = child_f
                    frontier_f.append(child_f)

                    if next_state in visited_b:
                        return self._join_paths(
                            child_f,
                            visited_b[next_state],
                            nodes_expanded,
                            nodes_generated,
                            max_frontier,
                            start_time,
                        )

            # 2. Expand backward step
            curr_b = frontier_b.popleft()
            nodes_expanded += 1

            if curr_b.state in visited_f:
                return self._join_paths(
                    visited_f[curr_b.state],
                    curr_b,
                    nodes_expanded,
                    nodes_generated,
                    max_frontier,
                    start_time,
                )

            rev_succs = (
                reverse_successors_fn(curr_b.state)
                if callable(reverse_successors_fn)
                else problem.successors(curr_b.state)
            )

            for action, prev_state, cost in rev_succs:
                if prev_state not in visited_b:
                    nodes_generated += 1
                    child_b = SearchNode(
                        state=prev_state,
                        parent=curr_b,
                        action=action,
                        g_cost=curr_b.g_cost + cost,
                        depth=curr_b.depth + 1,
                    )
                    visited_b[prev_state] = child_b
                    frontier_b.append(child_b)

                    if prev_state in visited_f:
                        return self._join_paths(
                            visited_f[prev_state],
                            child_b,
                            nodes_expanded,
                            nodes_generated,
                            max_frontier,
                            start_time,
                        )

        return SearchResult(
            success=False,
            path=[],
            states=[],
            total_cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier,
            runtime_seconds=time.perf_counter() - start_time,
        )

    def _join_paths(
        self,
        node_f: SearchNode[S, A],
        node_b: SearchNode[S, A],
        nodes_expanded: int,
        nodes_generated: int,
        max_frontier: int,
        start_time: float,
    ) -> SearchResult[S, A]:
        f_actions, f_states = reconstruct_path(node_f)

        # Unwind backward path from intersection to goal
        b_states: list[S] = []
        curr: SearchNode[S, A] | None = node_b.parent
        while curr is not None:
            b_states.append(curr.state)
            curr = curr.parent

        full_states = f_states + b_states
        # Compute total cost
        total_cost = node_f.g_cost + node_b.g_cost

        return SearchResult(
            success=True,
            path=f_actions,  # Primary forward actions to intersection
            states=full_states,
            total_cost=total_cost,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier,
            runtime_seconds=time.perf_counter() - start_time,
        )
