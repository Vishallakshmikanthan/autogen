"""Adversarial search algorithms: Minimax, Alpha-Beta pruning, and Expectimax (SYS-10, Module M07)."""

from __future__ import annotations

import time
from typing import Callable, Generic, Hashable, Sequence, TypeVar

from autogen.intelligence.adversarial.contracts import (
    AdversarialGame,
    AdversarialSearchResult,
    AdversarialStepEvent,
    EvaluationFunction,
    GameTreeNode,
    MoveOrderingFunction,
)

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)


class MinimaxSearch(Generic[S, A]):
    """Standard Minimax adversarial search with depth cutoff and heuristic evaluation."""

    def __init__(
        self,
        record_tree: bool = False,
        step_callback: Callable[[AdversarialStepEvent[S, A]], None] | None = None,
    ) -> None:
        self.record_tree = record_tree
        self.step_callback = step_callback
        self.nodes_expanded = 0
        self._step_counter = 0

    def search(
        self,
        game: AdversarialGame[S, A],
        state: S,
        max_depth: int,
        eval_fn: EvaluationFunction[S],
    ) -> AdversarialSearchResult[A]:
        """Executes minimax deliberation from state s up to max_depth."""
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        self._step_counter = 0

        if game.is_terminal(state):
            val = game.utility(state, 0)
            return AdversarialSearchResult(
                best_action=None,
                expected_value=val,
                nodes_expanded=1,
                nodes_pruned=0,
                max_depth_reached=0,
                runtime_seconds=time.perf_counter() - start_time,
            )

        legal_acts = game.legal_actions(state)
        if not legal_acts:
            val = eval_fn(state, 0)
            return AdversarialSearchResult(
                best_action=None,
                expected_value=val,
                nodes_expanded=1,
                nodes_pruned=0,
                max_depth_reached=0,
                runtime_seconds=time.perf_counter() - start_time,
            )

        cur_player = game.player(state)
        root_node = (
            GameTreeNode(
                node_id="root",
                player=cur_player,
                depth=0,
                value=0.0,
                state_repr=str(state),
            )
            if self.record_tree
            else None
        )

        best_action: A | None = None
        if cur_player == 0:  # MAX player
            best_val = float("-inf")
            for act in legal_acts:
                succ = game.successor(state, act)
                child_node = (
                    GameTreeNode(
                        node_id=f"root->{act}",
                        player=game.player(succ),
                        depth=1,
                        value=0.0,
                        best_action=act,
                        state_repr=str(succ),
                    )
                    if root_node
                    else None
                )
                val = self._min_value(game, succ, max_depth - 1, eval_fn, 1, child_node)
                if child_node and root_node:
                    child_node.value = val
                    root_node.children.append(child_node)
                if val > best_val:
                    best_val = val
                    best_action = act
            if root_node:
                root_node.value = best_val
                root_node.best_action = best_action
        else:  # MIN player
            best_val = float("inf")
            for act in legal_acts:
                succ = game.successor(state, act)
                child_node = (
                    GameTreeNode(
                        node_id=f"root->{act}",
                        player=game.player(succ),
                        depth=1,
                        value=0.0,
                        best_action=act,
                        state_repr=str(succ),
                    )
                    if root_node
                    else None
                )
                val = self._max_value(game, succ, max_depth - 1, eval_fn, 1, child_node)
                if child_node and root_node:
                    child_node.value = val
                    root_node.children.append(child_node)
                if val < best_val:
                    best_val = val
                    best_action = act
            if root_node:
                root_node.value = best_val
                root_node.best_action = best_action

        return AdversarialSearchResult(
            best_action=best_action,
            expected_value=best_val,
            nodes_expanded=self.nodes_expanded,
            nodes_pruned=0,
            max_depth_reached=max_depth,
            tree_root=root_node,
            runtime_seconds=time.perf_counter() - start_time,
        )

    def _max_value(
        self,
        game: AdversarialGame[S, A],
        state: S,
        depth: int,
        eval_fn: EvaluationFunction[S],
        current_depth: int,
        parent_node: GameTreeNode | None,
    ) -> float:
        self.nodes_expanded += 1
        if game.is_terminal(state):
            val = game.utility(state, 0)
            self._emit_step(state, 0, current_depth, val)
            return val
        if depth <= 0:
            val = eval_fn(state, 0)
            self._emit_step(state, 0, current_depth, val)
            return val

        v = float("-inf")
        legal_acts = game.legal_actions(state)
        if not legal_acts:
            return eval_fn(state, 0)

        for act in legal_acts:
            succ = game.successor(state, act)
            child_node = (
                GameTreeNode(
                    node_id=f"{parent_node.node_id}->{act}" if parent_node else "",
                    player=game.player(succ),
                    depth=current_depth + 1,
                    value=0.0,
                    best_action=act,
                    state_repr=str(succ),
                )
                if parent_node
                else None
            )
            val = self._min_value(
                game, succ, depth - 1, eval_fn, current_depth + 1, child_node
            )
            if child_node and parent_node:
                child_node.value = val
                parent_node.children.append(child_node)
            v = max(v, val)

        self._emit_step(state, 0, current_depth, v)
        return v

    def _min_value(
        self,
        game: AdversarialGame[S, A],
        state: S,
        depth: int,
        eval_fn: EvaluationFunction[S],
        current_depth: int,
        parent_node: GameTreeNode | None,
    ) -> float:
        self.nodes_expanded += 1
        if game.is_terminal(state):
            val = game.utility(state, 0)
            self._emit_step(state, 1, current_depth, val)
            return val
        if depth <= 0:
            val = eval_fn(state, 0)
            self._emit_step(state, 1, current_depth, val)
            return val

        v = float("inf")
        legal_acts = game.legal_actions(state)
        if not legal_acts:
            return eval_fn(state, 0)

        for act in legal_acts:
            succ = game.successor(state, act)
            child_node = (
                GameTreeNode(
                    node_id=f"{parent_node.node_id}->{act}" if parent_node else "",
                    player=game.player(succ),
                    depth=current_depth + 1,
                    value=0.0,
                    best_action=act,
                    state_repr=str(succ),
                )
                if parent_node
                else None
            )
            val = self._max_value(
                game, succ, depth - 1, eval_fn, current_depth + 1, child_node
            )
            if child_node and parent_node:
                child_node.value = val
                parent_node.children.append(child_node)
            v = min(v, val)

        self._emit_step(state, 1, current_depth, v)
        return v

    def _emit_step(self, state: S, player: int, depth: int, value: float) -> None:
        if self.step_callback is not None:
            self._step_counter += 1
            self.step_callback(
                AdversarialStepEvent(
                    step=self._step_counter,
                    current_state=state,
                    player=player,
                    depth=depth,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    evaluated_value=value,
                    nodes_expanded=self.nodes_expanded,
                    pruned=False,
                )
            )


class AlphaBetaSearch(Generic[S, A]):
    """Alpha-Beta pruning search algorithm with move ordering and cutoff verification."""

    def __init__(
        self,
        move_ordering_fn: MoveOrderingFunction[S, A] | None = None,
        record_tree: bool = False,
        step_callback: Callable[[AdversarialStepEvent[S, A]], None] | None = None,
    ) -> None:
        self.move_ordering_fn = move_ordering_fn
        self.record_tree = record_tree
        self.step_callback = step_callback
        self.nodes_expanded = 0
        self.nodes_pruned = 0
        self._step_counter = 0

    def search(
        self,
        game: AdversarialGame[S, A],
        state: S,
        max_depth: int,
        eval_fn: EvaluationFunction[S],
    ) -> AdversarialSearchResult[A]:
        """Executes Alpha-Beta deliberation from state s up to max_depth."""
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        self.nodes_pruned = 0
        self._step_counter = 0

        if game.is_terminal(state):
            val = game.utility(state, 0)
            return AdversarialSearchResult(
                best_action=None,
                expected_value=val,
                nodes_expanded=1,
                nodes_pruned=0,
                max_depth_reached=0,
                runtime_seconds=time.perf_counter() - start_time,
            )

        legal_acts = game.legal_actions(state)
        if not legal_acts:
            val = eval_fn(state, 0)
            return AdversarialSearchResult(
                best_action=None,
                expected_value=val,
                nodes_expanded=1,
                nodes_pruned=0,
                max_depth_reached=0,
                runtime_seconds=time.perf_counter() - start_time,
            )

        cur_player = game.player(state)
        ordered_acts = self._order_moves(legal_acts, state, cur_player)

        root_node = (
            GameTreeNode(
                node_id="root",
                player=cur_player,
                depth=0,
                value=0.0,
                alpha=float("-inf"),
                beta=float("inf"),
                state_repr=str(state),
            )
            if self.record_tree
            else None
        )

        alpha = float("-inf")
        beta = float("inf")
        best_action: A | None = None

        if cur_player == 0:  # MAX player
            best_val = float("-inf")
            for act in ordered_acts:
                succ = game.successor(state, act)
                child_node = (
                    GameTreeNode(
                        node_id=f"root->{act}",
                        player=game.player(succ),
                        depth=1,
                        value=0.0,
                        best_action=act,
                        alpha=alpha,
                        beta=beta,
                        state_repr=str(succ),
                    )
                    if root_node
                    else None
                )
                val = self._min_value(
                    game, succ, max_depth - 1, alpha, beta, eval_fn, 1, child_node
                )
                if child_node and root_node:
                    child_node.value = val
                    root_node.children.append(child_node)
                if val > best_val:
                    best_val = val
                    best_action = act
                alpha = max(alpha, best_val)
            if root_node:
                root_node.value = best_val
                root_node.best_action = best_action
                root_node.alpha = alpha
                root_node.beta = beta
        else:  # MIN player
            best_val = float("inf")
            for act in ordered_acts:
                succ = game.successor(state, act)
                child_node = (
                    GameTreeNode(
                        node_id=f"root->{act}",
                        player=game.player(succ),
                        depth=1,
                        value=0.0,
                        best_action=act,
                        alpha=alpha,
                        beta=beta,
                        state_repr=str(succ),
                    )
                    if root_node
                    else None
                )
                val = self._max_value(
                    game, succ, max_depth - 1, alpha, beta, eval_fn, 1, child_node
                )
                if child_node and root_node:
                    child_node.value = val
                    root_node.children.append(child_node)
                if val < best_val:
                    best_val = val
                    best_action = act
                beta = min(beta, best_val)
            if root_node:
                root_node.value = best_val
                root_node.best_action = best_action
                root_node.alpha = alpha
                root_node.beta = beta

        return AdversarialSearchResult(
            best_action=best_action,
            expected_value=best_val,
            nodes_expanded=self.nodes_expanded,
            nodes_pruned=self.nodes_pruned,
            max_depth_reached=max_depth,
            tree_root=root_node,
            runtime_seconds=time.perf_counter() - start_time,
        )

    def _max_value(
        self,
        game: AdversarialGame[S, A],
        state: S,
        depth: int,
        alpha: float,
        beta: float,
        eval_fn: EvaluationFunction[S],
        current_depth: int,
        parent_node: GameTreeNode | None,
    ) -> float:
        self.nodes_expanded += 1
        if game.is_terminal(state):
            val = game.utility(state, 0)
            self._emit_step(state, 0, current_depth, alpha, beta, val, pruned=False)
            return val
        if depth <= 0:
            val = eval_fn(state, 0)
            self._emit_step(state, 0, current_depth, alpha, beta, val, pruned=False)
            return val

        v = float("-inf")
        legal_acts = game.legal_actions(state)
        if not legal_acts:
            return eval_fn(state, 0)

        ordered_acts = self._order_moves(legal_acts, state, player=0)

        for i, act in enumerate(ordered_acts):
            succ = game.successor(state, act)
            child_node = (
                GameTreeNode(
                    node_id=f"{parent_node.node_id}->{act}" if parent_node else "",
                    player=game.player(succ),
                    depth=current_depth + 1,
                    value=0.0,
                    best_action=act,
                    alpha=alpha,
                    beta=beta,
                    state_repr=str(succ),
                )
                if parent_node
                else None
            )
            val = self._min_value(
                game, succ, depth - 1, alpha, beta, eval_fn, current_depth + 1, child_node
            )
            if child_node and parent_node:
                child_node.value = val
                parent_node.children.append(child_node)
            v = max(v, val)
            if v >= beta:
                # Prune remaining sibling branches
                remaining = len(ordered_acts) - 1 - i
                self.nodes_pruned += remaining
                self._emit_step(state, 0, current_depth, alpha, beta, v, pruned=True)
                if parent_node:
                    parent_node.pruned = True
                return v
            alpha = max(alpha, v)

        self._emit_step(state, 0, current_depth, alpha, beta, v, pruned=False)
        return v

    def _min_value(
        self,
        game: AdversarialGame[S, A],
        state: S,
        depth: int,
        alpha: float,
        beta: float,
        eval_fn: EvaluationFunction[S],
        current_depth: int,
        parent_node: GameTreeNode | None,
    ) -> float:
        self.nodes_expanded += 1
        if game.is_terminal(state):
            val = game.utility(state, 0)
            self._emit_step(state, 1, current_depth, alpha, beta, val, pruned=False)
            return val
        if depth <= 0:
            val = eval_fn(state, 0)
            self._emit_step(state, 1, current_depth, alpha, beta, val, pruned=False)
            return val

        v = float("inf")
        legal_acts = game.legal_actions(state)
        if not legal_acts:
            return eval_fn(state, 0)

        ordered_acts = self._order_moves(legal_acts, state, player=1)

        for i, act in enumerate(ordered_acts):
            succ = game.successor(state, act)
            child_node = (
                GameTreeNode(
                    node_id=f"{parent_node.node_id}->{act}" if parent_node else "",
                    player=game.player(succ),
                    depth=current_depth + 1,
                    value=0.0,
                    best_action=act,
                    alpha=alpha,
                    beta=beta,
                    state_repr=str(succ),
                )
                if parent_node
                else None
            )
            val = self._max_value(
                game, succ, depth - 1, alpha, beta, eval_fn, current_depth + 1, child_node
            )
            if child_node and parent_node:
                child_node.value = val
                parent_node.children.append(child_node)
            v = min(v, val)
            if v <= alpha:
                # Prune remaining sibling branches
                remaining = len(ordered_acts) - 1 - i
                self.nodes_pruned += remaining
                self._emit_step(state, 1, current_depth, alpha, beta, v, pruned=True)
                if parent_node:
                    parent_node.pruned = True
                return v
            beta = min(beta, v)

        self._emit_step(state, 1, current_depth, alpha, beta, v, pruned=False)
        return v

    def _order_moves(self, actions: Sequence[A], state: S, player: int) -> list[A]:
        if self.move_ordering_fn is not None:
            return self.move_ordering_fn(actions, state, player)
        return list(actions)

    def _emit_step(
        self,
        state: S,
        player: int,
        depth: int,
        alpha: float,
        beta: float,
        value: float,
        pruned: bool,
    ) -> None:
        if self.step_callback is not None:
            self._step_counter += 1
            self.step_callback(
                AdversarialStepEvent(
                    step=self._step_counter,
                    current_state=state,
                    player=player,
                    depth=depth,
                    alpha=alpha,
                    beta=beta,
                    evaluated_value=value,
                    nodes_expanded=self.nodes_expanded,
                    pruned=pruned,
                )
            )


class ExpectimaxSearch(Generic[S, A]):
    """Expectimax search algorithm modeling environments with stochastic opponents or chance events."""

    def __init__(
        self,
        record_tree: bool = False,
        step_callback: Callable[[AdversarialStepEvent[S, A]], None] | None = None,
    ) -> None:
        self.record_tree = record_tree
        self.step_callback = step_callback
        self.nodes_expanded = 0
        self._step_counter = 0

    def search(
        self,
        game: AdversarialGame[S, A],
        state: S,
        max_depth: int,
        eval_fn: EvaluationFunction[S],
    ) -> AdversarialSearchResult[A]:
        """Executes Expectimax deliberation from state s up to max_depth."""
        start_time = time.perf_counter()
        self.nodes_expanded = 0
        self._step_counter = 0

        if game.is_terminal(state):
            val = game.utility(state, 0)
            return AdversarialSearchResult(
                best_action=None,
                expected_value=val,
                nodes_expanded=1,
                nodes_pruned=0,
                max_depth_reached=0,
                runtime_seconds=time.perf_counter() - start_time,
            )

        legal_acts = game.legal_actions(state)
        if not legal_acts:
            val = eval_fn(state, 0)
            return AdversarialSearchResult(
                best_action=None,
                expected_value=val,
                nodes_expanded=1,
                nodes_pruned=0,
                max_depth_reached=0,
                runtime_seconds=time.perf_counter() - start_time,
            )

        cur_player = game.player(state)
        root_node = (
            GameTreeNode(
                node_id="root",
                player=cur_player,
                depth=0,
                value=0.0,
                state_repr=str(state),
            )
            if self.record_tree
            else None
        )

        best_val = float("-inf")
        best_action: A | None = None

        for act in legal_acts:
            # Check if this action leads to chance outcomes or deterministic successor
            outcomes = game.chance_outcomes(state, act)
            if not outcomes:
                succ = game.successor(state, act)
                outcomes = [(succ, 1.0)]

            act_expected_value = 0.0
            for succ_state, prob in outcomes:
                val = self._expectimax_value(
                    game, succ_state, max_depth - 1, eval_fn, 1
                )
                act_expected_value += prob * val

            if act_expected_value > best_val:
                best_val = act_expected_value
                best_action = act

            if root_node:
                child = GameTreeNode(
                    node_id=f"root->{act}",
                    player=1,
                    depth=1,
                    value=act_expected_value,
                    best_action=act,
                )
                root_node.children.append(child)

        if root_node:
            root_node.value = best_val
            root_node.best_action = best_action

        return AdversarialSearchResult(
            best_action=best_action,
            expected_value=best_val,
            nodes_expanded=self.nodes_expanded,
            nodes_pruned=0,
            max_depth_reached=max_depth,
            tree_root=root_node,
            runtime_seconds=time.perf_counter() - start_time,
        )

    def _expectimax_value(
        self,
        game: AdversarialGame[S, A],
        state: S,
        depth: int,
        eval_fn: EvaluationFunction[S],
        current_depth: int,
    ) -> float:
        self.nodes_expanded += 1
        if game.is_terminal(state):
            return game.utility(state, 0)
        if depth <= 0:
            return eval_fn(state, 0)

        p = game.player(state)
        if p == 0:  # MAX player
            v = float("-inf")
            legal_acts = game.legal_actions(state)
            if not legal_acts:
                return eval_fn(state, 0)
            for act in legal_acts:
                outcomes = game.chance_outcomes(state, act)
                if not outcomes:
                    succ = game.successor(state, act)
                    outcomes = [(succ, 1.0)]
                act_val = 0.0
                for succ_state, prob in outcomes:
                    act_val += prob * self._expectimax_value(
                        game, succ_state, depth - 1, eval_fn, current_depth + 1
                    )
                v = max(v, act_val)
            return v
        else:  # CHANCE or MIN player acting stochastically
            legal_acts = game.legal_actions(state)
            if not legal_acts:
                return eval_fn(state, 0)
            # Uniform expectation across opponent's legal actions (or specific chance outcomes)
            uniform_prob = 1.0 / len(legal_acts)
            expected_val = 0.0
            for act in legal_acts:
                outcomes = game.chance_outcomes(state, act)
                if not outcomes:
                    succ = game.successor(state, act)
                    outcomes = [(succ, 1.0)]
                branch_val = 0.0
                for succ_state, prob in outcomes:
                    branch_val += prob * self._expectimax_value(
                        game, succ_state, depth - 1, eval_fn, current_depth + 1
                    )
                expected_val += uniform_prob * branch_val
            return expected_val

