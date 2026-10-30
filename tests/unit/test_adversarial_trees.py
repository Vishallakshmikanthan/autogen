"""Mathematical verification of Minimax and Alpha-Beta pruning on random and crafted trees (Module M07)."""

from __future__ import annotations

import random
import pytest

from autogen.intelligence.adversarial.evaluations import (
    PredatorPreyEvaluator,
    PredatorPreyState,
)
from autogen.intelligence.adversarial.minimax import AlphaBetaSearch, MinimaxSearch
from autogen.intelligence.adversarial.trees import (
    generate_crafted_ordered_tree,
    generate_random_game_tree,
)
from autogen.simulation.state.types import Position


def test_alphabeta_equals_minimax_on_1000_random_trees() -> None:
    """Oracle verification: Alpha-Beta value == Minimax value on 1000 random game trees.

    Also asserts that nodes expanded by Alpha-Beta <= nodes expanded by Minimax on every tree.
    """
    minimax = MinimaxSearch[int, int]()
    alphabeta = AlphaBetaSearch[int, int]()

    # Test 1000 random game trees
    num_trees = 1000
    rng = random.Random(1337)

    for i in range(num_trees):
        # Vary branching factors 2..4 and depths 2..4 for fast exhaustive check
        b = rng.randint(2, 4)
        d = rng.randint(2, 4)
        tree = generate_random_game_tree(branching_factor=b, depth=d, rng=rng)

        # Default heuristic (evaluates to 0 at depth cutoff, but our trees are complete)
        dummy_eval = lambda state, player: 0.0

        m_res = minimax.search(tree, tree.initial_state, max_depth=d, eval_fn=dummy_eval)
        ab_res = alphabeta.search(tree, tree.initial_state, max_depth=d, eval_fn=dummy_eval)

        # Invariant 1: Values must be mathematically identical
        assert pytest.approx(m_res.expected_value, abs=1e-5) == ab_res.expected_value, (
            f"Tree {i} (b={b}, d={d}) value mismatch: Minimax={m_res.expected_value}, AlphaBeta={ab_res.expected_value}"
        )

        # Invariant 2: Alpha-Beta must never expand more nodes than Minimax
        assert ab_res.nodes_expanded <= m_res.nodes_expanded, (
            f"Tree {i} AlphaBeta expanded {ab_res.nodes_expanded} > Minimax {m_res.nodes_expanded}"
        )


def test_alphabeta_move_ordering_pruning_bound() -> None:
    """Verifies that best-case move ordering yields near O(b^(d/2)) pruning compared to worst-case."""
    b = 3
    d = 4
    # On full tree without pruning, nodes = sum_{i=0}^d b^i = (3^5 - 1)/2 = 121
    best_tree = generate_crafted_ordered_tree(branching_factor=b, depth=d, best_ordered=True)
    worst_tree = generate_crafted_ordered_tree(branching_factor=b, depth=d, best_ordered=False)

    alphabeta = AlphaBetaSearch[int, int]()
    dummy_eval = lambda s, p: 0.0

    res_best = alphabeta.search(best_tree, best_tree.initial_state, max_depth=d, eval_fn=dummy_eval)
    res_worst = alphabeta.search(worst_tree, worst_tree.initial_state, max_depth=d, eval_fn=dummy_eval)

    # Invariant: Best ordering prunes significantly more nodes than worst ordering
    assert res_best.nodes_expanded < res_worst.nodes_expanded
    # Theoretical optimal bound: b^(d/2) + b^(d/2) - 1 = 3^2 + 3^2 - 1 = 17 nodes
    # Verify that best ordering is within a factor of optimal bound (expands <= 40 vs 121)
    assert res_best.nodes_expanded <= 40
