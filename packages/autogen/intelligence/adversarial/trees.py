"""Synthetic game trees for mathematical oracle testing and pruning verification (SYS-10, Module M07)."""

from __future__ import annotations

import random
from typing import Sequence


class SyntheticGameTree:
    """Explicit in-memory adversarial game tree implementation of AdversarialGame protocol."""

    def __init__(
        self,
        tree_dict: dict[int, tuple[int, list[tuple[int, int]]]],  # node_id -> (player, [(action, child_id)])
        terminal_utilities: dict[int, float],  # node_id -> utility for MAX
        initial_node: int = 0,
    ) -> None:
        self.tree_dict = tree_dict
        self.terminal_utilities = terminal_utilities
        self._initial_node = initial_node

    @property
    def initial_state(self) -> int:
        return self._initial_node

    def player(self, state: int) -> int:
        if state in self.terminal_utilities:
            return 0
        return self.tree_dict[state][0]

    def legal_actions(self, state: int) -> list[int]:
        if state in self.terminal_utilities:
            return []
        return [act for act, _ in self.tree_dict[state][1]]

    def successor(self, state: int, action: int) -> int:
        for act, child in self.tree_dict[state][1]:
            if act == action:
                return child
        raise ValueError(f"Action {action} illegal at state {state}")

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_utilities

    def utility(self, state: int, player: int) -> float:
        val = self.terminal_utilities[state]
        return val if player == 0 else -val

    def chance_outcomes(self, state: int, action: int) -> list[tuple[int, float]]:
        return [(self.successor(state, action), 1.0)]


def generate_random_game_tree(
    branching_factor: int = 3,
    depth: int = 4,
    min_utility: float = -100.0,
    max_utility: float = 100.0,
    rng: random.Random | None = None,
) -> SyntheticGameTree:
    """Recursively generates a synthetic complete game tree with random terminal leaf values."""
    if rng is None:
        rng = random.Random(42)

    tree_dict: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    terminal_utilities: dict[int, float] = {}
    node_counter = 0

    def build_node(current_depth: int, player: int) -> int:
        nonlocal node_counter
        nid = node_counter
        node_counter += 1

        if current_depth >= depth:
            # Terminal leaf node
            terminal_utilities[nid] = round(rng.uniform(min_utility, max_utility), 2)
            return nid

        children: list[tuple[int, int]] = []
        next_player = 1 - player
        for action_idx in range(branching_factor):
            cid = build_node(current_depth + 1, next_player)
            children.append((action_idx, cid))

        tree_dict[nid] = (player, children)
        return nid

    root = build_node(0, player=0)
    return SyntheticGameTree(tree_dict, terminal_utilities, initial_node=root)


def generate_crafted_ordered_tree(
    branching_factor: int = 3,
    depth: int = 4,
    best_ordered: bool = True,
) -> SyntheticGameTree:
    """Generates a game tree where children are either strictly best-first or worst-first ordered.

    When best_ordered=True, alpha-beta achieves optimal O(b^(d/2)) pruning.
    """
    tree_dict: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    terminal_utilities: dict[int, float] = {}
    node_counter = 0

    def build_node(current_depth: int, player: int, parent_val: float) -> tuple[int, float]:
        nonlocal node_counter
        nid = node_counter
        node_counter += 1

        if current_depth >= depth:
            val = parent_val
            terminal_utilities[nid] = val
            return nid, val

        next_player = 1 - player
        sub_nodes: list[tuple[int, float]] = []

        for b in range(branching_factor):
            # Deterministic distinct values
            child_seed_val = parent_val * 10 + (b + 1)
            cid, cval = build_node(current_depth + 1, next_player, child_seed_val)
            sub_nodes.append((cid, cval))

        # Sort children according to best or worst ordering
        if player == 0:  # MAX player wants highest first if best_ordered
            sub_nodes.sort(key=lambda x: x[1], reverse=best_ordered)
        else:  # MIN player wants lowest first if best_ordered
            sub_nodes.sort(key=lambda x: x[1], reverse=not best_ordered)

        tree_dict[nid] = (player, [(act, cid) for act, (cid, _) in enumerate(sub_nodes)])
        node_val = sub_nodes[0][1] if player == 0 else sub_nodes[-1][1]
        return nid, node_val

    root, _ = build_node(0, 0, 1.0)
    return SyntheticGameTree(tree_dict, terminal_utilities, initial_node=root)

