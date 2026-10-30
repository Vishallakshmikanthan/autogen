"""GridWorld to exact Tabular MDP converter and canonical benchmarks (SYS-10, Module M11)."""

from __future__ import annotations

from typing import Mapping
import numpy as np

from autogen.intelligence.decision.contracts import TabularMDP
from autogen.simulation.state.types import ActionType, Position


class GridWorldMDPBuilder:
    """Constructs an exact TabularMDP from discrete grid specifications."""

    # 4 standard directional actions
    ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
    ACTION_OFFSETS: dict[str, tuple[int, int]] = {
        "UP": (0, 1),
        "DOWN": (0, -1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }
    # Orthogonal slip actions
    ORTHOGONAL_SLIPS: dict[str, tuple[str, str]] = {
        "UP": ("LEFT", "RIGHT"),
        "DOWN": ("LEFT", "RIGHT"),
        "LEFT": ("UP", "DOWN"),
        "RIGHT": ("UP", "DOWN"),
    }

    @classmethod
    def build_grid_mdp(
        cls,
        width: int,
        height: int,
        walls: set[tuple[int, int]],
        terminals: Mapping[tuple[int, int], float],
        step_cost: float = -0.04,
        slip_prob: float = 0.2,
        gamma: float = 0.99,
    ) -> TabularMDP:
        """Constructs an exact tabular MDP for a grid with stochastic slip and terminal rewards."""
        # Active states: all non-wall cells
        valid_cells: list[tuple[int, int]] = []
        for y in range(height):
            for x in range(width):
                if (x, y) not in walls:
                    valid_cells.append((x, y))

        num_states = len(valid_cells)
        num_actions = len(cls.ACTIONS)

        cell_to_idx = {c: i for i, c in enumerate(valid_cells)}
        idx_to_cell = {i: c for i, c in enumerate(valid_cells)}
        state_names = tuple(f"({c[0]},{c[1]})" for c in valid_cells)

        terminal_indices = frozenset(
            cell_to_idx[c] for c in terminals if c in cell_to_idx
        )

        T = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        R = np.zeros((num_states, num_actions, num_states), dtype=np.float64)

        for s_idx, (x, y) in enumerate(valid_cells):
            if s_idx in terminal_indices:
                # Terminal state self-loop
                for a_idx in range(num_actions):
                    T[s_idx, a_idx, s_idx] = 1.0
                    R[s_idx, a_idx, s_idx] = 0.0
                continue

            for a_idx, act_name in enumerate(cls.ACTIONS):
                # Outcomes: intended action with (1 - slip), orthogonal slips with slip/2
                outcomes: list[tuple[str, float]] = [
                    (act_name, 1.0 - slip_prob),
                    (cls.ORTHOGONAL_SLIPS[act_name][0], slip_prob / 2.0),
                    (cls.ORTHOGONAL_SLIPS[act_name][1], slip_prob / 2.0),
                ]

                for outcome_action, prob in outcomes:
                    dx, dy = cls.ACTION_OFFSETS[outcome_action]
                    target_cell = (x + dx, y + dy)

                    # Bounces back if out of bounds or hits a wall
                    if not (0 <= target_cell[0] < width and 0 <= target_cell[1] < height) or target_cell in walls:
                        dest_cell = (x, y)
                    else:
                        dest_cell = target_cell

                    dest_idx = cell_to_idx[dest_cell]
                    T[s_idx, a_idx, dest_idx] += prob

                    # Reward model: includes step cost plus terminal payoff if entering terminal cell
                    if dest_cell in terminals:
                        reward = step_cost + terminals[dest_cell]
                    else:
                        reward = step_cost

                    # Reward given transition
                    R[s_idx, a_idx, dest_idx] = reward

        return TabularMDP(
            num_states=num_states,
            num_actions=num_actions,
            states=state_names,
            actions=cls.ACTIONS,
            transition_tensor=T,
            reward_tensor=R,
            gamma=gamma,
            terminal_states=terminal_indices,
        )


def classic_russell_norvig_grid(
    step_cost: float = -0.04,
    slip_prob: float = 0.2,
    gamma: float = 1.0,
) -> TabularMDP:
    """The canonical 4x3 GridWorld environment from Russell & Norvig (AIMA 4th ed., Section 17.1).

    Layout:
      Row 2: (0, 2), (1, 2), (2, 2), (3, 2)[+1.0 Terminal]
      Row 1: (0, 1), [WALL], (2, 1), (3, 1)[-1.0 Terminal]
      Row 0: (0, 0), (1, 0), (2, 0), (3, 0)
    """
    walls = {(1, 1)}
    terminals = {(3, 2): 1.0, (3, 1): -1.0}
    return GridWorldMDPBuilder.build_grid_mdp(
        width=4,
        height=3,
        walls=walls,
        terminals=terminals,
        step_cost=step_cost,
        slip_prob=slip_prob,
        gamma=gamma,
    )
