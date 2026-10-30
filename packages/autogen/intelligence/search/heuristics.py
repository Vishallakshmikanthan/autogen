"""Admissible and consistent heuristic functions for grid search (SYS-06, Module M06).

Mathematical Notice on Admissibility:
On a grid where edge transitions have variable non-negative step costs c(s, a, s') in [c_min, c_max],
standard unscaled topological distance D(s, goal) is admissible IF AND ONLY IF it is scaled by
the minimum possible step cost:
    h(s) = c_min * D(s, goal) <= h*(s)

If c_min is omitted or assumed to be 1.0 while actual terrain allows cheaper traversal (or vice versa),
the heuristic will either overestimate true cost (breaking admissibility and optimality) or provide
a looser lower bound than necessary.
"""

from __future__ import annotations

import math
from typing import Protocol


class CoordinateLike(Protocol):
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...


def _extract_coords(p: tuple[int, int] | CoordinateLike) -> tuple[int, int]:
    if isinstance(p, tuple):
        return p[0], p[1]
    return p.x, p.y


def manhattan_distance(
    p1: tuple[int, int] | CoordinateLike,
    p2: tuple[int, int] | CoordinateLike,
    min_step_cost: float = 1.0,
) -> float:
    """Computes scaled Manhattan (L1) distance for 4-connected grid movement."""
    x1, y1 = _extract_coords(p1)
    x2, y2 = _extract_coords(p2)
    return min_step_cost * float(abs(x1 - x2) + abs(y1 - y2))


def euclidean_distance(
    p1: tuple[int, int] | CoordinateLike,
    p2: tuple[int, int] | CoordinateLike,
    min_step_cost: float = 1.0,
) -> float:
    """Computes scaled Euclidean (L2) straight-line distance."""
    x1, y1 = _extract_coords(p1)
    x2, y2 = _extract_coords(p2)
    return min_step_cost * math.hypot(x1 - x2, y1 - y2)


def chebyshev_distance(
    p1: tuple[int, int] | CoordinateLike,
    p2: tuple[int, int] | CoordinateLike,
    min_step_cost: float = 1.0,
) -> float:
    """Computes scaled Chebyshev (L_infinity) distance (diagonal step cost equal to cardinal)."""
    x1, y1 = _extract_coords(p1)
    x2, y2 = _extract_coords(p2)
    return min_step_cost * float(max(abs(x1 - x2), abs(y1 - y2)))


def octile_distance(
    p1: tuple[int, int] | CoordinateLike,
    p2: tuple[int, int] | CoordinateLike,
    min_step_cost: float = 1.0,
) -> float:
    """Computes scaled Octile distance for 8-connected grid movement (diagonal cost sqrt(2))."""
    x1, y1 = _extract_coords(p1)
    x2, y2 = _extract_coords(p2)
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return min_step_cost * (max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))


def null_heuristic(state: object) -> float:
    """Zero heuristic h(s) = 0.0. Reduces A* to Dijkstra / Uniform-Cost Search."""
    return 0.0
