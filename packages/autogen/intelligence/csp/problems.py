"""Benchmark and simulation-linked problem factories for CSP (SYS-08, Module M09)."""

from __future__ import annotations

from collections.abc import Callable

from autogen.intelligence.csp.contracts import (
    AllDifferentConstraint,
    BinaryConstraint,
    Constraint,
    CSP,
)


def map_coloring_problem(
    regions: tuple[str, ...],
    adjacency: dict[str, list[str]],
    colors: tuple[str, ...] = ("red", "green", "blue"),
) -> CSP[str, str]:
    """Generates a Map Coloring CSP instance.

    Variables: Regions. Domains: Colors. Constraints: Neighboring regions must have different colors.
    """
    domains = {r: list(colors) for r in regions}
    constraints: list[Constraint[str, str]] = []

    seen_edges: set[tuple[str, str]] = set()
    for r1, neighbors in adjacency.items():
        for r2 in neighbors:
            edge = tuple(sorted((r1, r2)))
            if edge not in seen_edges and r1 != r2:
                seen_edges.add(edge)  # type: ignore[arg-type]
                constraints.append(
                    BinaryConstraint(
                        var1=r1,
                        var2=r2,
                        predicate=lambda c1, c2: c1 != c2,
                        name=f"ColorDiff({r1}, {r2})",
                    )
                )

    return CSP(variables=regions, domains=domains, constraints=constraints)


def australia_map_coloring() -> CSP[str, str]:
    """Canonical Map Coloring problem for Australia's 7 states and territories."""
    regions = ("WA", "NT", "SA", "Q", "NSW", "V", "T")
    adjacency = {
        "WA": ["NT", "SA"],
        "NT": ["WA", "SA", "Q"],
        "SA": ["WA", "NT", "Q", "NSW", "V"],
        "Q": ["NT", "SA", "NSW"],
        "NSW": ["Q", "SA", "V"],
        "V": ["SA", "NSW"],
        "T": [],  # Tasmania has no land borders
    }
    return map_coloring_problem(regions, adjacency, colors=("red", "green", "blue"))


def n_queens_problem(n: int) -> CSP[int, int]:
    """Generates an N-Queens CSP instance.

    Variables: Columns 0 to n-1. Domains: Rows 0 to n-1.
    Constraints: No two queens may share a row or diagonal (|Q_i - Q_j| != |i - j|).
    """
    variables = tuple(range(n))
    domains = {i: list(range(n)) for i in variables}
    constraints: list[Constraint[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            col_diff = abs(i - j)

            def make_nqueens_predicate(diff: int) -> Callable[[int, int], bool]:
                return lambda row1, row2: row1 != row2 and abs(row1 - row2) != diff

            constraints.append(
                BinaryConstraint(
                    var1=i,
                    var2=j,
                    predicate=make_nqueens_predicate(col_diff),
                    name=f"NQueensConflict({i}, {j})",
                )
            )

    return CSP(variables=variables, domains=domains, constraints=constraints)


def sudoku_problem(grid: str) -> CSP[tuple[int, int], int]:
    """Generates a 9x9 Sudoku CSP instance from an 81-character string (0 or '.' for empty)."""
    clean_grid = "".join(grid.split()).replace(".", "0")
    if len(clean_grid) != 81:
        raise ValueError(f"Sudoku grid must contain exactly 81 cells, got {len(clean_grid)}.")

    variables = tuple((r, c) for r in range(9) for c in range(9))
    domains: dict[tuple[int, int], list[int]] = {}

    for r in range(9):
        for c in range(9):
            char = clean_grid[r * 9 + c]
            if char.isdigit() and int(char) in range(1, 10):
                domains[(r, c)] = [int(char)]
            else:
                domains[(r, c)] = list(range(1, 10))

    constraints: list[Constraint[tuple[int, int], int]] = []

    # Row all-different constraints
    for r in range(9):
        row_vars = tuple((r, c) for c in range(9))
        constraints.append(AllDifferentConstraint(variables=row_vars))

    # Col all-different constraints
    for c in range(9):
        col_vars = tuple((r, c) for r in range(9))
        constraints.append(AllDifferentConstraint(variables=col_vars))

    # 3x3 Box all-different constraints
    for box_r in range(3):
        for box_c in range(3):
            box_vars = tuple(
                (box_r * 3 + dr, box_c * 3 + dc)
                for dr in range(3)
                for dc in range(3)
            )
            constraints.append(AllDifferentConstraint(variables=box_vars))

    return CSP(variables=variables, domains=domains, constraints=constraints)


def multi_agent_schedule_csp(
    agent_ids: tuple[str, ...],
    tasks: tuple[str, ...],
    time_slots: tuple[int, ...],
    task_conflicts: list[tuple[str, str]] | None = None,
) -> CSP[str, tuple[str, int]]:
    """World-linked multi-agent patrol/foraging schedule assignment CSP.

    Variables: Tasks to complete.
    Domains: (AgentID, TimeSlot) assignments.
    Constraints:
      1. One agent cannot execute two different tasks at the same time slot.
      2. Conflicting tasks cannot be scheduled in the same time slot.
    """
    possible_assignments = [(ag, slot) for ag in agent_ids for slot in time_slots]
    domains = {t: list(possible_assignments) for t in tasks}
    constraints: list[Constraint[str, tuple[str, int]]] = []

    # Constraint 1: One agent cannot be double-booked across any two tasks
    for i, t1 in enumerate(tasks):
        for j in range(i + 1, len(tasks)):
            t2 = tasks[j]
            constraints.append(
                BinaryConstraint(
                    var1=t1,
                    var2=t2,
                    predicate=lambda assign1, assign2: not (
                        assign1[0] == assign2[0] and assign1[1] == assign2[1]
                    ),
                    name=f"NoDoubleBooking({t1}, {t2})",
                )
            )

    # Constraint 2: Mutex tasks cannot occur in the exact same time slot
    if task_conflicts:
        for t1, t2 in task_conflicts:
            constraints.append(
                BinaryConstraint(
                    var1=t1,
                    var2=t2,
                    predicate=lambda assign1, assign2: assign1[1] != assign2[1],
                    name=f"MutuallyExclusiveTaskSlots({t1}, {t2})",
                )
            )

    return CSP(variables=tasks, domains=domains, constraints=constraints)
