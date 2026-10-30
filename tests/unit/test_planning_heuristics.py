"""Unit tests for domain-independent delete-relaxation heuristics (h_max, h_add, h_ff)."""

from __future__ import annotations

import math
import pytest

from autogen.intelligence.planning.contracts import Fact, GroundAction, PlanningProblem
from autogen.intelligence.planning.domains.survival import create_survival_crafting_problem
from autogen.intelligence.planning.heuristics import h_add, h_ff, h_max


def test_heuristic_zero_at_goal() -> None:
    """Invariant: All heuristics evaluate to 0.0 when goal conditions are already met."""
    problem = create_survival_crafting_problem()
    goal_state = problem.initial_facts | problem.goal_facts

    assert h_max(goal_state, problem) == 0.0
    assert h_add(goal_state, problem) == 0.0
    assert h_ff(goal_state, problem) == 0.0


def test_heuristic_dominance_h_max_le_h_add() -> None:
    """Mathematical invariant: h_max(S) <= h_add(S) for all states S."""
    problem = create_survival_crafting_problem()
    val_max = h_max(problem.initial_facts, problem)
    val_add = h_add(problem.initial_facts, problem)

    assert val_max > 0.0
    assert val_add >= val_max


def test_heuristic_detects_infinite_cost_dead_ends() -> None:
    """Invariant: Heuristics return math.inf for states with unreachable goals."""
    problem = PlanningProblem(
        initial_facts=frozenset({Fact("At", ("A",))}),
        goal_facts=frozenset({Fact("At", ("Z",))}),
        actions=(),  # No actions available
    )

    assert math.isinf(h_max(problem.initial_facts, problem))
    assert math.isinf(h_add(problem.initial_facts, problem))
    assert math.isinf(h_ff(problem.initial_facts, problem))

