"""Unit tests for the independent PlanValidator and mutation sensitivity (Module M09)."""

from __future__ import annotations

import pytest

from autogen.intelligence.planning.domains.survival import create_survival_crafting_problem
from autogen.intelligence.planning.forward import ForwardStateSpacePlanner
from autogen.intelligence.planning.validator import PlanValidator


def test_plan_validator_accepts_valid_plan() -> None:
    """Verifies that PlanValidator confirms a valid forward A* plan."""
    problem = create_survival_crafting_problem()
    planner = ForwardStateSpacePlanner()
    plan = planner.solve_astar(problem)

    assert plan.is_solved is True
    valid, final_state, err = PlanValidator.validate_plan(problem, plan.actions)
    assert valid is True
    assert err is None
    assert problem.goal_facts.issubset(final_state)


def test_plan_validator_rejects_mutated_plans() -> None:
    """Mutation testing: Asserts that PlanValidator rejects 100% of mutated plans."""
    problem = create_survival_crafting_problem()
    planner = ForwardStateSpacePlanner()
    plan = planner.solve_astar(problem)
    assert plan.is_solved is True

    original_actions = list(plan.actions)
    assert len(original_actions) >= 3

    # Mutation 1: Drop an action from the middle
    mutated_drop = original_actions[:2] + original_actions[3:]
    v1, _, err1 = PlanValidator.validate_plan(problem, mutated_drop)
    assert v1 is False, "Validator failed to catch dropped action mutation!"
    assert err1 is not None

    # Mutation 2: Swap two adjacent actions that have ordering dependencies
    # (e.g. swap BuildShelter with SurviveWinter)
    mutated_swap = list(original_actions)
    mutated_swap[-1], mutated_swap[-2] = mutated_swap[-2], mutated_swap[-1]
    v2, _, err2 = PlanValidator.validate_plan(problem, mutated_swap)
    assert v2 is False, "Validator failed to catch swapped action dependency mutation!"
    assert err2 is not None
