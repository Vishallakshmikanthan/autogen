"""Unit tests for Planning Graph construction and the Graphplan algorithm (Module M09)."""

from __future__ import annotations

import pytest

from autogen.intelligence.planning.contracts import Fact, GroundAction, PlanningProblem
from autogen.intelligence.planning.graphplan import PlanningGraph, graphplan
from autogen.intelligence.planning.validator import PlanValidator


def test_graphplan_simple_problem() -> None:
    """Verifies that Graphplan extracts a valid parallel/serial plan."""
    have_cake = Fact("HaveCake", ())
    eaten_cake = Fact("EatenCake", ())

    eat = GroundAction(
        name="EatCake",
        preconditions=frozenset({have_cake}),
        add_effects=frozenset({eaten_cake}),
        delete_effects=frozenset({have_cake}),
    )
    bake = GroundAction(
        name="BakeCake",
        preconditions=frozenset(),
        add_effects=frozenset({have_cake}),
        delete_effects=frozenset(),
    )

    # Goal: HaveCake and EatenCake!
    # Valid plan: EatCake then BakeCake
    problem = PlanningProblem(
        initial_facts=frozenset({have_cake}),
        goal_facts=frozenset({have_cake, eaten_cake}),
        actions=(eat, bake),
    )

    plan = graphplan(problem, max_levels=10)
    assert plan.is_solved is True
    valid, final_state, err = PlanValidator.validate_plan(problem, plan.actions)
    assert valid is True, f"Graphplan cake plan invalid: {err}"


def test_graphplan_unsolvable_leveled_off() -> None:
    """Verifies that Graphplan terminates with is_proven_unsolvable when the graph levels off."""
    p_init = frozenset({Fact("At", ("A",))})
    p_goal = frozenset({Fact("At", ("Unreachable",))})

    problem = PlanningProblem(
        initial_facts=p_init,
        goal_facts=p_goal,
        actions=(),
    )

    plan = graphplan(problem, max_levels=5)
    assert plan.is_solved is False
    assert plan.is_proven_unsolvable is True
