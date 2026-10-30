"""Unit tests for STRIPS state-space planning (BFS, A*, and Dead-End detection)."""

from __future__ import annotations

import pytest

from autogen.intelligence.planning.contracts import (
    Fact,
    GroundAction,
    PlanningProblem,
)
from autogen.intelligence.planning.domains.survival import create_survival_crafting_problem
from autogen.intelligence.planning.forward import ForwardStateSpacePlanner
from autogen.intelligence.planning.heuristics import h_max
from autogen.intelligence.planning.validator import PlanValidator


def test_strips_action_application() -> None:
    """Tests action applicability and state progression semantics."""
    s0 = frozenset({Fact("At", ("A",)), Fact("Clean", ("A",))})

    act_move = GroundAction(
        name="Move(A, B)",
        preconditions=frozenset({Fact("At", ("A",))}),
        add_effects=frozenset({Fact("At", ("B",))}),
        delete_effects=frozenset({Fact("At", ("A",))}),
    )

    assert act_move.is_applicable(s0) is True
    s1 = act_move.apply(s0)
    assert s1 == frozenset({Fact("At", ("B",)), Fact("Clean", ("A",))})

    # Cannot move from A when at B
    assert act_move.is_applicable(s1) is False


def test_bfs_vs_astar_plan_optimality_agreement() -> None:
    """Oracle test: Forward A* with admissible h_max produces exact optimal plan length matching BFS."""
    # Simple blocks-world / package delivery problem
    p1 = Fact("At", ("Pkg1", "LocA"))
    truck = Fact("At", ("Truck", "LocA"))
    goal = Fact("At", ("Pkg1", "LocB"))

    load = GroundAction(
        name="Load",
        preconditions=frozenset({p1, truck}),
        add_effects=frozenset({Fact("In", ("Pkg1", "Truck"))}),
        delete_effects=frozenset({p1}),
    )
    drive = GroundAction(
        name="Drive",
        preconditions=frozenset({truck}),
        add_effects=frozenset({Fact("At", ("Truck", "LocB"))}),
        delete_effects=frozenset({truck}),
    )
    unload = GroundAction(
        name="Unload",
        preconditions=frozenset({Fact("In", ("Pkg1", "Truck")), Fact("At", ("Truck", "LocB"))}),
        add_effects=frozenset({goal}),
        delete_effects=frozenset({Fact("In", ("Pkg1", "Truck"))}),
    )

    problem = PlanningProblem(
        initial_facts=frozenset({p1, truck}),
        goal_facts=frozenset({goal}),
        actions=(load, drive, unload),
    )

    planner = ForwardStateSpacePlanner(heuristic_fn=h_max)
    plan_bfs = planner.solve_bfs(problem)
    plan_astar = planner.solve_astar(problem)

    assert plan_bfs.is_solved is True
    assert plan_astar.is_solved is True
    assert len(plan_bfs.actions) == len(plan_astar.actions) == 3
    assert plan_bfs.total_cost == plan_astar.total_cost == 3.0


def test_unsolvable_planning_problem_detected() -> None:
    """Verifies that an impossible planning problem is proven unsolvable."""
    p_init = frozenset({Fact("At", ("Home",))})
    p_goal = frozenset({Fact("HasTreasure", ())})

    # No action achieves HasTreasure
    actions = (
        GroundAction(
            name="Walk",
            preconditions=frozenset({Fact("At", ("Home",))}),
            add_effects=frozenset({Fact("At", ("Park",))}),
            delete_effects=frozenset({Fact("At", ("Home",))}),
        ),
    )

    problem = PlanningProblem(initial_facts=p_init, goal_facts=p_goal, actions=actions)
    planner = ForwardStateSpacePlanner()
    plan = planner.solve_astar(problem)

    assert plan.is_solved is False
    assert plan.is_proven_unsolvable is True
