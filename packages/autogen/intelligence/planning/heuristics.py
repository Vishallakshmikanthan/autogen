"""Domain-independent delete-relaxation heuristics for classical planning (SYS-08, Module M09)."""

from __future__ import annotations

import math
from typing import Mapping

from autogen.intelligence.planning.contracts import Fact, GroundAction, PlanningProblem


def compute_relaxed_costs(
    state_facts: frozenset[Fact],
    problem: PlanningProblem,
) -> dict[Fact, float]:
    """Computes relaxed reachability costs for all reachable facts from the given state.

    Uses Bellman-Ford fixed-point iteration over the delete-relaxed problem P^+.
    """
    costs: dict[Fact, float] = {f: 0.0 for f in state_facts}

    changed = True
    while changed:
        changed = False
        for action in problem.actions:
            # Check if all preconditions have finite cost
            if all(pre in costs for pre in action.preconditions):
                pre_cost = (
                    max(costs[pre] for pre in action.preconditions)
                    if action.preconditions
                    else 0.0
                )
                action_achieve_cost = pre_cost + action.cost

                for eff in action.add_effects:
                    if eff not in costs or action_achieve_cost < costs[eff]:
                        costs[eff] = action_achieve_cost
                        changed = True

    return costs


def h_max(state_facts: frozenset[Fact], problem: PlanningProblem) -> float:
    """Admissible delete-relaxation heuristic: h_max(S) = max_{g in G} cost^+(g).

    Guarantees h_max(S) <= h*(S), preserving A* optimality.
    Returns math.inf if any goal fact is unreachable (dead end).
    """
    if problem.is_goal(state_facts):
        return 0.0

    costs = compute_relaxed_costs(state_facts, problem)

    max_cost = 0.0
    for g in problem.goal_facts:
        if g not in costs:
            return math.inf  # Unreachable goal fact -> state is dead end
        if costs[g] > max_cost:
            max_cost = costs[g]

    return max_cost


def h_add(state_facts: frozenset[Fact], problem: PlanningProblem) -> float:
    """Additive delete-relaxation heuristic: h_add(S) = sum_{g in G} cost^+(g).

    Assumes subgoals are independent; may overestimate h*(S) when subgoals share actions,
    making it inadmissible but effective for greedy forward search.
    """
    if problem.is_goal(state_facts):
        return 0.0

    costs = compute_relaxed_costs(state_facts, problem)

    total_cost = 0.0
    for g in problem.goal_facts:
        if g not in costs:
            return math.inf
        total_cost += costs[g]

    return total_cost


def h_ff(state_facts: frozenset[Fact], problem: PlanningProblem) -> float:
    """Fast-Forward (FF) heuristic: extracts a relaxed plan from the relaxed planning graph

    and returns the number of actions in the relaxed plan.
    """
    if problem.is_goal(state_facts):
        return 0.0

    # 1. Build layered relaxed planning graph
    fact_layers: list[set[Fact]] = [set(state_facts)]
    action_layers: list[set[GroundAction]] = []

    # Map each fact to the first layer it appeared in and an achiever action
    first_achiever: dict[Fact, GroundAction | None] = {f: None for f in state_facts}

    while not problem.goal_facts.issubset(fact_layers[-1]):
        current_facts = fact_layers[-1]
        applicable_actions: set[GroundAction] = set()

        for a in problem.actions:
            if a.preconditions.issubset(current_facts):
                applicable_actions.add(a)

        next_facts = set(current_facts)
        new_facts_added = False

        for a in applicable_actions:
            for eff in a.add_effects:
                if eff not in first_achiever:
                    first_achiever[eff] = a
                    next_facts.add(eff)
                    new_facts_added = True

        if not new_facts_added and not problem.goal_facts.issubset(next_facts):
            return math.inf  # Goal cannot be reached in relaxed graph

        action_layers.append(applicable_actions)
        fact_layers.append(next_facts)

    # 2. Extract relaxed plan via backward search
    relaxed_plan_actions: set[GroundAction] = set()
    goals_to_support: set[Fact] = set(problem.goal_facts) - state_facts

    while goals_to_support:
        g = goals_to_support.pop()
        achiever = first_achiever.get(g)
        if achiever is not None and achiever not in relaxed_plan_actions:
            relaxed_plan_actions.add(achiever)
            for pre in achiever.preconditions:
                if pre not in state_facts:
                    goals_to_support.add(pre)

    return float(len(relaxed_plan_actions))
