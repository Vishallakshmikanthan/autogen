"""Planning Graph construction and Graphplan algorithm (SYS-08, Module M09)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from autogen.intelligence.planning.contracts import (
    Fact,
    GroundAction,
    Plan,
    PlanningMetrics,
    PlanningProblem,
)


@dataclass(frozen=True, slots=True)
class ActionLevel:
    """Action layer in a planning graph containing actions and mutex pairs."""

    actions: frozenset[GroundAction]
    mutex_pairs: frozenset[frozenset[GroundAction]]  # Each element is frozenset({a1, a2})


@dataclass(frozen=True, slots=True)
class PropositionLevel:
    """Proposition layer in a planning graph containing facts and mutex pairs."""

    facts: frozenset[Fact]
    mutex_pairs: frozenset[frozenset[Fact]]  # Each element is frozenset({p, q})


def _create_noop_action(fact: Fact) -> GroundAction:
    """Creates a persistence (no-op) action preserving fact across planning graph layers."""
    return GroundAction(
        name=f"NoOp({fact})",
        preconditions=frozenset({fact}),
        add_effects=frozenset({fact}),
        delete_effects=frozenset(),
        cost=0.0,
    )


class PlanningGraph:
    """Layered planning graph data structure maintaining alternating proposition and action levels."""

    def __init__(self, problem: PlanningProblem) -> None:
        self.problem = problem
        self.prop_levels: list[PropositionLevel] = [
            PropositionLevel(facts=problem.initial_facts, mutex_pairs=frozenset())
        ]
        self.action_levels: list[ActionLevel] = []

    def expand_level(self) -> None:
        """Expands the planning graph by one action level and one subsequent proposition level."""
        prev_prop = self.prop_levels[-1]

        # 1. Applicable actions: problem actions + no-ops
        applicable_actions: set[GroundAction] = set()

        # Problem actions whose preconditions are in prev_prop and not mutex
        for action in self.problem.actions:
            if action.preconditions.issubset(prev_prop.facts):
                # Preconditions must have no pairwise mutexes
                pre_mutex = False
                pre_list = list(action.preconditions)
                for i in range(len(pre_list)):
                    for j in range(i + 1, len(pre_list)):
                        if frozenset({pre_list[i], pre_list[j]}) in prev_prop.mutex_pairs:
                            pre_mutex = True
                            break
                    if pre_mutex:
                        break
                if not pre_mutex:
                    applicable_actions.add(action)

        # No-op actions for every fact in prev_prop
        for f in prev_prop.facts:
            applicable_actions.add(_create_noop_action(f))

        # 2. Compute Action Mutexes
        action_mutexes: set[frozenset[GroundAction]] = set()
        action_list = list(applicable_actions)

        for i in range(len(action_list)):
            for j in range(i + 1, len(action_list)):
                a1 = action_list[i]
                a2 = action_list[j]

                # Condition 1: Inconsistent Effects (a1 deletes an add effect of a2, or vice versa)
                if (a1.delete_effects & a2.add_effects) or (a2.delete_effects & a1.add_effects):
                    action_mutexes.add(frozenset({a1, a2}))
                    continue

                # Condition 2: Interference (a1 deletes a precondition of a2, or vice versa)
                if (a1.delete_effects & a2.preconditions) or (a2.delete_effects & a1.preconditions):
                    action_mutexes.add(frozenset({a1, a2}))
                    continue

                # Condition 3: Competing Needs (a precondition of a1 is mutex with a precondition of a2 in prev_prop)
                competing = False
                for p1 in a1.preconditions:
                    for p2 in a2.preconditions:
                        if frozenset({p1, p2}) in prev_prop.mutex_pairs:
                            competing = True
                            break
                    if competing:
                        break
                if competing:
                    action_mutexes.add(frozenset({a1, a2}))

        new_action_level = ActionLevel(
            actions=frozenset(applicable_actions),
            mutex_pairs=frozenset(action_mutexes),
        )
        self.action_levels.append(new_action_level)

        # 3. Next Proposition Level
        next_facts: set[Fact] = set()
        achievers: dict[Fact, list[GroundAction]] = {}

        for a in applicable_actions:
            for eff in a.add_effects:
                next_facts.add(eff)
                achievers.setdefault(eff, []).append(a)

        # 4. Compute Proposition Mutexes (Inconsistent Support)
        prop_mutexes: set[frozenset[Fact]] = set()
        next_fact_list = list(next_facts)

        for i in range(len(next_fact_list)):
            for j in range(i + 1, len(next_fact_list)):
                p = next_fact_list[i]
                q = next_fact_list[j]

                # All actions achieving p must be mutex with all actions achieving q
                all_mutex = True
                for a_p in achievers[p]:
                    for a_q in achievers[q]:
                        if a_p != a_q and frozenset({a_p, a_q}) not in action_mutexes:
                            all_mutex = False
                            break
                    if not all_mutex:
                        break

                if all_mutex:
                    prop_mutexes.add(frozenset({p, q}))

        new_prop_level = PropositionLevel(
            facts=frozenset(next_facts),
            mutex_pairs=frozenset(prop_mutexes),
        )
        self.prop_levels.append(new_prop_level)

    def goals_present_and_non_mutex(self, goals: frozenset[Fact]) -> bool:
        """Tests whether all goal facts are present in the latest layer without mutex conflicts."""
        latest_prop = self.prop_levels[-1]
        if not goals.issubset(latest_prop.facts):
            return False

        goal_list = list(goals)
        for i in range(len(goal_list)):
            for j in range(i + 1, len(goal_list)):
                if frozenset({goal_list[i], goal_list[j]}) in latest_prop.mutex_pairs:
                    return False
        return True

    def has_leveled_off(self) -> bool:
        """Determines if the planning graph has reached a fixed point (leveled off)."""
        if len(self.prop_levels) < 2:
            return False
        p_prev = self.prop_levels[-2]
        p_curr = self.prop_levels[-1]
        return p_prev.facts == p_curr.facts and p_prev.mutex_pairs == p_curr.mutex_pairs


def graphplan(problem: PlanningProblem, max_levels: int = 50) -> Plan:
    """Graphplan algorithm extracting conflict-free plans from a planning graph."""
    t0 = time.perf_counter()

    if problem.is_goal(problem.initial_facts):
        return Plan(
            is_solved=True,
            actions=(),
            total_cost=0.0,
            metrics=PlanningMetrics(nodes_expanded=0, nodes_generated=0, runtime_sec=time.perf_counter() - t0),
        )

    pg = PlanningGraph(problem)

    while len(pg.action_levels) < max_levels:
        if pg.goals_present_and_non_mutex(problem.goal_facts):
            # Attempt plan extraction via backward search
            level = len(pg.action_levels)
            plan_actions = _extract_plan(pg, level, problem.goal_facts)
            if plan_actions is not None:
                # Filter out no-ops
                concrete_actions = tuple(a for a in plan_actions if not a.name.startswith("NoOp("))
                total_cost = sum(a.cost for a in concrete_actions)
                t1 = time.perf_counter()
                return Plan(
                    is_solved=True,
                    actions=concrete_actions,
                    total_cost=total_cost,
                    metrics=PlanningMetrics(
                        nodes_expanded=level,
                        nodes_generated=len(pg.action_levels),
                        runtime_sec=t1 - t0,
                    ),
                )

        if pg.has_leveled_off():
            # Fixed point reached without solution -> provably unsolvable
            t1 = time.perf_counter()
            return Plan(
                is_solved=False,
                actions=(),
                total_cost=0.0,
                metrics=PlanningMetrics(
                    nodes_expanded=len(pg.action_levels),
                    nodes_generated=len(pg.action_levels),
                    runtime_sec=t1 - t0,
                ),
                is_proven_unsolvable=True,
            )

        pg.expand_level()

    t1 = time.perf_counter()
    return Plan(
        is_solved=False,
        actions=(),
        total_cost=0.0,
        metrics=PlanningMetrics(
            nodes_expanded=len(pg.action_levels),
            nodes_generated=len(pg.action_levels),
            runtime_sec=t1 - t0,
        ),
    )


def _extract_plan(
    pg: PlanningGraph,
    level: int,
    subgoals: frozenset[Fact],
) -> list[GroundAction] | None:
    """Backward recursive search extracting conflict-free actions supporting subgoals."""
    if level == 0:
        return [] if subgoals.issubset(pg.prop_levels[0].facts) else None

    action_level = pg.action_levels[level - 1]
    achievers: dict[Fact, list[GroundAction]] = {
        g: [a for a in action_level.actions if g in a.add_effects]
        for g in subgoals
    }

    subgoal_list = list(subgoals)

    def find_action_set(
        idx: int,
        selected_actions: set[GroundAction],
    ) -> list[GroundAction] | None:
        if idx == len(subgoal_list):
            # All subgoals covered! Collect new subgoals at level-1
            prev_subgoals: set[Fact] = set()
            for a in selected_actions:
                prev_subgoals.update(a.preconditions)

            # Check if previous subgoals have any mutex conflicts at level-1
            prev_prop = pg.prop_levels[level - 1]
            prev_list = list(prev_subgoals)
            for i in range(len(prev_list)):
                for j in range(i + 1, len(prev_list)):
                    if frozenset({prev_list[i], prev_list[j]}) in prev_prop.mutex_pairs:
                        return None

            earlier_plan = _extract_plan(pg, level - 1, frozenset(prev_subgoals))
            if earlier_plan is not None:
                return earlier_plan + list(selected_actions)
            return None

        current_goal = subgoal_list[idx]
        # Check if already satisfied by an already-selected action
        if any(current_goal in a.add_effects for a in selected_actions):
            return find_action_set(idx + 1, selected_actions)

        for candidate_act in achievers.get(current_goal, []):
            # Candidate must not be mutex with any already selected action
            if any(frozenset({candidate_act, other}) in action_level.mutex_pairs for other in selected_actions):
                continue

            res = find_action_set(idx + 1, selected_actions | {candidate_act})
            if res is not None:
                return res

        return None

    return find_action_set(0, set())

