"""Independent Plan Validator (SYS-08, Module M09)."""

from __future__ import annotations

from typing import Sequence

from autogen.intelligence.planning.contracts import Fact, GroundAction, PlanningProblem


class PlanValidator:
    """Independent validator checking plan validity against STRIPS semantics."""

    @staticmethod
    def validate_plan(
        problem: PlanningProblem,
        actions: Sequence[GroundAction],
    ) -> tuple[bool, frozenset[Fact], str | None]:
        """Validates that `actions` execute legitimately from problem.initial_facts and achieve problem.goal_facts.

        Returns:
            (is_valid, final_state_facts, failure_reason)
        """
        current_state = set(problem.initial_facts)

        for step_idx, action in enumerate(actions):
            # 1. Precondition check
            missing_preconditions = action.preconditions - current_state
            if missing_preconditions:
                return (
                    False,
                    frozenset(current_state),
                    f"Step {step_idx} ({action.name}) failed: missing preconditions {missing_preconditions}",
                )

            # 2. State transition
            current_state.difference_update(action.delete_effects)
            current_state.update(action.add_effects)

        # 3. Goal satisfaction check
        final_facts = frozenset(current_state)
        missing_goals = problem.goal_facts - final_facts
        if missing_goals:
            return (
                False,
                final_facts,
                f"Plan completed but failed to satisfy goal facts: {missing_goals}",
            )

        return True, final_facts, None
