"""Property-based and adversarial tests for the Symbolic Safety Shield (SYS-15, Module M17)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.hybrid.safety_shield import SymbolicSafetyShield
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position


def test_safety_shield_non_interference_when_safe() -> None:
    """Verifies that safe actions pass through without alteration or vetoes."""
    shield = SymbolicSafetyShield(hazard_threshold=0.5, grid_width=10, grid_height=10)

    obs = Observation(
        agent_id="agent",
        step=1,
        position=Position(x=5, y=5),
        energy=100.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    action, record = shield.verify_and_shield(
        step=1,
        proposed_action=ActionType.UP,
        observation=obs,
        hazard_probabilities={},
    )

    assert action == ActionType.UP
    assert record is None
    assert shield.veto_count == 0


def test_safety_shield_vetoes_out_of_bounds() -> None:
    """Verifies that shield vetoes moves outside environment grid boundaries."""
    shield = SymbolicSafetyShield(hazard_threshold=0.5, grid_width=10, grid_height=10)

    # Agent at top edge (y=0) attempting to move UP
    obs_top = Observation(
        agent_id="agent",
        step=1,
        position=Position(x=5, y=0),
        energy=100.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    action, record = shield.verify_and_shield(
        step=1,
        proposed_action=ActionType.UP,
        observation=obs_top,
        hazard_probabilities={},
    )

    assert action != ActionType.UP
    assert record is not None
    assert "OUT_OF_BOUNDS" in record.violated_rule
    assert shield.veto_count == 1


def test_safety_shield_zero_hazard_violations_property() -> None:
    """Invariant: Over 10,000 random/adversarial states, zero moves into lethal hazard cells permitted."""
    shield = SymbolicSafetyShield(hazard_threshold=0.5, grid_width=16, grid_height=16)
    rng = np.random.default_rng(42)

    actions = [ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT, ActionType.STAY]
    violations_permitted = 0

    for step in range(10_000):
        # Random agent position
        px = int(rng.integers(0, 16))
        py = int(rng.integers(0, 16))
        curr_pos = Position(x=px, y=py)

        # Random hazard field
        hazards: dict[Position, float] = {}
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            h_pos = Position(x=px + dx, y=py + dy)
            if rng.random() < 0.4:
                hazards[h_pos] = float(rng.uniform(0.6, 1.0))  # High hazard

        obs = Observation(
            agent_id="agent",
            step=step,
            position=curr_pos,
            energy=100.0,
            health=100.0,
            carried_food=0,
            visible_cells=[],
            visible_agents=[],
        )

        proposed_act = actions[int(rng.integers(0, len(actions)))]

        final_act, _ = shield.verify_and_shield(
            step=step,
            proposed_action=proposed_act,
            observation=obs,
            hazard_probabilities=hazards,
        )

        final_target = shield.get_target_position(curr_pos, final_act)

        # Invariant check: target must NOT be out-of-bounds
        assert shield.is_within_bounds(final_target), f"Out of bounds move permitted: {final_target}"

        # Invariant check: target must NOT be known lethal hazard if any safer alternative existed
        if hazards.get(final_target, 0.0) >= 0.5:
            # Check if there was any legal candidate with lower risk
            cand_risks = [
                hazards.get(shield.get_target_position(curr_pos, a), 0.0)
                for a in actions
                if shield.is_within_bounds(shield.get_target_position(curr_pos, a))
            ]
            if min(cand_risks) < 0.5:
                violations_permitted += 1

    assert violations_permitted == 0, f"Safety invariant violated: {violations_permitted} times"

