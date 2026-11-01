"""Symbolic Safety Shield providing runtime constraint verification and non-bypassable overrides (SYS-15, Module M17)."""

from __future__ import annotations

from autogen.intelligence.hybrid.contracts import ShieldViolationRecord
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position


class SymbolicSafetyShield:
    """Runtime verification safety shield providing formal vetoes and fallbacks against hazardous moves."""

    def __init__(
        self,
        hazard_threshold: float = 0.5,
        grid_width: int = 32,
        grid_height: int = 32,
    ) -> None:
        self.hazard_threshold = hazard_threshold
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.veto_count = 0
        self.history: list[ShieldViolationRecord] = []

    def get_target_position(self, current_pos: Position, action: ActionType) -> Position:
        """Computes intended target position assuming standard directional transitions."""
        if action == ActionType.UP:
            return Position(x=current_pos.x, y=current_pos.y - 1)
        elif action == ActionType.DOWN:
            return Position(x=current_pos.x, y=current_pos.y + 1)
        elif action == ActionType.LEFT:
            return Position(x=current_pos.x - 1, y=current_pos.y)
        elif action == ActionType.RIGHT:
            return Position(x=current_pos.x + 1, y=current_pos.y)
        else:
            return current_pos

    def is_within_bounds(self, pos: Position) -> bool:
        """Verifies target coordinates lie strictly within environment grid boundaries."""
        return 0 <= pos.x < self.grid_width and 0 <= pos.y < self.grid_height

    def verify_and_shield(
        self,
        step: int,
        proposed_action: ActionType,
        observation: Observation,
        hazard_probabilities: dict[Position, float],
    ) -> tuple[ActionType, ShieldViolationRecord | None]:
        """Validates proposed action. If safety constraints are violated, vetoes and substitutes safe fallback."""
        curr_pos = observation.position
        target_pos = self.get_target_position(curr_pos, proposed_action)

        violation_reason = ""
        risk_score = 0.0

        # Rule 1: Out of bounds check
        if not self.is_within_bounds(target_pos):
            violation_reason = "OUT_OF_BOUNDS_VIOLATION"
            risk_score = 1.0

        # Rule 2: Lethal hazard risk check
        hazard_p = hazard_probabilities.get(target_pos, 0.0)
        if hazard_p >= self.hazard_threshold:
            violation_reason = f"LETHAL_HAZARD_RISK_{hazard_p:.2f}"
            risk_score = hazard_p

        # Rule 3: Direct hazard visible cell check
        for cell in observation.visible_cells:
            is_haz = getattr(cell, "danger", 0.0) >= 0.5 or getattr(cell, "is_hazard", False)
            if cell.position == target_pos and is_haz:
                violation_reason = "DIRECT_HAZARD_CELL_VISIBLE"
                risk_score = 1.0
                break

        # If action is safe, permit execution unmodified
        if not violation_reason:
            return proposed_action, None

        # Veto: search for safest alternative action
        candidates = [
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
            ActionType.STAY,
        ]

        safest_action = ActionType.STAY
        lowest_candidate_risk = float("inf")

        for act in candidates:
            cand_target = self.get_target_position(curr_pos, act)
            if not self.is_within_bounds(cand_target):
                continue
            cand_p = hazard_probabilities.get(cand_target, 0.0)

            # Check direct visible hazards
            is_vis_hazard = any(
                c.position == cand_target and (getattr(c, "danger", 0.0) >= 0.5 or getattr(c, "is_hazard", False))
                for c in observation.visible_cells
            )
            if is_vis_hazard:
                cand_p = 1.0

            if cand_p < lowest_candidate_risk:
                lowest_candidate_risk = cand_p
                safest_action = act

        self.veto_count += 1
        record = ShieldViolationRecord(
            step=step,
            proposed_action=proposed_action,
            safe_action=safest_action,
            violated_rule=violation_reason,
            hazard_risk=risk_score,
            target_position=target_pos,
        )
        self.history.append(record)
        return safest_action, record
