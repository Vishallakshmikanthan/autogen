"""Hierarchical Options Framework and Semi-Markov Decision Process (SMDP) engine (SYS-15, Module M17)."""

from __future__ import annotations

from typing import Any

import numpy as np

from autogen.intelligence.hybrid.contracts import Option
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position


class ForageOption:
    """Option omega_forage: initiates when food is visible, navigates to food, terminates when eaten."""

    option_id = "FORAGE"
    description = "Navigates towards visible food resource patches."

    def can_initiate(self, state: dict[str, Any], observation: Observation) -> bool:
        """Initiates if visible cells contain at least one resource patch."""
        if not observation.visible_cells:
            return False
        return any(getattr(c, "food", getattr(c, "resource", 0)) > 0 for c in observation.visible_cells)

    def select_action(self, state: dict[str, Any], observation: Observation) -> ActionType:
        """Heuristic subpolicy: steps toward closest resource cell."""
        food_cells = [
            c for c in observation.visible_cells if getattr(c, "food", getattr(c, "resource", 0)) > 0
        ]
        if not food_cells:
            return ActionType.STAY

        target = food_cells[0]
        dx = target.position.x - observation.position.x
        dy = target.position.y - observation.position.y

        if abs(dx) > abs(dy):
            return ActionType.RIGHT if dx > 0 else ActionType.LEFT
        elif dy != 0:
            return ActionType.DOWN if dy > 0 else ActionType.UP
        return ActionType.STAY

    def should_terminate(self, state: dict[str, Any], observation: Observation) -> bool:
        """Terminates when food is reached (at agent position) or no food visible."""
        food_cells = [
            c for c in observation.visible_cells if getattr(c, "food", getattr(c, "resource", 0)) > 0
        ]
        if not food_cells:
            return True
        # Terminate if standing directly on a food cell
        return any(c.position == observation.position for c in food_cells)


class FleeHazardOption:
    """Option omega_flee: initiates when nearby hazard probability > threshold, steers away."""

    option_id = "FLEE_HAZARD"
    description = "Steers away from high-risk hazard belief areas."

    def __init__(self, risk_threshold: float = 0.5) -> None:
        self.risk_threshold = risk_threshold

    def can_initiate(self, state: dict[str, Any], observation: Observation) -> bool:
        hazards = state.get("hazard_probabilities", {})
        if not hazards:
            return False
        return any(p >= self.risk_threshold for p in hazards.values())

    def select_action(self, state: dict[str, Any], observation: Observation) -> ActionType:
        hazards: dict[Position, float] = state.get("hazard_probabilities", {})
        pos = observation.position

        # Evaluate risk of 4 candidate directions
        dirs = [
            (ActionType.UP, Position(x=pos.x, y=pos.y - 1)),
            (ActionType.DOWN, Position(x=pos.x, y=pos.y + 1)),
            (ActionType.LEFT, Position(x=pos.x - 1, y=pos.y)),
            (ActionType.RIGHT, Position(x=pos.x + 1, y=pos.y)),
        ]

        safest_action = ActionType.STAY
        lowest_risk = float("inf")

        for act, next_pos in dirs:
            risk = hazards.get(next_pos, 0.0)
            if risk < lowest_risk:
                lowest_risk = risk
                safest_action = act

        return safest_action

    def should_terminate(self, state: dict[str, Any], observation: Observation) -> bool:
        hazards = state.get("hazard_probabilities", {})
        pos = observation.position
        # Terminate when immediate neighbourhood risk is low
        local_risks = [
            hazards.get(Position(x=pos.x + dx, y=pos.y + dy), 0.0)
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
        ]
        return all(r < self.risk_threshold for r in local_risks)


class WanderOption:
    """Option omega_wander: default exploratory walk with directional inertia."""

    option_id = "WANDER"
    description = "Default exploratory random walk."

    def __init__(self) -> None:
        self.step_count = 0
        self.dirs = [ActionType.UP, ActionType.RIGHT, ActionType.DOWN, ActionType.LEFT]

    def can_initiate(self, state: dict[str, Any], observation: Observation) -> bool:
        return True  # Always initiable as fallback

    def select_action(self, state: dict[str, Any], observation: Observation) -> ActionType:
        self.step_count += 1
        return self.dirs[(self.step_count // 2) % len(self.dirs)]

    def should_terminate(self, state: dict[str, Any], observation: Observation) -> bool:
        """Terminates after 5 steps to allow re-evaluating high-level options."""
        return (self.step_count % 5) == 0


class SMDPQLearning:
    """Semi-Markov Decision Process (SMDP) high-level Q-learning over options."""

    def __init__(
        self,
        options: list[Option],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.options = options
        self.option_map = {opt.option_id: opt for opt in options}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Q-table: Q[state_key, option_id]
        self.q_table: dict[str, dict[str, float]] = {}

        # Active option execution tracking
        self.active_option: Option | None = None
        self.option_start_state_key: str | None = None
        self.option_duration: int = 0
        self.option_discounted_return: float = 0.0

    def _get_q_row(self, state_key: str) -> dict[str, float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = {opt.option_id: 0.0 for opt in self.options}
        return self.q_table[state_key]

    def select_option(
        self,
        state_key: str,
        state: dict[str, Any],
        observation: Observation,
    ) -> Option:
        """Selects an option from initiation set using epsilon-greedy SMDP Q-values."""
        available = [
            opt for opt in self.options if opt.can_initiate(state, observation)
        ]
        if not available:
            # Fallback to first option if none explicitly initiable
            available = [self.options[0]]

        if self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(0, len(available)))
            return available[idx]

        q_row = self._get_q_row(state_key)
        best_option = max(available, key=lambda opt: q_row.get(opt.option_id, 0.0))
        return best_option

    def step(
        self,
        state_key: str,
        state: dict[str, Any],
        observation: Observation,
    ) -> tuple[ActionType, Option, bool]:
        """Manages option execution, termination checking, and subpolicy action selection."""
        # Check if active option should terminate
        just_initiated = False
        if (
            self.active_option is None
            or self.active_option.should_terminate(state, observation)
        ):
            # Terminate previous option if one was running
            if self.active_option is not None and self.option_start_state_key is not None:
                self._update_smdp(terminal_state_key=state_key)

            # Initiate new option
            self.active_option = self.select_option(state_key, state, observation)
            self.option_start_state_key = state_key
            self.option_duration = 0
            self.option_discounted_return = 0.0
            just_initiated = True

        # Select primitive action from active option subpolicy
        action = self.active_option.select_action(state, observation)
        return action, self.active_option, just_initiated

    def record_step_reward(self, reward: float) -> None:
        """Accumulates discounted reward over option duration: sum gamma^k r_k."""
        discount = self.gamma**self.option_duration
        self.option_discounted_return += discount * reward
        self.option_duration += 1

    def _update_smdp(self, terminal_state_key: str) -> None:
        """Updates SMDP Q-value upon option termination: Q(s, w) += alpha [r_s^w + gamma^tau max Q(s', w') - Q(s, w)]."""
        if self.active_option is None or self.option_start_state_key is None:
            return

        s = self.option_start_state_key
        w = self.active_option.option_id
        r_w = self.option_discounted_return
        tau = self.option_duration

        q_row = self._get_q_row(s)
        current_q = q_row[w]

        # max_w' Q(s', w')
        next_q_row = self._get_q_row(terminal_state_key)
        max_next_q = max(next_q_row.values()) if next_q_row else 0.0

        # SMDP Bellman update
        target = r_w + (self.gamma**tau) * max_next_q
        q_row[w] += self.alpha * (target - current_q)
