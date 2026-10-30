"""Strategic forager agent solving contested resource interactions via normal-form matrix games (SYS-10, Module M07)."""

from __future__ import annotations

import random
from typing import Sequence

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.game_theory.contracts import NormalFormGame
from autogen.intelligence.game_theory.matrix_games import make_hawk_dove
from autogen.intelligence.game_theory.solvers import (
    find_pure_nash_equilibria,
    solve_2x2_mixed_nash,
)
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position, TERRAIN_PROPERTIES


class StrategicForager(BaseAgent):
    """Forager agent resolving resource competition through game-theoretic matrix duel strategies."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities | None = None,
        game_type: str = "hawk_dove",
        strategy_mode: str = "mixed_nash",  # 'mixed_nash', 'hawk', 'dove', or 'tit_for_tat'
        seed: int = 42,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(planning=True))
        self.game_type = game_type
        self.strategy_mode = strategy_mode
        self.rng = random.Random(seed)
        self.game = make_hawk_dove(resource_value=4.0, injury_cost=6.0)

        self.visible_food: dict[Position, float] = {}
        self.blocked_cells: set[Position] = set()
        self.competitor_history: dict[str, list[int]] = {}  # agent_id -> list of past actions (0: Hawk, 1: Dove)
        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

        self.visible_food.clear()
        for cell in observation.visible_cells:
            if not TERRAIN_PROPERTIES[cell.terrain].walkable:
                self.blocked_cells.add(cell.position)
            if cell.food > 0.0:
                self.visible_food[cell.position] = cell.food

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            trace = (
                DecisionTraceBuilder(self.agent_id, self._step_counter, "GameTheoreticForager")
                .set_selected_action(ActionType.STAY, "No observation available")
                .build()
            )
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        my_pos = self._last_observation.position
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "GameTheoreticForager")
        trace_builder.set_observation_summary(self._last_observation)

        # 1. If currently on a food cell and carrying capacity allows, collect food
        current_cell_food = self.visible_food.get(my_pos, 0.0)
        if current_cell_food > 0.0 and not self._last_observation.visible_agents:
            trace_builder.add_candidate_action(ActionType.COLLECT, current_cell_food)
            trace_builder.set_selected_action(ActionType.COLLECT, "Collecting uncontested food")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.COLLECT), trace

        # 2. Check for resource competition with nearby visible agents (within distance 2)
        contesting_agent_id = None
        contesting_agent_pos = None
        for other_id, other_pos in self._last_observation.visible_agents:
            if my_pos.manhattan_distance(other_pos) <= 2:
                contesting_agent_id = other_id
                contesting_agent_pos = other_pos
                break

        if contesting_agent_id is not None and contesting_agent_pos is not None:
            # Solve game-theoretic matrix game to determine disposition: 0 = Hawk (fight for food), 1 = Dove (yield/retreat)
            disposition = self._resolve_matrix_disposition(contesting_agent_id)
            disposition_name = self.game.actions[0][disposition]

            trace_builder.add_metric("contesting_agent_id", contesting_agent_id)
            trace_builder.add_metric("strategic_disposition", disposition_name)

            if disposition == 0:  # Hawk: pursue food aggressively even if contested
                target = self._find_closest_food(my_pos) or contesting_agent_pos
                chosen_act = self._step_towards(my_pos, target)
                reason = f"Strategic Hawk disposition against {contesting_agent_id}"
            else:  # Dove: yield contested territory and move away
                chosen_act = self._step_away_from(my_pos, contesting_agent_pos)
                reason = f"Strategic Dove disposition yielding to {contesting_agent_id}"

            trace_builder.add_candidate_action(chosen_act, 1.0)
            trace_builder.set_selected_action(chosen_act, reason)
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=chosen_act), trace

        # 3. Standard uncontested foraging: move toward closest visible food
        closest_food = self._find_closest_food(my_pos)
        if closest_food is not None:
            chosen_act = self._step_towards(my_pos, closest_food)
            trace_builder.add_candidate_action(chosen_act, 1.0)
            trace_builder.set_selected_action(chosen_act, f"Navigating towards food at {closest_food}")
        else:
            chosen_act = ActionType.STAY
            trace_builder.add_candidate_action(chosen_act, 0.0)
            trace_builder.set_selected_action(chosen_act, "No food visible, resting")

        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_act), trace

    def _resolve_matrix_disposition(self, rival_id: str) -> int:
        """Determines action index (0: Hawk, 1: Dove) using selected game theory strategy."""
        if self.strategy_mode == "hawk":
            return 0
        elif self.strategy_mode == "dove":
            return 1
        elif self.strategy_mode == "tit_for_tat":
            history = self.competitor_history.get(rival_id, [])
            if not history:
                return 1  # Cooperate (Dove) first
            return history[-1]  # Mimic opponent's previous choice
        else:
            # mixed_nash equilibrium of Hawk-Dove:
            # Hawk prob p = V / C = 4 / 6 = 2/3
            eqs = solve_2x2_mixed_nash(self.game)
            if eqs:
                p_hawk = float(eqs[0].probabilities[0][0])
            else:
                p_hawk = 0.5
            return 0 if self.rng.random() < p_hawk else 1

    def _find_closest_food(self, my_pos: Position) -> Position | None:
        best_dist = float("inf")
        best_pos = None
        for fpos in self.visible_food:
            dist = my_pos.manhattan_distance(fpos)
            if dist < best_dist:
                best_dist = dist
                best_pos = fpos
        return best_pos

    def _step_towards(self, current: Position, target: Position) -> ActionType:
        dx = target.x - current.x
        dy = target.y - current.y
        candidates: list[tuple[ActionType, Position]] = []
        if abs(dx) >= abs(dy) and dx != 0:
            act = ActionType.RIGHT if dx > 0 else ActionType.LEFT
            candidates.append((act, Position(x=current.x + (1 if dx > 0 else -1), y=current.y)))
        if dy != 0:
            act = ActionType.DOWN if dy > 0 else ActionType.UP
            candidates.append((act, Position(x=current.x, y=current.y + (1 if dy > 0 else -1))))

        for act, next_pos in candidates:
            if next_pos not in self.blocked_cells:
                return act
        return ActionType.STAY

    def _step_away_from(self, current: Position, rival: Position) -> ActionType:
        # Move in opposite direction from rival
        dx = current.x - rival.x
        dy = current.y - rival.y
        candidates: list[tuple[ActionType, Position]] = []
        if dx != 0:
            act = ActionType.RIGHT if dx > 0 else ActionType.LEFT
            candidates.append((act, Position(x=current.x + (1 if dx > 0 else -1), y=current.y)))
        if dy != 0:
            act = ActionType.DOWN if dy > 0 else ActionType.UP
            candidates.append((act, Position(x=current.x, y=current.y + (1 if dy > 0 else -1))))

        for act, next_pos in candidates:
            if next_pos not in self.blocked_cells:
                return act
        return ActionType.STAY

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.visible_food.clear()
        self.blocked_cells.clear()
        self.competitor_history.clear()
        self.latest_trace = None
