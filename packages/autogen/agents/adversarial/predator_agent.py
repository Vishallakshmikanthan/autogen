"""Predator agent deliberating with Alpha-Beta adversarial lookahead (SYS-10, Module M07)."""

from __future__ import annotations

from typing import Any

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.adversarial.evaluations import (
    PredatorPreyEvaluator,
    PredatorPreyState,
    order_predator_moves,
)
from autogen.intelligence.adversarial.minimax import AlphaBetaSearch
from autogen.intelligence.adversarial.predator_prey_game import PredatorPreySubGame
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position, TERRAIN_PROPERTIES


class PredatorAgent(BaseAgent):
    """Adversarial predator agent anticipating prey evasion through depth-limited Alpha-Beta search."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities | None = None,
        max_lookahead_depth: int = 3,
        target_prey_id: str | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(planning=True))
        self.max_lookahead_depth = max_lookahead_depth
        self.target_prey_id = target_prey_id
        self.evaluator = PredatorPreyEvaluator()
        self.search_engine = AlphaBetaSearch[PredatorPreyState, ActionType](
            move_ordering_fn=order_predator_moves,
            record_tree=True,
        )

        self.last_known_prey_pos: Position | None = None
        self.blocked_cells: set[Position] = set()
        self.latest_trace: DecisionTrace | None = None
        self.last_search_tree: Any | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

        # Update known impassable obstacles from observation
        for cell in observation.visible_cells:
            if not TERRAIN_PROPERTIES[cell.terrain].walkable:
                self.blocked_cells.add(cell.position)

        # Detect visible prey
        for other_id, other_pos in observation.visible_agents:
            if self.target_prey_id is None or other_id == self.target_prey_id:
                self.last_known_prey_pos = other_pos
                break

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            trace = (
                DecisionTraceBuilder(self.agent_id, self._step_counter, "AlphaBetaSearch")
                .set_selected_action(ActionType.STAY, "No observation available")
                .build()
            )
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        my_pos = self._last_observation.position
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "AlphaBetaSearch")
        trace_builder.set_observation_summary(self._last_observation)

        # If prey position is known, formulate adversarial subgame
        if self.last_known_prey_pos is not None:
            subgame_state = PredatorPreyState(
                predator_pos=my_pos,
                prey_pos=self.last_known_prey_pos,
                predator_energy=self._last_observation.energy,
                prey_energy=100.0,
                turn=0,  # Predator is MAX
                blocked_cells=frozenset(self.blocked_cells),
            )
            subgame = PredatorPreySubGame(subgame_state)
            search_res = self.search_engine.search(
                subgame,
                subgame_state,
                max_depth=self.max_lookahead_depth,
                eval_fn=self.evaluator,
            )
            self.last_search_tree = search_res.tree_root

            best_act = search_res.best_action or ActionType.STAY

            # Populate candidate action scores from tree root
            if search_res.tree_root and search_res.tree_root.children:
                for child in search_res.tree_root.children:
                    if child.best_action is not None:
                        trace_builder.add_candidate_action(child.best_action, child.value)
            else:
                trace_builder.add_candidate_action(best_act, search_res.expected_value)

            trace_builder.set_selected_action(
                best_act,
                f"AlphaBeta lookahead depth={self.max_lookahead_depth} expected_val={search_res.expected_value:.2f}",
            )
            trace_builder.add_metric("nodes_expanded", search_res.nodes_expanded)
            trace_builder.add_metric("nodes_pruned", search_res.nodes_pruned)
            trace_builder.add_metric("expected_value", search_res.expected_value)
            trace_builder.add_metric("runtime_seconds", search_res.runtime_seconds)

            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=best_act), trace

        # Default exploration when prey is not visible
        legal_deltas = [
            (ActionType.UP, 0, -1),
            (ActionType.DOWN, 0, 1),
            (ActionType.LEFT, -1, 0),
            (ActionType.RIGHT, 1, 0),
        ]
        chosen_action = ActionType.STAY
        for act, dx, dy in legal_deltas:
            cand = Position(x=my_pos.x + dx, y=my_pos.y + dy)
            if cand not in self.blocked_cells:
                chosen_action = act
                break

        trace_builder.add_candidate_action(chosen_action, 0.0)
        trace_builder.set_selected_action(chosen_action, "Exploratory movement: no prey detected")
        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.last_known_prey_pos = None
        self.blocked_cells.clear()
        self.latest_trace = None
        self.last_search_tree = None
