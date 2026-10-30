"""Prey agent evading predators using Minimax or Expectimax lookahead (SYS-10, Module M07)."""

from __future__ import annotations

from typing import Any

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.adversarial.contracts import AdversarialAlgorithm
from autogen.intelligence.adversarial.evaluations import (
    PredatorPreyEvaluator,
    PredatorPreyState,
)
from autogen.intelligence.adversarial.minimax import ExpectimaxSearch, MinimaxSearch
from autogen.intelligence.adversarial.predator_prey_game import PredatorPreySubGame
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position, TERRAIN_PROPERTIES


class PreyAgent(BaseAgent):
    """Prey agent utilizing Minimax or Expectimax adversarial lookahead to maximize evasion."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities | None = None,
        max_lookahead_depth: int = 3,
        use_expectimax: bool = False,
        predator_id: str | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(planning=True))
        self.max_lookahead_depth = max_lookahead_depth
        self.use_expectimax = use_expectimax
        self.predator_id = predator_id
        self.evaluator = PredatorPreyEvaluator()

        self.search_engine: AdversarialAlgorithm[PredatorPreyState, ActionType]
        if use_expectimax:
            self.search_engine = ExpectimaxSearch[PredatorPreyState, ActionType](record_tree=True)
            self._algo_name = "ExpectimaxSearch"
        else:
            self.search_engine = MinimaxSearch[PredatorPreyState, ActionType](record_tree=True)
            self._algo_name = "MinimaxSearch"

        self.last_known_predator_pos: Position | None = None
        self.blocked_cells: set[Position] = set()
        self.latest_trace: DecisionTrace | None = None
        self.last_search_tree: Any | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

        for cell in observation.visible_cells:
            if not TERRAIN_PROPERTIES[cell.terrain].walkable:
                self.blocked_cells.add(cell.position)

        for other_id, other_pos in observation.visible_agents:
            if self.predator_id is None or other_id == self.predator_id:
                self.last_known_predator_pos = other_pos
                break

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            trace = (
                DecisionTraceBuilder(self.agent_id, self._step_counter, self._algo_name)
                .set_selected_action(ActionType.STAY, "No observation available")
                .build()
            )
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        my_pos = self._last_observation.position
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, self._algo_name)
        trace_builder.set_observation_summary(self._last_observation)

        if self.last_known_predator_pos is not None:
            # Formulate subgame with Prey turn (turn=1)
            subgame_state = PredatorPreyState(
                predator_pos=self.last_known_predator_pos,
                prey_pos=my_pos,
                predator_energy=100.0,
                prey_energy=self._last_observation.energy,
                turn=1,  # Prey's turn
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

            if search_res.tree_root and search_res.tree_root.children:
                for child in search_res.tree_root.children:
                    if child.best_action is not None:
                        # In subgame, evaluator is defined from MAX (predator) perspective,
                        # so prey's utility is inverted (-child.value)
                        trace_builder.add_candidate_action(child.best_action, -child.value)
            else:
                trace_builder.add_candidate_action(best_act, -search_res.expected_value)

            trace_builder.set_selected_action(
                best_act,
                f"{self._algo_name} evasion depth={self.max_lookahead_depth} expected_val={-search_res.expected_value:.2f}",
            )
            trace_builder.add_metric("nodes_expanded", search_res.nodes_expanded)
            trace_builder.add_metric("expected_value", -search_res.expected_value)
            trace_builder.add_metric("runtime_seconds", search_res.runtime_seconds)

            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=best_act), trace

        # Fallback forage / wander if predator is not seen
        # Collect food if on cell, otherwise move
        chosen_action = ActionType.STAY
        deltas = [
            (ActionType.RIGHT, 1, 0),
            (ActionType.UP, 0, -1),
            (ActionType.LEFT, -1, 0),
            (ActionType.DOWN, 0, 1),
        ]
        for act, dx, dy in deltas:
            cand = Position(x=my_pos.x + dx, y=my_pos.y + dy)
            if cand not in self.blocked_cells:
                chosen_action = act
                break

        trace_builder.add_candidate_action(chosen_action, 0.0)
        trace_builder.set_selected_action(chosen_action, "Exploratory evasion: no predator visible")
        trace = trace_builder.build()
        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.last_known_predator_pos = None
        self.blocked_cells.clear()
        self.latest_trace = None
        self.last_search_tree = None
