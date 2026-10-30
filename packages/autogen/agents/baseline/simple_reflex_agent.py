"""Condition-action SimpleReflexAgent."""

from __future__ import annotations

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.environment.generator import TERRAIN_PROPERTIES
from autogen.simulation.state.types import ActionType

DIRECTION_DELTAS: dict[ActionType, tuple[int, int]] = {
    ActionType.UP: (0, -1),
    ActionType.DOWN: (0, 1),
    ActionType.LEFT: (-1, 0),
    ActionType.RIGHT: (1, 0),
}


class SimpleReflexAgent(BaseAgent):
    """Reflex agent mapping observations directly to actions via production rules: $a_t = \\pi(o_t)$."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities())
        self._heading = ActionType.RIGHT
        self._step_count = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            raise RuntimeError("Agent must observe before deciding.")

        obs = self._last_observation
        self._step_count += 1
        pos = obs.position

        # Index visible cells for fast geometric query
        cell_lookup = {(c.position.x, c.position.y): c for c in obs.visible_cells}
        current_cell = cell_lookup.get((pos.x, pos.y))

        builder = (
            DecisionTraceBuilder(self.agent_id, obs.step, "condition_action_rules")
            .set_mechanism("production_rules")
            .set_observation_summary(obs)
            .add_metric("step_count", self._step_count)
            .add_metric("heading", self._heading.value)
        )

        all_actions = [
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
            ActionType.STAY,
            ActionType.COLLECT,
        ]
        scores: dict[ActionType, float] = dict.fromkeys(all_actions, 0.0)

        selected_action: ActionType
        decision_reason: str

        # RULE 1: Harvest food present at feet
        if current_cell is not None and current_cell.food > 0.0:
            scores[ActionType.COLLECT] = 100.0
            selected_action = ActionType.COLLECT
            decision_reason = (
                f"Food available at current position ({current_cell.food:.1f}); harvesting."
            )
        else:
            # Check for visible food within observation window
            visible_food = [
                c
                for c in obs.visible_cells
                if c.food > 0.0 and (c.position.x, c.position.y) != (pos.x, pos.y)
            ]

            if visible_food:
                # RULE 2: Navigate toward nearest visible food
                visible_food.sort(
                    key=lambda c: abs(c.position.x - pos.x) + abs(c.position.y - pos.y)
                )
                target = visible_food[0].position
                target_dist = abs(target.x - pos.x) + abs(target.y - pos.y)

                best_dir: ActionType | None = None
                best_score = -1.0

                for act, (dx, dy) in DIRECTION_DELTAS.items():
                    nxt = (pos.x + dx, pos.y + dy)
                    nxt_cell = cell_lookup.get(nxt)
                    if nxt_cell and TERRAIN_PROPERTIES[nxt_cell.terrain].walkable:
                        nxt_dist = abs(target.x - nxt[0]) + abs(target.y - nxt[1])
                        # Shorter distance yields higher score
                        dir_score = 50.0 + (target_dist - nxt_dist) * 10.0
                        scores[act] = dir_score
                        if dir_score > best_score:
                            best_score = dir_score
                            best_dir = act
                    else:
                        scores[act] = 0.0

                if best_dir is not None:
                    selected_action = best_dir
                    self._heading = best_dir
                    decision_reason = (
                        f"Navigating toward visible food at {target} (dist={target_dist})."
                    )
                else:
                    selected_action = ActionType.STAY
                    decision_reason = "Food visible but all approach vectors blocked."
            else:
                # RULE 3 & 4: Obstacle evasion and forward exploration
                h_dx, h_dy = DIRECTION_DELTAS[self._heading]
                forward_pos = (pos.x + h_dx, pos.y + h_dy)
                forward_cell = cell_lookup.get(forward_pos)

                if forward_cell and TERRAIN_PROPERTIES[forward_cell.terrain].walkable:
                    scores[self._heading] = 20.0
                    selected_action = self._heading
                    decision_reason = (
                        f"Forward path clear; advancing along heading {self._heading.value}."
                    )
                else:
                    # Obstacle ahead -> turn to alternate walkable direction
                    turn_order = [
                        ActionType.UP,
                        ActionType.RIGHT,
                        ActionType.DOWN,
                        ActionType.LEFT,
                    ]
                    # Start search after current heading
                    start_idx = (turn_order.index(self._heading) + 1) % len(turn_order)
                    candidates = turn_order[start_idx:] + turn_order[:start_idx]

                    best_turn: ActionType | None = None
                    for act in candidates:
                        tdx, tdy = DIRECTION_DELTAS[act]
                        turn_pos = (pos.x + tdx, pos.y + tdy)
                        t_cell = cell_lookup.get(turn_pos)
                        if t_cell and TERRAIN_PROPERTIES[t_cell.terrain].walkable:
                            scores[act] = 15.0
                            if best_turn is None:
                                best_turn = act
                        else:
                            scores[act] = 0.0

                    if best_turn is not None:
                        selected_action = best_turn
                        self._heading = best_turn
                        decision_reason = (
                            f"Obstacle ahead at {forward_pos}; turning to {best_turn.value}."
                        )
                    else:
                        selected_action = ActionType.STAY
                        decision_reason = "Surrounded by non-walkable terrain; staying in place."

        for act in all_actions:
            builder.add_candidate_action(act, scores[act])

        builder.set_selected_action(selected_action, decision_reason)
        trace = builder.build()
        action = Action(agent_id=self.agent_id, action_type=selected_action)
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self._heading = ActionType.RIGHT
        self._step_count = 0
