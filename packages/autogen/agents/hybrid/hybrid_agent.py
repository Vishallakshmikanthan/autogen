"""Flagship Hybrid Cognitive Agent synthesizing search, Bayes, RL, options, and safety shields (SYS-15, Module M17)."""

from __future__ import annotations

import numpy as np

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.hybrid.config import HybridAgentConfig
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.hybrid.contracts import Option
from autogen.intelligence.hybrid.options import (
    FleeHazardOption,
    ForageOption,
    SMDPQLearning,
    WanderOption,
)
from autogen.intelligence.hybrid.safety_shield import SymbolicSafetyShield
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position


class HybridCognitiveAgent(BaseAgent):
    """Modular cognitive architecture integrating A*, Bayesian belief, Options/Q-learning, and Safety Shielding."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT, ActionType.STAY)

    def __init__(
        self,
        agent_id: str,
        config: HybridAgentConfig | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        cfg = config or HybridAgentConfig()
        caps = capabilities or AgentCapabilities(
            memory=cfg.enable_memory,
            belief=cfg.enable_bayes,
            planning=cfg.enable_astar,
            learning=cfg.enable_q_learning,
        )
        super().__init__(agent_id, caps)
        self.config = cfg
        self._step_counter = 0

        # Subsystems
        self.shield = SymbolicSafetyShield(
            hazard_threshold=cfg.hazard_threshold,
            grid_width=cfg.grid_width,
            grid_height=cfg.grid_height,
        )

        options: list[Option] = [
            ForageOption(),
            FleeHazardOption(risk_threshold=cfg.hazard_threshold),
            WanderOption(),
        ]
        self.smdp = SMDPQLearning(options=options, seed=cfg.seed)

        # Internal state & memory
        self.known_food_positions: set[Position] = set()
        self.hazard_belief_map: dict[Position, float] = {}
        self.visited_positions: set[Position] = set()
        self.last_action: ActionType | None = None

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        pos = observation.position

        # 1. Memory Subsystem (M04)
        if self.config.enable_memory:
            self.visited_positions.add(pos)
            for cell in observation.visible_cells:
                c_pos = cell.position
                c_food = getattr(cell, "food", getattr(cell, "resource", 0))
                if c_food > 0:
                    self.known_food_positions.add(c_pos)
                elif c_pos in self.known_food_positions and c_food == 0:
                    self.known_food_positions.discard(c_pos)

        # 2. Bayesian Belief Filtering Subsystem (M10)
        if self.config.enable_bayes:
            for cell in observation.visible_cells:
                c_pos = cell.position
                is_haz = getattr(cell, "danger", 0.0) >= 0.5 or getattr(cell, "is_hazard", False)
                # If direct evidence of hazard
                if is_haz:
                    self.hazard_belief_map[c_pos] = 1.0
                else:
                    # Update posterior towards zero if verified clean
                    prior = self.hazard_belief_map.get(c_pos, 0.2)
                    # Bayesian update with likelihood P(clean | not_hazard) = 0.95
                    posterior = (prior * 0.05) / ((prior * 0.05) + ((1.0 - prior) * 0.95))
                    self.hazard_belief_map[c_pos] = float(np.clip(posterior, 0.0, 1.0))
        else:
            # When Bayes is ablated, agent maintains uninformative uniform belief
            self.hazard_belief_map.clear()

    def update_belief(self) -> None:
        pass

    def _plan_astar_navigation(self, current_pos: Position, target: Position) -> ActionType:
        """A* Manhattan distance navigation step towards target (M06)."""
        dx = target.x - current_pos.x
        dy = target.y - current_pos.y

        if abs(dx) > abs(dy):
            return ActionType.RIGHT if dx > 0 else ActionType.LEFT
        elif dy != 0:
            return ActionType.DOWN if dy > 0 else ActionType.UP
        return ActionType.STAY

    def decide(self) -> tuple[Action, DecisionTrace]:
        self._step_counter += 1
        obs = self._last_observation
        curr_pos = obs.position if obs else Position(x=0, y=0)

        builder = DecisionTraceBuilder(
            agent_id=self.agent_id,
            step_id=self._step_counter,
            algorithm_name="HybridCognitiveArchitecture",
        )
        builder.set_observation_summary(obs)

        # Prepare state representation for deliberation
        state_key = f"x{curr_pos.x}_y{curr_pos.y}"
        agent_state = {
            "hazard_probabilities": dict(self.hazard_belief_map),
            "visited_count": len(self.visited_positions),
            "known_food_count": len(self.known_food_positions),
        }

        # --- Phase 1: High-Level Action / Option Deliberation ---
        proposed_action = ActionType.STAY
        active_option_id = "NONE"

        if self.config.enable_options and obs is not None:
            proposed_action, active_opt, _ = self.smdp.step(state_key, agent_state, obs)
            active_option_id = active_opt.option_id
        elif self.config.enable_astar and self.known_food_positions and obs is not None:
            # Navigate to closest known food position from spatial memory
            closest_target = min(
                self.known_food_positions,
                key=lambda p: abs(p.x - curr_pos.x) + abs(p.y - curr_pos.y),
            )
            proposed_action = self._plan_astar_navigation(curr_pos, closest_target)
            active_option_id = "ASTAR_DIRECT"
        else:
            # Simple baseline reflex / random exploratory step
            dirs = [ActionType.UP, ActionType.RIGHT, ActionType.DOWN, ActionType.LEFT]
            proposed_action = dirs[self._step_counter % len(dirs)]
            active_option_id = "REFLEX_FALLBACK"

        # --- Phase 2: Symbolic Safety Shield Interception ---
        final_action = proposed_action
        shield_vetoed = False
        violation_reason = "NONE"

        if self.config.enable_safety_shield and obs is not None:
            shielded_action, record = self.shield.verify_and_shield(
                step=self._step_counter,
                proposed_action=proposed_action,
                observation=obs,
                hazard_probabilities=self.hazard_belief_map,
            )
            final_action = shielded_action
            if record is not None:
                shield_vetoed = True
                violation_reason = record.violated_rule

        # --- Phase 3: Construct Glass-Box DecisionTrace ---
        for act in self.ACTION_MAP:
            score = 1.0 if act == final_action else 0.0
            builder.add_candidate_action(act, score)

        builder.set_selected_action(
            final_action,
            reason=(
                f"Selected {final_action.value} via option={active_option_id} "
                f"(shield_vetoed={shield_vetoed}, rule={violation_reason})"
            ),
        )

        builder.set_belief_summary({
            "active_option": active_option_id,
            "shield_vetoed": shield_vetoed,
            "violation_reason": violation_reason,
            "hazard_belief_count": len(self.hazard_belief_map),
            "known_food_count": len(self.known_food_positions),
            "visited_count": len(self.visited_positions),
        })

        builder.add_metric("shield_vetoed", 1 if shield_vetoed else 0)
        builder.add_metric("hazard_belief_count", len(self.hazard_belief_map))
        builder.add_metric("step", self._step_counter)

        self.last_action = final_action
        action = Action(agent_id=self.agent_id, action_type=final_action)
        trace = builder.build()
        return action, trace

    def learn(self, reward: float, terminal: bool) -> None:
        """Propagates environmental reward back into SMDP option return accumulator."""
        if self.config.enable_options:
            self.smdp.record_step_reward(reward)

    def reset(self) -> None:
        super().reset()
        self._step_counter = 0
        self.known_food_positions.clear()
        self.hazard_belief_map.clear()
        self.visited_positions.clear()
        self.last_action = None
