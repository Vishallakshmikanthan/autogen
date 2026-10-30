"""Autonomous agent deliberating with deductive reasoning and verifiable safety proofs (SYS-07, Module M08)."""

from __future__ import annotations

from typing import Any

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.symbolic.safety_layer import SafetyRuleLayer
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.logic.provenance import ProofTree
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, DIRECTION_DISPLACEMENTS, Position


class LogicAgent(BaseAgent):
    """Symbolic autonomous agent filtering candidate actions against deductive safety proofs."""

    def __init__(
        self,
        agent_id: str,
        capabilities: AgentCapabilities | None = None,
        energy_safety_threshold: float = 25.0,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities(belief=True, planning=True))
        self.safety_layer = SafetyRuleLayer(energy_threshold=energy_safety_threshold)
        self.latest_trace: DecisionTrace | None = None
        self.latest_proof: ProofTree | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step
        # Ground sensory observation into symbolic logic facts
        self.safety_layer.ground_perceptions(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        if self._last_observation is None:
            trace = (
                DecisionTraceBuilder(self.agent_id, self._step_counter, "LogicDeduction")
                .set_selected_action(ActionType.STAY, "No observation available")
                .build()
            )
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        my_pos = self._last_observation.position
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "LogicDeduction")
        trace_builder.set_observation_summary(self._last_observation)

        candidate_actions = [
            ActionType.UP,
            ActionType.DOWN,
            ActionType.LEFT,
            ActionType.RIGHT,
            ActionType.STAY,
        ]

        verified_safe_actions: list[ActionType] = []
        filtered_proofs: dict[str, str] = {}
        last_proof: ProofTree | None = None

        for act in candidate_actions:
            if act == ActionType.STAY:
                verified_safe_actions.append(act)
                trace_builder.add_candidate_action(act, 0.5)
                continue

            unsafe, proof = self.safety_layer.is_action_provably_unsafe(my_pos, act)
            if unsafe and proof is not None:
                filtered_proofs[act.value] = f"Provably unsafe via proof root step [{proof.root_step_id}]"
                last_proof = proof
                trace_builder.add_candidate_action(act, -1.0)
            else:
                verified_safe_actions.append(act)
                trace_builder.add_candidate_action(act, 1.0)

        # Select highest-scoring verified action (first non-STAY safe action if available)
        chosen_action = ActionType.STAY
        for act in [ActionType.UP, ActionType.DOWN, ActionType.RIGHT, ActionType.LEFT, ActionType.STAY]:
            if act in verified_safe_actions:
                chosen_action = act
                break

        self.latest_proof = last_proof
        proof_count = len(last_proof.steps) if last_proof else 0

        trace_builder.add_metric("provenance_steps_count", proof_count)
        trace_builder.add_metric("safe_actions_count", len(verified_safe_actions))
        trace_builder.set_selected_action(
            chosen_action,
            f"Deductively verified action '{chosen_action.value}' (Filtered {len(filtered_proofs)} hazards)",
        )

        trace = trace_builder.build()
        # Enrich rationale with symbolic justification
        if last_proof:
            trace.rationale["symbolic_justification"] = {
                "rejected_actions": filtered_proofs,
                "proof_summary": last_proof.pretty_print()[:256],
            }

        self.latest_trace = trace
        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self.latest_trace = None
        self.latest_proof = None
