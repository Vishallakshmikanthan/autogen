"""Cooperative agent with stigmergic sensing, message broadcasting, and contract bidding (SYS-14, Module M15)."""

from __future__ import annotations

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.trace_builder import DecisionTraceBuilder
from autogen.core.types import DecisionTrace
from autogen.intelligence.multiagent.communication import CommunicationBus
from autogen.intelligence.multiagent.stigmergy import PheromoneField
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class CooperativeAgent(BaseAgent):
    """Cooperative collective agent integrating stigmergic chemical gradient following and communication."""

    ACTION_MAP = (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT)

    def __init__(
        self,
        agent_id: str,
        pheromone_field: PheromoneField | None = None,
        comm_bus: CommunicationBus | None = None,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        super().__init__(agent_id, capabilities or AgentCapabilities())
        self.pheromone_field = pheromone_field
        self.comm_bus = comm_bus

        self.latest_trace: DecisionTrace | None = None
        self._step_counter = 0

    def observe(self, observation: Observation) -> None:
        super().observe(observation)
        self._step_counter = observation.step

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        trace_builder = DecisionTraceBuilder(self.agent_id, self._step_counter, "CooperativeCollective")
        trace_builder.set_observation_summary(self._last_observation)

        if self._last_observation is None:
            trace_builder.set_selected_action(ActionType.STAY, "No observation available")
            trace = trace_builder.build()
            self.latest_trace = trace
            return Action(agent_id=self.agent_id, action_type=ActionType.STAY), trace

        current_pos = self._last_observation.position

        # 1. Sense local pheromone field gradient
        grad_x, grad_y = 0.0, 0.0
        chem_conc = 0.0
        if self.pheromone_field is not None:
            grad_x, grad_y = self.pheromone_field.gradient(current_pos)
            chem_conc = self.pheromone_field.sense(current_pos)

        trace_builder.add_metric("pheromone_conc", chem_conc)
        trace_builder.add_metric("grad_x", grad_x)
        trace_builder.add_metric("grad_y", grad_y)

        # 2. Receive communication messages
        inbox = []
        if self.comm_bus is not None:
            inbox = self.comm_bus.receive(self.agent_id, current_pos)
        trace_builder.add_metric("inbox_count", len(inbox))

        # 3. Determine action based on gradient uphill or exploratory drift
        if abs(grad_x) > abs(grad_y) and abs(grad_x) > 0.01:
            chosen_action = ActionType.RIGHT if grad_x > 0 else ActionType.LEFT
        elif abs(grad_y) > 0.01:
            chosen_action = ActionType.DOWN if grad_y > 0 else ActionType.UP
        else:
            # Neutral / default drift
            chosen_action = ActionType.UP

        for act in self.ACTION_MAP:
            score = 1.0 if act == chosen_action else 0.0
            trace_builder.add_candidate_action(act, score)

        reason = f"Stigmergic gradient pursuit {chosen_action.value} (conc={chem_conc:.2f}, grad=({grad_x:.2f}, {grad_y:.2f}))"
        trace_builder.set_selected_action(chosen_action, reason)
        trace = trace_builder.build()
        self.latest_trace = trace

        # 4. Deposit pheromones or send messages if resources are found
        if self._last_observation.carried_food > 0.0 and self.pheromone_field is not None:
            self.pheromone_field.deposit(current_pos, amount=15.0)

        return Action(agent_id=self.agent_id, action_type=chosen_action), trace

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self._step_counter = 0
