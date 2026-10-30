"""Deterministic simulation execution engine implementing the canonical 9-step tick lifecycle."""

from __future__ import annotations

import copy

from autogen.core.rng import SeedManager
from autogen.core.types import DecisionTrace
from autogen.simulation.core.conflict_resolution import ConflictResolver
from autogen.simulation.core.contracts import (
    Action,
    AgentProtocol,
    BaseEnvironment,
    ConflictPolicy,
    DecisionTraceEmittedEvent,
    InvalidActionEvent,
    Observation,
    StepCompletedEvent,
)
from autogen.simulation.core.events import EventBus
from autogen.simulation.core.replay import ReplayLog, ReplayStep
from autogen.simulation.core.snapshot import SimulationSnapshot
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.state.types import ActionType
from autogen.simulation.state.world_state import WorldState, canonical_world_hash


class _ActionOverrideAgent:
    """Wraps an agent to override its action for a single step during forking."""

    def __init__(self, original_agent: AgentProtocol, override_action: Action) -> None:
        self.agent_id = original_agent.agent_id
        self._orig = original_agent
        self._override = override_action
        self._used = False
        self._last_step = 0

    def observe(self, observation: Observation) -> None:
        self._last_step = observation.step
        self._orig.observe(observation)

    def update_belief(self) -> None:
        self._orig.update_belief()

    def decide(self) -> tuple[Action, DecisionTrace]:
        if not self._used:
            self._used = True
            trace = DecisionTrace(
                agent_id=self.agent_id,
                step_id=self._last_step,
                algorithm_name="fork_override",
                decision_type="action_override",
                rationale={"override_action": self._override.action_type.value},
            )
            return self._override, trace
        return self._orig.decide()

    def learn(self, reward: float, terminal: bool) -> None:
        self._orig.learn(reward, terminal)

    def reset(self) -> None:
        self._orig.reset()


class SimulationEngine:
    """Controls simulation execution, determinism, time-machine branching, and event dispatch.

    Execution order per tick is strictly maintained according to ADR-005:
    1. Generate observations
    2. Update beliefs / memory
    3. Request agent decisions
    4. Validate actions (substitute STAY on invalid actions)
    5. Resolve simultaneous conflicts
    6. Apply environment transition
    7. Compute rewards / costs and call agent.learn()
    8. Update metrics, publish StepCompletedEvent, take snapshot if scheduled
    9. Publish new world state
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        config: EnvironmentConfig | None = None,
        agents: dict[str, AgentProtocol] | None = None,
        seed: int = 42,
        snapshot_cadence: int = 50,
        conflict_policy: ConflictPolicy = ConflictPolicy.CANCEL_ALL,
    ) -> None:
        self.config = config or EnvironmentConfig(seed=seed)
        self.environment = environment
        self.agents: dict[str, AgentProtocol] = dict(agents) if agents else {}
        self.seed = seed
        self.seed_manager = SeedManager(master_seed=seed)
        self.conflict_rng = self.seed_manager.spawn_stream("conflict_arbitration")
        self.conflict_resolver = ConflictResolver(conflict_policy=conflict_policy)

        self.event_bus = EventBus()
        self.snapshot_cadence = snapshot_cadence
        self.snapshots: dict[int, SimulationSnapshot] = {}
        self.action_history: list[dict[str, Action]] = []
        self.replay_steps: list[ReplayStep] = []

    def register_agent(self, agent: AgentProtocol) -> None:
        """Registers an agent into the simulation engine."""
        self.agents[agent.agent_id] = agent

    def step(self) -> WorldState:
        """Executes a single discrete time tick according to the 9-phase lifecycle."""
        current_state = self.environment.state
        active_agents = {
            aid: agent
            for aid, agent in self.agents.items()
            if aid in current_state.agents and current_state.agents[aid].is_alive
        }

        # Phase 1: Generate observations
        for aid, agent in active_agents.items():
            obs = self.environment.get_observation(aid)
            agent.observe(obs)

        # Phase 2: Update beliefs and memory
        for agent in active_agents.values():
            agent.update_belief()

        # Phase 3: Request agent decisions
        proposed_actions: dict[str, Action] = {}
        for aid, agent in active_agents.items():
            action, trace = agent.decide()
            proposed_actions[aid] = action
            self.event_bus.publish(
                DecisionTraceEmittedEvent(
                    step=current_state.step,
                    agent_id=aid,
                    algorithm_name=trace.algorithm_name,
                    decision_type=trace.decision_type,
                    metrics=trace.metrics,
                )
            )

        # Phase 4: Validate actions (replace invalid actions with STAY)
        validated_actions: dict[str, Action] = {}
        for aid, action in proposed_actions.items():
            if self.environment.validate_action(aid, action.action_type):
                validated_actions[aid] = action
            else:
                substituted = Action(
                    agent_id=aid,
                    action_type=ActionType.STAY,
                    parameters={"reason": "validation_failed"},
                )
                validated_actions[aid] = substituted
                self.event_bus.publish(
                    InvalidActionEvent(
                        step=current_state.step,
                        agent_id=aid,
                        attempted_action=action,
                        substituted_action=substituted,
                        reason="Action geometry or walkability validation failed",
                    )
                )

        # Phase 5: Resolve simultaneous conflicts
        arbitrated_actions = self.conflict_resolver.resolve_conflicts(
            current_state=current_state,
            actions=validated_actions,
            rng=self.conflict_rng,
        )

        # Phase 6: Apply environment transitions
        action_types_dict = {aid: act.action_type for aid, act in arbitrated_actions.items()}
        next_world_state = self.environment.apply_transitions(action_types_dict)

        # Phase 7: Compute rewards and notify agent learning
        for aid, agent in active_agents.items():
            was_alive = current_state.agents[aid].is_alive
            is_now_alive = aid in next_world_state.agents and next_world_state.agents[aid].is_alive
            # Default biological reward: energy delta + survival bonus
            prev_energy = current_state.agents[aid].energy
            curr_energy = next_world_state.agents[aid].energy if is_now_alive else 0.0
            reward = (curr_energy - prev_energy) + (1.0 if is_now_alive else -50.0)
            agent.learn(reward=reward, terminal=(was_alive and not is_now_alive))

        # Phase 8: Update metrics, record replay, emit StepCompletedEvent
        state_hash = canonical_world_hash(next_world_state)
        self.action_history.append(arbitrated_actions)
        self.replay_steps.append(
            ReplayStep(
                step=next_world_state.step,
                actions=action_types_dict,
                state_hash=state_hash,
            )
        )

        self.event_bus.publish(
            StepCompletedEvent(
                step=next_world_state.step,
                state_hash=state_hash,
                active_agents_count=len(
                    [a for a in next_world_state.agents.values() if a.is_alive]
                ),
            )
        )

        if self.snapshot_cadence > 0 and next_world_state.step % self.snapshot_cadence == 0:
            self.snapshots[next_world_state.step] = self.snapshot()

        # Phase 9: Publish and return new world state
        return next_world_state

    def run(self, max_ticks: int) -> WorldState:
        """Runs the simulation engine for max_ticks or until all agents are deceased."""
        state = self.environment.state
        for _ in range(max_ticks):
            # Check if all agents are deceased
            if self.agents and all(not a.is_alive for a in self.environment.state.agents.values()):
                break
            state = self.step()
        return state

    def snapshot(self) -> SimulationSnapshot:
        """Captures an exact, immutable snapshot of the current state and action history."""
        current_state = self.environment.state
        return SimulationSnapshot.create(
            step=current_state.step,
            world_state=current_state,
            agents=self.agents,
            conflict_rng_state=dict(self.conflict_rng.bit_generator.state),
            action_history=self.action_history,
        )

    def restore(self, snapshot: SimulationSnapshot) -> None:
        """Restores simulation engine exactly to the state represented by snapshot."""
        # Restore environment state
        self.environment.restore_state(snapshot.world_state)

        # Restore agents state
        self.agents = copy.deepcopy(snapshot.agents_backup)

        # Restore RNG state
        if snapshot.conflict_rng_state:
            self.conflict_rng.bit_generator.state = copy.deepcopy(snapshot.conflict_rng_state)

        self.action_history = []
        for _step_idx, step_actions in snapshot.recorded_actions:
            self.action_history.append({act.agent_id: act for act in step_actions})

        self.replay_steps = [s for s in self.replay_steps if s.step <= snapshot.step]

    def fork(
        self,
        snapshot: SimulationSnapshot | None = None,
        overridden_actions: dict[str, Action] | None = None,
    ) -> SimulationEngine:
        """Forks a new independent simulation run from a snapshot with optional action overrides."""
        target_snapshot = snapshot if snapshot is not None else self.snapshot()

        # Create cloned environment from the snapshot's state
        cloned_env = copy.deepcopy(self.environment)
        cloned_env.restore_state(target_snapshot.world_state)

        cloned_agents = {aid: copy.deepcopy(agent) for aid, agent in self.agents.items()}

        forked_engine = SimulationEngine(
            environment=cloned_env,
            config=self.config,
            agents=cloned_agents,
            seed=self.seed,
            snapshot_cadence=self.snapshot_cadence,
            conflict_policy=self.conflict_resolver.conflict_policy,
        )
        forked_engine.restore(target_snapshot)

        # If overridden actions are provided, wrap target agents with _ActionOverrideAgent
        if overridden_actions is not None:
            for aid, override_action in overridden_actions.items():
                if aid in forked_engine.agents:
                    forked_engine.agents[aid] = _ActionOverrideAgent(
                        forked_engine.agents[aid], override_action
                    )
            forked_engine.step()

        return forked_engine

    def get_replay_log(self) -> ReplayLog:
        """Produces the complete ReplayLog for this run."""
        return ReplayLog(
            engine_seed=self.seed,
            config=self.config,
            steps=tuple(self.replay_steps),
        )
