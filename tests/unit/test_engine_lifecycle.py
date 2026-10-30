"""Unit tests for SimulationEngine lifecycle, event dispatch, and invalid action handling."""

from __future__ import annotations

import pytest

from autogen.agents.base.dummy_agents import ScriptedAgent
from autogen.simulation.core.contracts import (
    DecisionTraceEmittedEvent,
    InvalidActionEvent,
    StepCompletedEvent,
)
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import ActionType, Position


@pytest.mark.unit
def test_engine_substitutes_stay_on_invalid_action():
    """Environment substitutes STAY when an agent proposes an out-of-bounds or illegal move."""
    world = GridWorld(EnvironmentConfig(width=6, height=6, seed=42))
    world.reset()
    world.spawn_agent(
        "agent_walker", position=Position(0, 0) if world.is_walkable(Position(0, 0)) else None
    )

    engine = SimulationEngine(environment=world, seed=123)

    invalid_events: list[InvalidActionEvent] = []
    engine.event_bus.subscribe(InvalidActionEvent, invalid_events.append)

    # Script an illegal move (e.g. moving into boundary or obstacle)
    agent_pos = world.state.agents["agent_walker"].position
    illegal_action = ActionType.UP if agent_pos.y == 0 else ActionType.LEFT
    scripted_agent = ScriptedAgent("agent_walker", script=[illegal_action])
    engine.register_agent(scripted_agent)

    # Step the engine
    next_state = engine.step()

    # Agent should remain at the same position
    assert next_state.agents["agent_walker"].position == agent_pos
    # Invalid action event should have been emitted
    assert len(invalid_events) == 1
    assert invalid_events[0].agent_id == "agent_walker"
    assert invalid_events[0].substituted_action.action_type == ActionType.STAY


@pytest.mark.unit
def test_engine_event_bus_dispatches_events():
    """Validates that step completion and decision trace events are published synchronously."""
    world = GridWorld(EnvironmentConfig(width=8, height=8, seed=10))
    world.reset()
    world.spawn_agent("agent_0")

    engine = SimulationEngine(environment=world, seed=10)
    agent = ScriptedAgent("agent_0", script=[ActionType.STAY])
    engine.register_agent(agent)

    step_events: list[StepCompletedEvent] = []
    trace_events: list[DecisionTraceEmittedEvent] = []

    engine.event_bus.subscribe(StepCompletedEvent, step_events.append)
    engine.event_bus.subscribe(DecisionTraceEmittedEvent, trace_events.append)

    for _ in range(5):
        engine.step()

    assert len(step_events) == 5
    assert len(trace_events) == 5
    assert step_events[-1].step == 5
    assert trace_events[-1].agent_id == "agent_0"
