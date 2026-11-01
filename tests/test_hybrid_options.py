"""Tests for Hierarchical Options Framework and SMDP Q-learning (SYS-15, Module M17)."""

from __future__ import annotations

from autogen.intelligence.hybrid.options import (
    FleeHazardOption,
    ForageOption,
    SMDPQLearning,
    WanderOption,
)
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.simulation.state.world_state import CellState


def test_forage_option_lifecycle() -> None:
    """Verifies initiation, action selection, and termination of ForageOption."""
    opt = ForageOption()

    # Observation with visible food cell at (3, 2)
    food_cell = CellState(
        position=Position(x=3, y=2),
        terrain=TerrainType.GRASSLAND,
        food=10.0,
        danger=0.0,
        movement_cost=1.0,
    )
    obs_with_food = Observation(
        agent_id="test_agent",
        step=1,
        position=Position(x=1, y=2),
        energy=80.0,
        health=100.0,
        carried_food=0,
        visible_cells=[food_cell],
        visible_agents=[],
    )

    # 1. Initiation check
    assert opt.can_initiate(state={}, observation=obs_with_food)

    # 2. Action selection: should move RIGHT towards (3, 2) from (1, 2)
    act = opt.select_action(state={}, observation=obs_with_food)
    assert act == ActionType.RIGHT

    # 3. Non-termination while away from food
    assert not opt.should_terminate(state={}, observation=obs_with_food)

    # 4. Termination when agent reaches food cell (3, 2)
    obs_at_food = Observation(
        agent_id="test_agent",
        step=2,
        position=Position(x=3, y=2),
        energy=79.0,
        health=100.0,
        carried_food=0,
        visible_cells=[food_cell],
        visible_agents=[],
    )
    assert opt.should_terminate(state={}, observation=obs_at_food)


def test_flee_hazard_option_lifecycle() -> None:
    """Verifies initiation, steering, and termination of FleeHazardOption."""
    opt = FleeHazardOption(risk_threshold=0.5)

    pos = Position(x=5, y=5)
    # High hazard directly UP at (5, 4)
    hazards = {
        Position(x=5, y=4): 0.9,  # UP
        Position(x=5, y=6): 0.1,  # DOWN
        Position(x=4, y=5): 0.2,  # LEFT
        Position(x=6, y=5): 0.3,  # RIGHT
    }
    state = {"hazard_probabilities": hazards}

    obs = Observation(
        agent_id="test_agent",
        step=1,
        position=pos,
        energy=80.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    # Can initiate because hazard risk 0.9 >= 0.5
    assert opt.can_initiate(state=state, observation=obs)

    # Action selection must steer towards safest direction (DOWN with risk 0.1)
    act = opt.select_action(state=state, observation=obs)
    assert act == ActionType.DOWN

    # If all local risks drop below threshold, option terminates
    safe_state = {"hazard_probabilities": {}}
    assert opt.should_terminate(state=safe_state, observation=obs)


def test_smdp_q_learning_multi_step_accumulation() -> None:
    """Verifies SMDP Q-learning accumulates multi-step discounted rewards over option durations."""
    options = [ForageOption(), WanderOption()]
    smdp = SMDPQLearning(options=options, alpha=0.5, gamma=0.9, epsilon=0.0, seed=42)

    obs = Observation(
        agent_id="test_agent",
        step=1,
        position=Position(x=0, y=0),
        energy=100.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )

    # Step 1: initiate option
    act, opt, just_init = smdp.step(state_key="s0", state={}, observation=obs)
    assert just_init
    assert smdp.active_option is not None

    # Record 2 steps of rewards
    smdp.record_step_reward(reward=2.0)  # r_0 = 2.0
    smdp.record_step_reward(reward=4.0)  # r_1 = 4.0 * 0.9 = 3.6

    # Total accumulated return: 2.0 + 3.6 = 5.6
    assert abs(smdp.option_discounted_return - 5.6) < 1e-6
    assert smdp.option_duration == 2
