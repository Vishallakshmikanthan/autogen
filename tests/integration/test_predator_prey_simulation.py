"""Integration test: Multi-agent adversarial simulation with predator-prey lookahead and viewer adapter (Module M07)."""

from __future__ import annotations

import pytest

from autogen.agents.adversarial.predator_agent import PredatorAgent
from autogen.agents.adversarial.prey_agent import PreyAgent
from autogen.agents.adversarial.stochastic_opponent import StochasticOpponent
from autogen.agents.adversarial.strategic_forager import StrategicForager
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position, TERRAIN_PROPERTIES
from autogen.simulation.state.world_state import canonical_world_hash
from autogen.visualization.adapter import ViewerDataAdapter


def _build_adversarial_simulation(seed: int = 123, prey_near_predator: bool = False) -> SimulationEngine:
    config = EnvironmentConfig(
        width=20,
        height=20,
        seed=seed,
    )
    world = GridWorld(config)
    world.reset()

    predator = PredatorAgent(agent_id="pred_1", max_lookahead_depth=2, target_prey_id="prey_1")
    prey = PreyAgent(agent_id="prey_1", max_lookahead_depth=2, predator_id="pred_1")
    forager = StrategicForager(agent_id="forager_1", strategy_mode="mixed_nash", seed=seed)
    stoch = StochasticOpponent(agent_id="stoch_1", seed=seed)

    pred_state = world.spawn_agent("pred_1")
    prey_pos = None
    if prey_near_predator:
        for dx, dy in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]:
            cand = Position(pred_state.position.x + dx, pred_state.position.y + dy)
            if world.is_in_bounds(cand) and TERRAIN_PROPERTIES[world.state.get_cell(cand).terrain].walkable:
                prey_pos = cand
                break
    world.spawn_agent("prey_1", position=prey_pos)
    world.spawn_agent("forager_1")
    world.spawn_agent("stoch_1")

    engine = SimulationEngine(environment=world, seed=seed)
    p_cfg = PerceptionConfig(radius=6)

    engine.register_agent(predator, p_cfg)
    engine.register_agent(prey, p_cfg)
    engine.register_agent(forager, p_cfg)
    engine.register_agent(stoch, p_cfg)

    return engine


def test_adversarial_simulation_determinism() -> None:
    """Verifies that running the adversarial predator-prey simulation is strictly deterministic."""
    engine1 = _build_adversarial_simulation(seed=42)
    engine2 = _build_adversarial_simulation(seed=42)

    hashes1: list[str] = []
    hashes2: list[str] = []

    for _ in range(50):
        s1 = engine1.step()
        s2 = engine2.step()
        hashes1.append(canonical_world_hash(s1))
        hashes2.append(canonical_world_hash(s2))

    assert hashes1 == hashes2, "Non-deterministic simulation divergence detected!"


def test_hunt_lookahead_tree_exported_to_viewer_adapter() -> None:
    """Verifies the integration hook: 'Hunt' preset lookahead tree is exposed in ViewerState."""
    engine = _build_adversarial_simulation(seed=99, prey_near_predator=True)
    adapter = ViewerDataAdapter(engine)

    # Step simulation until predator has made decisions
    for _ in range(5):
        engine.step()

    viewer_state = adapter.generate_viewer_state(selected_agent_id="pred_1")

    assert viewer_state.selected_agent_id == "pred_1"
    assert viewer_state.decision_trace is not None
    assert viewer_state.decision_trace.agent_id == "pred_1"

    # Verify lookahead tree serialization
    assert viewer_state.lookahead_tree is not None
    assert "node_id" in viewer_state.lookahead_tree
    assert "value" in viewer_state.lookahead_tree
    assert "children" in viewer_state.lookahead_tree
    assert len(viewer_state.lookahead_tree["children"]) > 0
