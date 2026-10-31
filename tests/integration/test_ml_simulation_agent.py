"""Integration tests for LearnedPerceptionAgent in deterministic SimulationEngine (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np

from autogen.agents.ml.learned_perception_agent import LearnedPerceptionAgent
from autogen.intelligence.ml.applications import (
    CellDangerClassifier,
    LearnedPerceptionComponent,
    ResourceYieldPredictor,
)
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import canonical_world_hash
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def test_learned_perception_agent_simulation_run() -> None:
    """Verifies that LearnedPerceptionAgent runs in SimulationEngine and produces authentic DecisionTraces."""
    rng = np.random.default_rng(42)
    config = EnvironmentConfig(width=8, height=8, seed=42)
    env = GridWorld(config)
    env.reset()

    # Train danger classifier: high (x+y) has danger
    synth_X = rng.uniform(0, 8, size=(150, 3))
    synth_y = (synth_X[:, 0] + synth_X[:, 1] > 8.0).astype(int)
    danger_clf = CellDangerClassifier(model_type="logistic", seed=42)
    danger_clf.fit(synth_X, synth_y)

    # Train yield predictor
    yield_pred = ResourceYieldPredictor(model_type="linear", seed=42)
    yield_pred.fit(synth_X, synth_X[:, 0] * 1.5)

    comp = LearnedPerceptionComponent(danger_classifier=danger_clf, resource_predictor=yield_pred)
    agent = LearnedPerceptionAgent(agent_id="ml_scout", perception_component=comp, danger_threshold=0.45)
    env.spawn_agent("ml_scout")

    sim = SimulationEngine(environment=env, agents={"ml_scout": agent}, seed=42)

    for _ in range(25):
        sim.step()
        trace = agent.latest_trace
        assert trace is not None
        assert trace.agent_id == "ml_scout"
        assert trace.algorithm_name == "LearnedPerceptionPolicy"
        assert "belief_summary" in trace.rationale
        assert "candidate_evaluations" in trace.rationale["belief_summary"]
        assert "chosen_predicted_danger" in trace.metrics
        assert len(trace.rationale["belief_summary"]["candidate_evaluations"]) == 4


def test_learned_perception_agent_determinism() -> None:
    """Verifies that two simulation runs with identical seeds produce bit-identical traces and state hashes."""
    def run_sim(seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        config = EnvironmentConfig(width=6, height=6, seed=seed)
        env = GridWorld(config)
        env.reset()

        synth_X = rng.uniform(0, 6, size=(80, 3))
        synth_y = (synth_X[:, 0] > 3.0).astype(int)
        danger_clf = CellDangerClassifier(model_type="logistic", seed=seed)
        danger_clf.fit(synth_X, synth_y)

        comp = LearnedPerceptionComponent(danger_classifier=danger_clf)
        agent = LearnedPerceptionAgent(agent_id="scout", perception_component=comp)
        env.spawn_agent("scout")

        sim = SimulationEngine(environment=env, agents={"scout": agent}, seed=seed)

        hashes = []
        for _ in range(15):
            s = sim.step()
            hashes.append(canonical_world_hash(s))
        return hashes

    hashes_run1 = run_sim(42)
    hashes_run2 = run_sim(42)
    assert hashes_run1 == hashes_run2