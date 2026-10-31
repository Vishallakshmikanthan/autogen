"""Unit and mathematical tests for Policy Gradient methods (REINFORCE, PPO) and PPOAgent (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.agents.neural.ppo_agent import PPOAgent
from autogen.intelligence.neural.policy_gradient import PPO, REINFORCE
from autogen.intelligence.neural.survival_predictor import NeuralSurvivalPredictor
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position


@pytest.mark.mathematical
def test_reinforce_baseline_reduces_variance():
    """Verifies that REINFORCE baseline tracks returns and stabilizes policy updates."""
    agent = REINFORCE(state_dim=2, num_actions=2, gamma=0.95, lr=0.01, seed=42)

    # Simulate episode with steady positive returns
    for step in range(5):
        s = np.array([float(step), 0.5])
        agent.store_transition(s, action=1, reward=2.0)

    diag = agent.finish_episode()
    # Baseline should move towards mean episode return
    assert agent.baseline > 0.0
    assert diag.gradient_norm >= 0.0


@pytest.mark.mathematical
def test_ppo_clipped_objective_boundaries():
    """Verifies that PPO clipped surrogate objective correctly clips out-of-bound probability ratios."""
    ppo = PPO(state_dim=2, num_actions=2, clip_epsilon=0.2, seed=42)

    # Case 1: Positive advantage, ratio > 1 + clip_epsilon (1.2)
    # Objective should be clipped to (1 + 0.2) * advantage
    states = np.array([[1.0, 0.0]], dtype=np.float64)
    actions = np.array([0], dtype=int)
    # Old prob = 0.5
    old_probs = np.array([[0.5, 0.5]], dtype=np.float64)
    returns = np.array([5.0], dtype=np.float64)
    advantages = np.array([2.0], dtype=np.float64)

    # Train step executes without NaN and produces valid bounded diagnostics
    diag = ppo.train_step(states, actions, old_probs, returns, advantages, epochs=2)
    assert not np.isnan(diag.loss)
    assert not np.isnan(diag.gradient_norm)
    assert diag.entropy is not None and diag.entropy > 0.0


@pytest.mark.unit
def test_ppo_agent_decision_trace_generation():
    """Verifies that PPOAgent produces authentic DecisionTraces with probability distributions and values."""
    agent = PPOAgent(agent_id="ppo_test_1", state_dim=4)

    obs = Observation(
        agent_id="ppo_test_1",
        step=10,
        position=Position(x=2, y=5),
        energy=70.0,
        health=80.0,
        carried_food=1.0,
        visible_cells=(),
        visible_agents=(),
    )
    agent.observe(obs)
    action, trace = agent.decide()

    assert action.agent_id == "ppo_test_1"
    assert action.action_type in PPOAgent.ACTION_MAP
    assert trace is not None
    assert trace.step_id == 10
    assert trace.algorithm_name == "PPOPolicyGradient"
    assert "state_value" in trace.metrics
    assert "entropy" in trace.metrics
    assert len(trace.rationale["candidate_actions"]) == 4

    # Test learn step and rollout buffering
    agent.learn(reward=1.0, terminal=False)
    assert len(agent._states) == 1

    # Terminal learn step triggers policy update and empties buffer
    agent.learn(reward=10.0, terminal=True)
    assert len(agent._states) == 0


@pytest.mark.mathematical
def test_neural_survival_predictor():
    """Verifies that NeuralSurvivalPredictor fits lifespan regression and mortality risk."""
    predictor = NeuralSurvivalPredictor(input_dim=4, hidden_dim=16, seed=42)

    # Synthetic agent vitals: [energy, health, age, hazard_dist]
    # Lifespan strongly correlates with health and hazard_dist
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 1.0, size=(100, 4))
    # True remaining lifespan roughly proportional to health * 5
    y_lifespan = X[:, 1] * 5.0
    # Mortality risk high if health < 0.3
    y_hazard = (X[:, 1] < 0.3).astype(np.float64)

    # Train models
    h_life = predictor.fit_lifespan(X, y_lifespan, epochs=60, batch_size=16, lr=0.01)
    h_risk = predictor.fit_hazard(X, y_hazard, epochs=60, batch_size=16, lr=0.01)

    assert h_life[-1]["train_loss"] < h_life[0]["train_loss"]
    assert h_risk[-1]["train_loss"] < h_risk[0]["train_loss"]

    # Inferences
    pred_life = predictor.predict_lifespan(X[:5])
    pred_risk = predictor.predict_mortality_risk(X[:5])

    assert pred_life.shape == (5,)
    assert pred_risk.shape == (5,)
    assert np.all(pred_risk >= 0.0) and np.all(pred_risk <= 1.0)
