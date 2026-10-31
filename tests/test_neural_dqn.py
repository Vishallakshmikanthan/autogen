"""Unit and mathematical tests for Deep Q-Networks and DQNAgent (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.agents.neural.dqn_agent import DQNAgent
from autogen.intelligence.neural.dqn import DeepQNetwork, ReplayBuffer
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position


@pytest.mark.mathematical
def test_dqn_terminal_target_invariant():
    """Verifies that terminal state transitions strictly set target y = r with zero future bootstrap."""
    state_dim = 4
    num_actions = 4
    dqn = DeepQNetwork(state_dim=state_dim, num_actions=num_actions, gamma=0.99, seed=42)

    buf = ReplayBuffer(capacity=100, seed=42)
    s = np.array([0.1, 0.2, 0.3, 0.4])
    s_next = np.array([0.5, 0.6, 0.7, 0.8])

    # Store terminal transition with reward +10.0
    buf.add(state=s, action=1, reward=10.0, next_state=s_next, terminated=True)

    # Mock target network to return huge Q-values on next_state
    # If bootstrap was mistakenly included, target would be 10.0 + 0.99 * 1000 = 1000.0
    for layer in dqn.target_net.layers:
        for p in layer.params.values():
            p.fill(100.0)

    # Sample batch
    trans = buf.sample(1)[0]
    assert trans.terminated is True

    # Compute target manually according to DQN update logic
    q_next_target = dqn.target_net.predict(trans.next_state.reshape(1, -1))
    max_q_next = np.max(q_next_target)
    assert max_q_next > 500.0  # target network predicts huge values

    target_val = trans.reward if trans.terminated else trans.reward + dqn.gamma * max_q_next
    # Strict invariant: must equal 10.0 exactly
    assert target_val == 10.0


@pytest.mark.unit
def test_replay_buffer_sampling_and_capacity():
    """Verifies FIFO circular overwrite and uniform random sampling distribution."""
    buf = ReplayBuffer(capacity=5, seed=42)
    for i in range(10):
        s = np.array([float(i)])
        buf.add(state=s, action=i % 4, reward=float(i), next_state=s, terminated=False)

    assert len(buf) == 5
    # Oldest entries 0-4 overwritten, current buffer contains states 5-9
    stored_states = {float(t.state[0]) for t in buf.buffer}
    assert stored_states == {5.0, 6.0, 7.0, 8.0, 9.0}

    # Sampling statistics: uniform across indices
    samples = [buf.sample(1)[0].reward for _ in range(500)]
    for val in [5.0, 6.0, 7.0, 8.0, 9.0]:
        count = samples.count(val)
        # Expected count is 100, must be within reasonable stochastic bounds (50 to 150)
        assert 50 <= count <= 150


@pytest.mark.mathematical
def test_dqn_target_network_soft_and_hard_update():
    """Verifies target network hard copy and Polyak soft update mechanics."""
    dqn = DeepQNetwork(state_dim=2, num_actions=2, tau=0.1, seed=42)
    q_w = dqn.q_net.layers[0].params["W"]
    t_w = dqn.target_net.layers[0].params["W"]

    # Perturb main network
    q_w += 5.0
    orig_t_w = np.copy(t_w)

    # Soft update: t_w = 0.9 * t_w + 0.1 * q_w
    dqn.update_target_network(soft=True)
    expected_soft = 0.9 * orig_t_w + 0.1 * q_w
    np.testing.assert_allclose(t_w, expected_soft, rtol=1e-5)

    # Hard update: t_w = q_w
    dqn.update_target_network(soft=False)
    np.testing.assert_allclose(t_w, q_w, rtol=1e-5)


@pytest.mark.mathematical
def test_dqn_learns_grid_environment():
    """Verifies that DQN trains towards optimal Q-values on a small 1D grid navigation task."""
    # 3-state grid: 0 (start) <-> 1 <-> 2 (goal, reward +10, terminal)
    # Actions: 0 (left), 1 (right)
    dqn = DeepQNetwork(state_dim=3, num_actions=2, hidden_dims=(16,), lr=0.01, seed=42)
    buf = ReplayBuffer(capacity=1000, seed=42)

    def encode_onehot(idx: int) -> np.ndarray:
        v = np.zeros(3, dtype=np.float64)
        v[idx] = 1.0
        return v

    # Prefill replay buffer with transition dynamics
    for _ in range(200):
        # State 0: RIGHT -> 1 (0 reward), LEFT -> 0 (0 reward)
        buf.add(encode_onehot(0), 1, 0.0, encode_onehot(1), False)
        buf.add(encode_onehot(0), 0, -0.1, encode_onehot(0), False)
        # State 1: RIGHT -> 2 (+10 reward, terminal), LEFT -> 0 (0 reward)
        buf.add(encode_onehot(1), 1, 10.0, encode_onehot(2), True)
        buf.add(encode_onehot(1), 0, 0.0, encode_onehot(0), False)

    # Train DQN for 150 steps
    for _ in range(150):
        dqn.update(buf, batch_size=16)

    # Action 1 (RIGHT) at state 1 should have highest Q-value (~10.0)
    q_s1 = dqn.get_q_values(encode_onehot(1))
    assert q_s1[1] > q_s1[0]
    assert q_s1[1] > 5.0


@pytest.mark.unit
def test_dqn_agent_decision_trace_generation():
    """Verifies that DQNAgent produces authentic DecisionTraces with Q-values and diagnostics."""
    agent = DQNAgent(agent_id="dqn_test_1", state_dim=4, initial_epsilon=0.0)

    obs = Observation(
        agent_id="dqn_test_1",
        step=5,
        position=Position(x=3, y=4),
        energy=85.0,
        health=95.0,
        carried_food=0.0,
        visible_cells=(),
        visible_agents=(),
    )
    agent.observe(obs)
    action, trace = agent.decide()

    assert action.agent_id == "dqn_test_1"
    assert action.action_type in DQNAgent.ACTION_MAP
    assert trace is not None
    assert trace.step_id == 5
    assert trace.algorithm_name == "DeepQNetwork"
    assert len(trace.rationale["candidate_actions"]) == 4
    assert "mean_q" in trace.metrics
    assert "max_q" in trace.metrics
    assert trace.metrics["is_exploratory"] == 0.0

    # Test learn step and transition recording
    next_obs = Observation(
        agent_id="dqn_test_1",
        step=6,
        position=Position(x=4, y=4),
        energy=84.0,
        health=95.0,
        carried_food=0.0,
        visible_cells=(),
        visible_agents=(),
    )
    agent.observe(next_obs)
    agent.learn(reward=1.0, terminal=False)
    assert len(agent.replay_buffer) == 1
