"""Deep Q-Network (DQN) with experience replay and target networks (SYS-13, Module M14)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from autogen.intelligence.neural.activations import Linear, ReLU
from autogen.intelligence.neural.contracts import ExperienceTransition, NetworkDiagnostics
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.losses import HuberLoss
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import Adam, clip_grad_norm


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with uniform random sampling."""

    def __init__(self, capacity: int = 10000, seed: int = 42) -> None:
        self.capacity = max(1, capacity)
        self.rng = np.random.default_rng(seed)
        self.buffer: list[ExperienceTransition] = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Stores transition in circular buffer."""
        trans = ExperienceTransition(
            state=np.asarray(state, dtype=np.float64),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float64),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(trans)
        else:
            self.buffer[self.position] = trans
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[ExperienceTransition]:
        """Samples a uniform random mini-batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} samples, fewer than batch_size {batch_size}")
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class DeepQNetwork:
    """Deep Q-Learning algorithm with target network and experience replay."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (64, 64),
        gamma: float = 0.99,
        lr: float = 0.001,
        tau: float = 0.005,
        target_update_interval: int = 200,
        max_grad_norm: float = 10.0,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.tau = float(tau)
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.rng = np.random.default_rng(seed)

        # Build Q-network
        self.q_net = self._build_mlp(state_dim, num_actions, hidden_dims, seed=seed)
        # Build Target Q-network
        self.target_net = self._build_mlp(state_dim, num_actions, hidden_dims, seed=seed + 100)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.loss_fn = HuberLoss(delta=1.0)
        self.optimizer = Adam(lr=self.lr)
        self.step_counter = 0

    def _build_mlp(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        seed: int,
    ) -> NeuralNetwork:
        layers = []
        curr_dim = in_dim
        rng_seed = seed
        for h_dim in hidden_dims:
            layers.append(
                DenseLayer(
                    in_features=curr_dim,
                    out_features=h_dim,
                    activation=ReLU(),
                    init_method="he",
                    seed=rng_seed,
                )
            )
            curr_dim = h_dim
            rng_seed += 1
        # Output layer with linear activation (Q-values can be arbitrary scalars)
        layers.append(
            DenseLayer(
                in_features=curr_dim,
                out_features=out_dim,
                activation=Linear(),
                init_method="xavier",
                seed=rng_seed,
            )
        )
        return NeuralNetwork(layers)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a) for all actions."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return cast(np.ndarray, self.q_net.predict(state)[0])

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Selects action using epsilon-greedy policy over current Q-network."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.num_actions))
        q_vals = self.get_q_values(state)
        max_q = float(np.max(q_vals))
        best_actions = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 32) -> NetworkDiagnostics:
        """Samples mini-batch and performs one step of DQN gradient descent."""
        self.step_counter += 1
        transitions = replay_buffer.sample(batch_size)

        states = np.array([t.state for t in transitions], dtype=np.float64)
        actions = np.array([t.action for t in transitions], dtype=int)
        rewards = np.array([t.reward for t in transitions], dtype=np.float64)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float64)
        terminated = np.array([t.terminated for t in transitions], dtype=bool)

        # 1. Evaluate target network on next states: max_a' Q(s', a'; theta^-)
        q_next_target = self.target_net.predict(next_states)
        max_q_next = np.max(q_next_target, axis=1)

        # 2. Compute Bellman targets: y = r if terminated else r + gamma * max_a' Q(s', a')
        targets = np.where(terminated, rewards, rewards + self.gamma * max_q_next)

        # 3. Current Q-predictions
        current_q_all = self.q_net.forward(states, training=True)
        target_q_matrix = np.copy(current_q_all)
        for i in range(batch_size):
            target_q_matrix[i, actions[i]] = targets[i]

        # 4. Compute loss and backpropagate
        loss_val = self.loss_fn.loss(target_q_matrix, current_q_all)
        grad_out = self.loss_fn.gradient(target_q_matrix, current_q_all)

        self.q_net.zero_grad()
        self.q_net.backward(grad_out)

        # 5. Gradient clipping and optimizer step
        gnorm = clip_grad_norm(self.q_net.layers, self.max_grad_norm)
        self.optimizer.step(self.q_net.layers)

        # 6. Target network update
        if self.step_counter % self.target_update_interval == 0:
            self.update_target_network(soft=False)
        elif self.tau > 0.0:
            self.update_target_network(soft=True)

        return NetworkDiagnostics(
            loss=float(loss_val),
            gradient_norm=float(gnorm),
            mean_q=float(np.mean(current_q_all)),
            max_q=float(np.max(current_q_all)),
            learning_rate=float(self.optimizer.lr),
            step=self.step_counter,
        )

    def update_target_network(self, soft: bool = False) -> None:
        """Synchronizes target network with main Q-network (hard copy or Polyak soft update)."""
        if not soft:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            q_state = self.q_net.state_dict()
            target_state = self.target_net.state_dict()
            for layer_key in target_state:
                for p_name in target_state[layer_key]:
                    target_state[layer_key][p_name] = (
                        (1.0 - self.tau) * target_state[layer_key][p_name]
                        + self.tau * q_state[layer_key][p_name]
                    )
            self.target_net.load_state_dict(target_state)
