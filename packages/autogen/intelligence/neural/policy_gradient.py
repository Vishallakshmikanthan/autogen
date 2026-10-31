"""Policy gradient family: REINFORCE, Actor-Critic (A2C), and PPO (SYS-13, Module M14)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from autogen.intelligence.neural.activations import Linear, ReLU, Softmax
from autogen.intelligence.neural.contracts import NetworkDiagnostics
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.losses import MeanSquaredError
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import Adam, clip_grad_norm


class REINFORCE:
    """Monte Carlo Policy Gradient (REINFORCE) with running baseline."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (64,),
        gamma: float = 0.99,
        lr: float = 0.002,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        # Policy network: states -> action probabilities
        layers = []
        curr = state_dim
        rng_seed = seed
        for h in hidden_dims:
            layers.append(DenseLayer(curr, h, ReLU(), init_method="he", seed=rng_seed))
            curr = h
            rng_seed += 1
        layers.append(DenseLayer(curr, num_actions, Softmax(), init_method="xavier", seed=rng_seed))
        self.policy_net = NeuralNetwork(layers)

        self.optimizer = Adam(lr=lr)
        self.baseline = 0.0
        self.baseline_alpha = 0.05
        self.episode_states: list[np.ndarray] = []
        self.episode_actions: list[int] = []
        self.episode_rewards: list[float] = []

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return cast(np.ndarray, self.policy_net.predict(state)[0])

    def select_action(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        probs = self.get_action_probs(state)
        # Numerical safeguard for categorical sampling
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)
        action = int(self.rng.choice(self.num_actions, p=probs))
        return action, probs

    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        self.episode_states.append(np.asarray(state, dtype=np.float64))
        self.episode_actions.append(int(action))
        self.episode_rewards.append(float(reward))

    def finish_episode(self) -> NetworkDiagnostics:
        T = len(self.episode_rewards)
        if T == 0:
            return NetworkDiagnostics(loss=0.0, gradient_norm=0.0, learning_rate=self.optimizer.lr, step=0)

        # 1. Compute discounted returns G_t = sum_{k=t}^T gamma^(k-t) r_k
        returns = np.zeros(T, dtype=np.float64)
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = self.episode_rewards[t] + self.gamma * running_g
            returns[t] = running_g

        # Update baseline
        mean_return = float(np.mean(returns))
        self.baseline = (1.0 - self.baseline_alpha) * self.baseline + self.baseline_alpha * mean_return
        advantages = returns - self.baseline

        # Normalize advantages if variance > 0
        std_adv = np.std(advantages)
        if std_adv > 1e-8:
            advantages = (advantages - np.mean(advantages)) / std_adv

        states = np.array(self.episode_states, dtype=np.float64)
        actions = np.array(self.episode_actions, dtype=int)

        # Forward pass on all episode states
        probs = self.policy_net.forward(states, training=True)

        # Gradient of -log pi(a_t|s_t) * A_t:
        # Softmax + CrossEntropy gradient simplifies to: (probs - one_hot) * advantage
        grad_out = np.zeros_like(probs)
        for t in range(T):
            grad_out[t] = probs[t]
            grad_out[t, actions[t]] -= 1.0
            grad_out[t] *= advantages[t] / T

        self.policy_net.zero_grad()
        self.policy_net.backward(grad_out)
        gnorm = clip_grad_norm(self.policy_net.layers, 5.0)
        self.optimizer.step(self.policy_net.layers)

        # Reset episode buffers
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

        return NetworkDiagnostics(
            loss=float(np.mean(returns)),
            gradient_norm=float(gnorm),
            learning_rate=float(self.optimizer.lr),
            step=T,
        )


class PPO:
    """Proximal Policy Optimization (PPO) with clipped surrogate objective and actor-critic."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (64, 64),
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        lr_actor: float = 0.001,
        lr_critic: float = 0.002,
        c_value: float = 0.5,
        c_entropy: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.clip_epsilon = float(clip_epsilon)
        self.c_value = float(c_value)
        self.c_entropy = float(c_entropy)
        self.rng = np.random.default_rng(seed)

        # Actor network: state -> action probabilities (Softmax)
        act_layers = []
        curr = state_dim
        rng_seed = seed
        for h in hidden_dims:
            act_layers.append(DenseLayer(curr, h, ReLU(), init_method="he", seed=rng_seed))
            curr = h
            rng_seed += 1
        act_layers.append(DenseLayer(curr, num_actions, Softmax(), init_method="xavier", seed=rng_seed))
        self.actor = NeuralNetwork(act_layers)

        # Critic network: state -> state value V(s) (Linear)
        crit_layers = []
        curr = state_dim
        rng_seed += 100
        for h in hidden_dims:
            crit_layers.append(DenseLayer(curr, h, ReLU(), init_method="he", seed=rng_seed))
            curr = h
            rng_seed += 1
        crit_layers.append(DenseLayer(curr, 1, Linear(), init_method="xavier", seed=rng_seed))
        self.critic = NeuralNetwork(crit_layers)

        self.actor_optimizer = Adam(lr=lr_actor)
        self.critic_optimizer = Adam(lr=lr_critic)
        self.mse_loss = MeanSquaredError()

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return cast(np.ndarray, self.actor.predict(state)[0])

    def get_value(self, state: np.ndarray) -> float:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return float(self.critic.predict(state)[0, 0])

    def select_action(self, state: np.ndarray) -> tuple[int, np.ndarray, float]:
        probs = self.get_action_probs(state)
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)
        action = int(self.rng.choice(self.num_actions, p=probs))
        val = self.get_value(state)
        return action, probs, val

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        epochs: int = 4,
    ) -> NetworkDiagnostics:
        """Optimizes actor and critic over recorded rollouts."""
        n_samples = states.shape[0]

        # Normalize advantages
        adv_std = np.std(advantages)
        adv_norm = (advantages - np.mean(advantages)) / (adv_std + 1e-8) if adv_std > 1e-8 else advantages

        last_loss = 0.0
        last_entropy = 0.0
        last_gnorm = 0.0

        for _ in range(epochs):
            # 1. Update Critic: MSE between V(s) and returns
            v_pred = self.critic.forward(states, training=True)
            target_v = returns.reshape(-1, 1)
            v_loss = self.mse_loss.loss(target_v, v_pred)
            v_grad = self.mse_loss.gradient(target_v, v_pred)

            self.critic.zero_grad()
            self.critic.backward(v_grad)
            clip_grad_norm(self.critic.layers, 5.0)
            self.critic_optimizer.step(self.critic.layers)

            # 2. Update Actor: PPO Clipped Surrogate Objective
            curr_probs = self.actor.forward(states, training=True)
            # Clip for numerical stability
            curr_probs = np.clip(curr_probs, 1e-8, 1.0)
            curr_probs /= np.sum(curr_probs, axis=-1, keepdims=True)

            # Policy ratio: r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
            pi_a = curr_probs[np.arange(n_samples), actions]
            old_pi_a = old_probs[np.arange(n_samples), actions]
            ratio = pi_a / (old_pi_a + 1e-10)

            # Clipped ratio
            clipped_ratio = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

            surr1 = ratio * adv_norm
            surr2 = clipped_ratio * adv_norm
            policy_loss = -np.mean(np.minimum(surr1, surr2))

            # Entropy calculation: H = - sum p * log(p)
            entropy = -np.sum(curr_probs * np.log(curr_probs), axis=-1)
            mean_entropy = float(np.mean(entropy))

            # Backward through actor: gradient of surrogate with respect to logits
            # d(-min(surr1, surr2))/d(curr_probs)
            grad_actor = np.zeros_like(curr_probs)
            for i in range(n_samples):
                a_idx = actions[i]
                adv = adv_norm[i]
                r = ratio[i]

                # If surr1 <= surr2 or inside clipping window
                if surr1[i] <= surr2[i] or (1.0 - self.clip_epsilon < r < 1.0 + self.clip_epsilon):
                    grad_pi = -adv / (old_pi_a[i] + 1e-10)
                else:
                    grad_pi = 0.0

                # Backprop through softmax: grad_z_j = p_j * (grad_p_a * (delta_aj - p_a))
                for j in range(self.num_actions):
                    if j == a_idx:
                        grad_actor[i, j] += curr_probs[i, j] * (1.0 - curr_probs[i, a_idx]) * grad_pi
                    else:
                        grad_actor[i, j] += -curr_probs[i, j] * curr_probs[i, a_idx] * grad_pi

                    # Entropy regularization bonus gradient: -c_entropy * dH/dz
                    # dH/dz_j = - p_j * (log p_j + 1) - p_j * sum_k(-p_k(log p_k + 1))
                    log_p_j = np.log(curr_probs[i, j])
                    grad_actor[i, j] -= self.c_entropy * (curr_probs[i, j] * (log_p_j - (-entropy[i])))

            grad_actor /= n_samples

            self.actor.zero_grad()
            self.actor.backward(grad_actor)
            last_gnorm = clip_grad_norm(self.actor.layers, 5.0)
            self.actor_optimizer.step(self.actor.layers)

            last_loss = float(policy_loss + self.c_value * v_loss)
            last_entropy = mean_entropy

        return NetworkDiagnostics(
            loss=last_loss,
            gradient_norm=last_gnorm,
            entropy=last_entropy,
            learning_rate=float(self.actor_optimizer.lr),
            step=n_samples,
        )
