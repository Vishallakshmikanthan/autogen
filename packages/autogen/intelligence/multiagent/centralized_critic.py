"""Centralized Training with Decentralized Execution (CTDE) centralized critic (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from autogen.intelligence.neural.activations import Linear, ReLU, Softmax
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.losses import MeanSquaredError
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import Adam, clip_grad_norm


class CentralizedCriticMARL:
    """CTDE controller with decentralized actors and a shared centralized critic."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        local_obs_dim: int,
        num_actions: int,
        global_state_dim: int,
        gamma: float = 0.95,
        lr_actor: float = 0.005,
        lr_critic: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.local_obs_dim = local_obs_dim
        self.num_actions = num_actions
        self.global_state_dim = global_state_dim
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        # 1. Decentralized Actors: o_i -> pi_i(a_i | o_i)
        self.actors: dict[str, NeuralNetwork] = {}
        self.actor_optimizers: dict[str, Adam] = {}

        for idx, aid in enumerate(self.agent_ids):
            actor_net = NeuralNetwork([
                DenseLayer(local_obs_dim, 32, ReLU(), init_method="he", seed=seed + idx * 10),
                DenseLayer(32, num_actions, Softmax(), init_method="xavier", seed=seed + idx * 10 + 1),
            ])
            self.actors[aid] = actor_net
            self.actor_optimizers[aid] = Adam(lr=lr_actor)

        # 2. Centralized Critic: global_state s -> V(s)
        self.critic = NeuralNetwork([
            DenseLayer(global_state_dim, 64, ReLU(), init_method="he", seed=seed + 100),
            DenseLayer(64, 1, Linear(), init_method="xavier", seed=seed + 101),
        ])
        self.critic_optimizer = Adam(lr=lr_critic)
        self.mse_loss = MeanSquaredError()

    def select_action(self, agent_id: str, local_obs: np.ndarray) -> tuple[int, np.ndarray]:
        """Decentralized action selection conditioned strictly on local observation."""
        actor = self.actors[agent_id]
        if local_obs.ndim == 1:
            local_obs = local_obs.reshape(1, -1)
        probs = actor.predict(local_obs)[0]
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)
        action = int(self.rng.choice(self.num_actions, p=probs))
        return action, probs

    def update(
        self,
        global_state: np.ndarray,
        local_obs_map: Mapping[str, np.ndarray],
        actions_map: Mapping[str, int],
        team_reward: float,
        next_global_state: np.ndarray,
        terminated: bool,
    ) -> dict[str, float]:
        """Performs centralized critic TD update and decentralized actor policy gradient updates."""
        s = global_state.reshape(1, -1) if global_state.ndim == 1 else global_state
        s_next = next_global_state.reshape(1, -1) if next_global_state.ndim == 1 else next_global_state

        # 1. Critic evaluation & TD target: y = r_team + gamma * V(s')
        v_next = float(self.critic.predict(s_next)[0, 0])
        target_v = team_reward if terminated else team_reward + self.gamma * v_next
        target_arr = np.array([[target_v]], dtype=np.float64)

        v_pred = self.critic.forward(s, training=True)
        critic_loss = self.mse_loss.loss(target_arr, v_pred)
        critic_grad = self.mse_loss.gradient(target_arr, v_pred)

        self.critic.zero_grad()
        self.critic.backward(critic_grad)
        clip_grad_norm(self.critic.layers, 5.0)
        self.critic_optimizer.step(self.critic.layers)

        # Advantage estimate: A = target - V(s)
        v_current = float(v_pred[0, 0])
        advantage = target_v - v_current

        # 2. Update decentralized actors using centralized advantage
        actor_losses: dict[str, float] = {}
        for aid in self.agent_ids:
            obs = local_obs_map[aid]
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)

            act_idx = actions_map[aid]
            actor = self.actors[aid]
            opt = self.actor_optimizers[aid]

            probs = actor.forward(obs, training=True)
            # Softmax policy gradient: (probs - one_hot) * (-advantage)
            grad_out = np.zeros_like(probs)
            grad_out[0] = probs[0]
            grad_out[0, act_idx] -= 1.0
            grad_out[0] *= -advantage

            actor.zero_grad()
            actor.backward(grad_out)
            clip_grad_norm(actor.layers, 5.0)
            opt.step(actor.layers)

            actor_losses[f"actor_{aid}_loss"] = float(-advantage)

        actor_losses["critic_loss"] = float(critic_loss)
        actor_losses["advantage"] = float(advantage)
        return actor_losses
