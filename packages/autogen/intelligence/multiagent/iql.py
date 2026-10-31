"""Independent Q-Learning (IQL) and Independent Policy Gradient (IPG) (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autogen.intelligence.rl.tabular import QLearning


class IndependentQLearning:
    """Multi-agent controller managing independent decentralized tabular Q-learners."""

    def __init__(
        self,
        agent_ids: list[str],
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.num_actions = num_actions
        self.learners: dict[str, QLearning] = {}

        for idx, aid in enumerate(agent_ids):
            self.learners[aid] = QLearning(
                num_actions=num_actions,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                seed=seed + idx * 10,
            )

    def select_actions(
        self,
        observations: Mapping[str, Any],
        explore: bool = True,
    ) -> dict[str, int]:
        """Selects decentralized action for each observed agent."""
        joint_actions: dict[str, int] = {}
        for aid, obs in observations.items():
            if aid in self.learners:
                joint_actions[aid] = self.learners[aid].select_action(obs, explore=explore)
        return joint_actions

    def update(
        self,
        prev_obs: Mapping[str, Any],
        actions: Mapping[str, int],
        rewards: Mapping[str, float],
        next_obs: Mapping[str, Any],
        terminated: bool,
    ) -> dict[str, float]:
        """Performs decentralized independent Bellman updates for each agent."""
        td_errors: dict[str, float] = {}
        for aid in self.learners:
            if aid in prev_obs and aid in actions and aid in rewards and aid in next_obs:
                td_err = self.learners[aid].update(
                    state=prev_obs[aid],
                    action=actions[aid],
                    reward=rewards[aid],
                    next_state=next_obs[aid],
                    terminated=terminated,
                )
                td_errors[aid] = td_err
        return td_errors
