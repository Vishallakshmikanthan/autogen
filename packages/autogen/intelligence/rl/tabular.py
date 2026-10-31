"""Tabular Reinforcement Learning Control algorithms: Q-Learning, SARSA, Expected SARSA, Double Q-Learning (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.intelligence.rl.contracts import QTable, RLAlgorithm


class QLearning(RLAlgorithm):
    """Off-policy TD control (Q-learning) with strict terminal zero-bootstrapping."""

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = QTable(num_actions=num_actions)

    def select_action(self, state: Any, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.q_table.best_action(state, self.rng)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        # STRICT TERMINAL HANDLING: if terminated, target = reward exactly (zero bootstrap)
        if terminated:
            target = reward
        else:
            q_next = self.q_table.get_q(next_state)
            target = reward + self.gamma * float(np.max(q_next))

        current_q = self.q_table.get_q(state)[action]
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table.set_q(state, action, new_q)
        return float(td_error)


class SARSA(RLAlgorithm):
    """On-policy TD control (SARSA) with strict terminal zero-bootstrapping."""

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = QTable(num_actions=num_actions)

    def select_action(self, state: Any, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.q_table.best_action(state, self.rng)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        if terminated:
            target = reward
        else:
            if next_action is None:
                next_action = self.select_action(next_state, explore=True)
            q_next_a = self.q_table.get_q(next_state)[next_action]
            target = reward + self.gamma * q_next_a

        current_q = self.q_table.get_q(state)[action]
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table.set_q(state, action, new_q)
        return float(td_error)


class ExpectedSARSA(RLAlgorithm):
    """Expected SARSA computing expected next-state action value under current epsilon-greedy policy."""

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = QTable(num_actions=num_actions)

    def select_action(self, state: Any, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.q_table.best_action(state, self.rng)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        if terminated:
            target = reward
        else:
            q_next = self.q_table.get_q(next_state)
            best_a = self.q_table.best_action(next_state)
            # Epsilon-greedy probabilities over next state actions
            probs = np.full(self.num_actions, self.epsilon / self.num_actions, dtype=np.float64)
            probs[best_a] += (1.0 - self.epsilon)
            expected_next_q = float(np.sum(probs * q_next))
            target = reward + self.gamma * expected_next_q

        current_q = self.q_table.get_q(state)[action]
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table.set_q(state, action, new_q)
        return float(td_error)


class DoubleQLearning(RLAlgorithm):
    """Double Q-Learning eliminating maximization bias by maintaining two independent estimators."""

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q1_table = QTable(num_actions=num_actions)
        self.q2_table = QTable(num_actions=num_actions)

    def select_action(self, state: Any, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        # Choose action greedily with respect to Q1 + Q2
        combined_q = self.q1_table.get_q(state) + self.q2_table.get_q(state)
        max_q = float(np.max(combined_q))
        best_actions = np.where(np.isclose(combined_q, max_q, atol=1e-8))[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        # 50/50 update alternating between Q1 and Q2
        if self.rng.random() < 0.5:
            # Update Q1 using Q2 for value evaluation
            if terminated:
                target = reward
            else:
                best_a = self.q1_table.best_action(next_state, self.rng)
                target = reward + self.gamma * self.q2_table.get_q(next_state)[best_a]

            current_q = self.q1_table.get_q(state)[action]
            td_error = target - current_q
            self.q1_table.set_q(state, action, current_q + self.alpha * td_error)
            return float(td_error)
        else:
            # Update Q2 using Q1 for value evaluation
            if terminated:
                target = reward
            else:
                best_a = self.q2_table.best_action(next_state, self.rng)
                target = reward + self.gamma * self.q1_table.get_q(next_state)[best_a]

            current_q = self.q2_table.get_q(state)[action]
            td_error = target - current_q
            self.q2_table.set_q(state, action, current_q + self.alpha * td_error)
            return float(td_error)

