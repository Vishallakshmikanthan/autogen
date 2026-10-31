"""Model-based Tabular Reinforcement Learning: Dyna-Q (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.intelligence.rl.contracts import QTable, RLAlgorithm


class DynaQ(RLAlgorithm):
    """Dyna-Q combining direct RL updates with model-based planning backups."""

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        n_planning_steps: int = 10,
        seed: int = 42,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        self.rng = np.random.default_rng(seed)

        self.q_table = QTable(num_actions=num_actions)
        # Deterministic tabular model: (s, a) -> (r, s', terminated)
        self.model: dict[tuple[Any, int], tuple[float, Any, bool]] = {}
        # Track observed (s, a) pairs for planning sampling
        self.observed_states: list[Any] = []
        self.observed_actions_by_state: dict[Any, set[int]] = {}

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
        # 1. Direct Q-learning update
        if terminated:
            target = reward
        else:
            q_next = self.q_table.get_q(next_state)
            target = reward + self.gamma * float(np.max(q_next))

        current_q = self.q_table.get_q(state)[action]
        td_error = target - current_q
        self.q_table.set_q(state, action, current_q + self.alpha * td_error)

        # 2. Learn environment model
        self.model[(state, action)] = (reward, next_state, terminated)
        if state not in self.observed_actions_by_state:
            self.observed_states.append(state)
            self.observed_actions_by_state[state] = set()
        self.observed_actions_by_state[state].add(action)

        # 3. Model-based planning loop (simulate n steps from learned model)
        if self.observed_states and self.n_planning_steps > 0:
            for _ in range(self.n_planning_steps):
                s_sim_idx = int(self.rng.integers(0, len(self.observed_states)))
                s_sim = self.observed_states[s_sim_idx]
                actions = list(self.observed_actions_by_state[s_sim])
                a_sim = actions[int(self.rng.integers(0, len(actions)))]

                r_sim, next_s_sim, term_sim = self.model[(s_sim, a_sim)]
                if term_sim:
                    plan_target = r_sim
                else:
                    plan_target = r_sim + self.gamma * float(np.max(self.q_table.get_q(next_s_sim)))

                old_plan_q = self.q_table.get_q(s_sim)[a_sim]
                self.q_table.set_q(s_sim, a_sim, old_plan_q + self.alpha * (plan_target - old_plan_q))

        return float(td_error)

