"""Multi-step Temporal Difference and Eligibility Trace algorithms (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.intelligence.rl.contracts import QTable


class MonteCarlo:
    """Monte Carlo prediction for state-value estimation (First-visit and Every-visit)."""

    def __init__(self, gamma: float = 0.99, first_visit: bool = True) -> None:
        self.gamma = gamma
        self.first_visit = first_visit
        self.v_values: dict[Any, float] = {}
        self.returns_count: dict[Any, int] = {}

    def update_episode(self, episode: list[tuple[Any, float]]) -> None:
        """Processes a complete episode of (state, reward) pairs."""
        G = 0.0
        visited_states: set[Any] = set()
        # Backward sweep to compute discounted returns G_t
        for state, reward in reversed(episode):
            G = reward + self.gamma * G
            if self.first_visit and state in visited_states:
                continue
            visited_states.add(state)

            n = self.returns_count.get(state, 0) + 1
            self.returns_count[state] = n
            curr_v = self.v_values.get(state, 0.0)
            self.v_values[state] = curr_v + (1.0 / n) * (G - curr_v)


class TDZero:
    """TD(0) state-value prediction."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.v_values: dict[Any, float] = {}

    def update(self, state: Any, reward: float, next_state: Any, terminated: bool) -> float:
        curr_v = self.v_values.get(state, 0.0)
        next_v = 0.0 if terminated else self.v_values.get(next_state, 0.0)
        target = reward + self.gamma * next_v
        td_error = target - curr_v
        self.v_values[state] = curr_v + self.alpha * td_error
        return float(td_error)


class NStepTD:
    """n-step Temporal Difference prediction."""

    def __init__(self, n: int = 3, alpha: float = 0.1, gamma: float = 0.99) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.v_values: dict[Any, float] = {}

    def update_trajectory(
        self,
        states: list[Any],
        rewards: list[float],
        terminated: bool,
    ) -> None:
        """Updates values along a collected trajectory of length T."""
        T = len(rewards)
        for t in range(T):
            # Target G_{t:t+n}
            G = 0.0
            limit = min(t + self.n, T)
            for i in range(t, limit):
                G += (self.gamma ** (i - t)) * rewards[i]

            if t + self.n < T:
                G += (self.gamma ** self.n) * self.v_values.get(states[t + self.n], 0.0)
            elif not terminated and T > 0:
                G += (self.gamma ** (T - t)) * self.v_values.get(states[-1], 0.0)

            st = states[t]
            curr_v = self.v_values.get(st, 0.0)
            self.v_values[st] = curr_v + self.alpha * (G - curr_v)


class TDLambda:
    """Watkins' Q(lambda) with eligibility traces for tabular control."""

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.8,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.q_table = QTable(num_actions=num_actions)
        self.traces: dict[tuple[Any, int], float] = {}

    def select_action(self, state: Any, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.q_table.best_action(state, self.rng)

    def reset_traces(self) -> None:
        self.traces.clear()

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool,
    ) -> float:
        # TD error
        best_next_a = self.q_table.best_action(next_state)
        q_next = 0.0 if terminated else self.q_table.get_q(next_state)[best_next_a]
        target = reward + self.gamma * q_next
        td_error = target - self.q_table.get_q(state)[action]

        # Accumulating trace
        sa = (state, action)
        self.traces[sa] = self.traces.get(sa, 0.0) + 1.0

        # Update Q values for all active traces
        for (s, a), trace_val in list(self.traces.items()):
            old_q = self.q_table.get_q(s)[a]
            self.q_table.set_q(s, a, old_q + self.alpha * td_error * trace_val)

        # Decay eligibility traces
        if terminated:
            self.reset_traces()
        else:
            decay = self.gamma * self.lam
            for key in list(self.traces.keys()):
                self.traces[key] *= decay
                if self.traces[key] < 1e-4:
                    del self.traces[key]

        return float(td_error)

