"""Value Decomposition Networks (VDN) and QMIX-Lite monotonic factorization (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ValueDecompositionNetworks:
    """VDN: Additive joint action-value factorization satisfying Individual-Global-Max (IGM)."""

    @staticmethod
    def joint_q(agent_q_values: Sequence[float]) -> float:
        """Computes joint Q_tot = sum_i Q_i."""
        return float(np.sum(agent_q_values))

    @staticmethod
    def individual_best_actions(all_agent_q_tables: Sequence[np.ndarray]) -> tuple[int, ...]:
        """Returns tuple of individual greedy actions: argmax_a_i Q_i."""
        return tuple(int(np.argmax(q_table)) for q_table in all_agent_q_tables)

    @staticmethod
    def verify_igm_condition(all_agent_q_tables: Sequence[np.ndarray]) -> bool:
        """Mathematically verifies that argmax_u Q_tot(u) equals (argmax_u1 Q1, ..., argmax_uN QN)."""
        # Optimal individual actions
        ind_best = tuple(int(np.argmax(q)) for q in all_agent_q_tables)

        # Enumerate joint actions on the cartesian product to find global argmax
        grids = np.meshgrid(*[range(len(q)) for q in all_agent_q_tables], indexing="ij")
        joint_coords = np.stack([g.ravel() for g in grids], axis=-1)

        best_joint_q = -float("inf")
        best_joint_action = None

        for coord in joint_coords:
            q_sum = sum(float(all_agent_q_tables[i][coord[i]]) for i in range(len(all_agent_q_tables)))
            if q_sum > best_joint_q:
                best_joint_q = q_sum
                best_joint_action = tuple(int(c) for c in coord)

        return best_joint_action == ind_best


class QMIXLite:
    """QMIX-Lite monotonic mixing network with non-negative weights ensuring dQ_tot / dQ_i >= 0."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 16, seed: int = 42) -> None:
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)

        # Hypernetwork 1: state -> absolute weights W1 of shape (num_agents, embed_dim)
        self.w1_net_w = rng.normal(0.0, 0.1, size=(state_dim, num_agents * embed_dim))
        self.b1_net_w = np.zeros(embed_dim)

        # Hypernetwork 2: state -> absolute weights W2 of shape (embed_dim, 1)
        self.w2_net_w = rng.normal(0.0, 0.1, size=(state_dim, embed_dim))
        self.b2_scalar = 0.0

    def forward(self, agent_qs: np.ndarray, state: np.ndarray) -> float:
        """Mixes individual agent Q-values into a single scalar joint Q_tot."""
        # agent_qs shape: (num_agents,)
        # state shape: (state_dim,)
        s = state.reshape(1, -1)
        q = agent_qs.reshape(1, self.num_agents)

        # 1. First mixing layer with strictly non-negative weights: |W1|
        w1_raw = s @ self.w1_net_w
        w1 = np.abs(w1_raw).reshape(self.num_agents, self.embed_dim)
        b1 = self.b1_net_w.reshape(1, self.embed_dim)

        hidden = np.maximum(0.0, q @ w1 + b1)  # (1, embed_dim) ReLU

        # 2. Second mixing layer with strictly non-negative weights: |W2|
        w2_raw = s @ self.w2_net_w
        w2 = np.abs(w2_raw).reshape(self.embed_dim, 1)

        q_tot = hidden @ w2 + self.b2_scalar
        return float(q_tot[0, 0])

    def check_monotonicity(self, state: np.ndarray, delta: float = 1.0) -> bool:
        """Verifies that increasing any agent's Q_i monotonically increases Q_tot."""
        base_qs = np.ones(self.num_agents, dtype=np.float64) * 5.0
        base_q_tot = self.forward(base_qs, state)

        for i in range(self.num_agents):
            perturbed_qs = np.copy(base_qs)
            perturbed_qs[i] += delta
            perturbed_q_tot = self.forward(perturbed_qs, state)
            if perturbed_q_tot < base_q_tot - 1e-9:
                return False
        return True
