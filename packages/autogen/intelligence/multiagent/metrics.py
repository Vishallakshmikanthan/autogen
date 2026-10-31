"""Multi-agent metrics: Gini inequality, cooperation index, and collective survival (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MultiAgentMetrics:
    """Calculates statistical metrics for multi-agent collective dynamics and fairness."""

    @staticmethod
    def gini_coefficient(wealth_vector: Sequence[float]) -> float:
        """Calculates Gini coefficient of inequality in [0.0, 1.0].

        0.0 = perfect equality (everyone holds identical resources).
        1.0 = total inequality (one agent holds everything).
        """
        arr = np.asarray(wealth_vector, dtype=np.float64)
        n = arr.size
        if n <= 1:
            return 0.0

        total = np.sum(arr)
        if total <= 1e-12:
            return 0.0

        # Mean absolute difference between all pairs
        diff_sum = np.sum(np.abs(arr[:, np.newaxis] - arr[np.newaxis, :]))
        mean_val = total / n
        gini = float(diff_sum / (2.0 * n * n * mean_val))
        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def group_survival_rate(alive_flags: Sequence[bool]) -> float:
        """Proportion of agents that have survived."""
        if not alive_flags:
            return 0.0
        return float(np.mean(alive_flags))

    @staticmethod
    def cooperation_index(joint_actions: Sequence[tuple[int, ...]], cooperative_action_indices: set[int]) -> float:
        """Fraction of total individual actions that fall into cooperative action set."""
        if not joint_actions:
            return 0.0

        total_actions = 0
        coop_actions = 0
        for joint in joint_actions:
            for act in joint:
                total_actions += 1
                if act in cooperative_action_indices:
                    coop_actions += 1

        return float(coop_actions / max(1, total_actions))

    @staticmethod
    def calculate_summary(
        individual_returns: Sequence[float],
        alive_flags: Sequence[bool],
        resource_holdings: Sequence[float],
        communication_volume: int,
        commons_stock: float,
    ) -> dict[str, float]:
        """Compiles standard multi-agent metric summary dictionary."""
        ret_arr = np.asarray(individual_returns, dtype=np.float64)
        return {
            "mean_individual_return": float(np.mean(ret_arr)) if ret_arr.size > 0 else 0.0,
            "team_return": float(np.sum(ret_arr)),
            "group_survival_rate": MultiAgentMetrics.group_survival_rate(alive_flags),
            "gini_coefficient": MultiAgentMetrics.gini_coefficient(resource_holdings),
            "communication_volume": float(communication_volume),
            "commons_stock": float(commons_stock),
        }
