"""Emergent behavioral role detection via unsupervised clustering (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from autogen.intelligence.ml.clustering import KMeans


class RoleDetector:
    """Detects emergent multi-agent social roles (Scout, Hoarder, Free-Rider) from behavioral trajectories."""

    ROLE_NAMES = ("SCOUT", "HOARDER", "FREE_RIDER", "COOPERATOR")

    def __init__(self, n_roles: int = 3, seed: int = 42) -> None:
        self.n_roles = min(4, max(2, n_roles))
        self.clusterer = KMeans(n_clusters=self.n_roles, seed=seed)
        self.fitted = False

    def fit_predict(
        self,
        agent_features: Mapping[str, np.ndarray],
    ) -> dict[str, str]:
        """Clusters agent behavior vectors and assigns functional role identities.

        Features per agent expected as:
        [foraging_rate, exploration_radius, communication_count, conflict_rate]
        """
        agent_ids = list(agent_features.keys())
        if len(agent_ids) < self.n_roles:
            # Fallback if agent count is smaller than requested clusters
            return {aid: "COOPERATOR" for aid in agent_ids}

        X = np.array([agent_features[aid] for aid in agent_ids], dtype=np.float64)
        labels = self.clusterer.fit(X).predict(X)
        self.fitted = True

        role_assignments: dict[str, str] = {}
        for idx, aid in enumerate(agent_ids):
            cluster_id = int(labels[idx])
            role_assignments[aid] = self.ROLE_NAMES[cluster_id % len(self.ROLE_NAMES)]

        return role_assignments
