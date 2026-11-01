"""Village Trust Graph, Reciprocity Dynamics, Gossip Propagation, and Trust PageRank."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrustGraphConfig(BaseModel):
    """Hyperparameters governing trust dynamics and PageRank societal status (§2.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_coop: float = Field(
        default=0.14,
        gt=0.0,
        le=1.0,
        description="Reciprocity trust gain rate on cooperation or tithing",
    )
    beta_defect: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Trust loss rate on defection, lying, or hoarding (must exceed alpha_coop)",
    )
    gamma_crime: float = Field(
        default=0.55,
        gt=0.0,
        le=1.0,
        description="Severe trust penalty when caught stealing or sabotaging",
    )
    lambda_decay: float = Field(
        default=0.004,
        ge=0.0,
        lt=1.0,
        description="Per-step memory attenuation rate",
    )
    damping_factor: float = Field(
        default=0.85,
        gt=0.0,
        lt=1.0,
        description="PageRank damping factor d",
    )
    initial_trust: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Initial neutral trust rating between strangers",
    )
    gossip_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Credibility weight when integrating third-party gossip",
    )
    pagerank_tol: float = Field(default=1e-8, gt=0.0)
    pagerank_max_iter: int = Field(default=200, ge=10)

    @model_validator(mode="after")
    def _check_asymmetry_invariant(self) -> TrustGraphConfig:
        if self.beta_defect <= self.alpha_coop:
            raise ValueError(
                f"Sociological asymmetry invariant violated: beta_defect ({self.beta_defect}) "
                f"must strictly exceed alpha_coop ({self.alpha_coop})."
            )
        return self


class PageRankStepEvent(BaseModel):
    """Step-traceable event emitted at each power-iteration step of Trust PageRank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration: int = Field(ge=0)
    l1_residual: float = Field(ge=0.0)
    status_vector: dict[str, float]


class TrustEdgeSummary(BaseModel):
    """Serializable directed trust edge $T_{j \\to i}$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    target_id: str
    trust: float = Field(ge=0.0, le=1.0)


class VillageTrustGraph:
    """Directed asymmetric trust network $\\mathbf{T}(t) \\in [0, 1]^{N \\times N}$ and PageRank engine.

    Maintains $T_{j \\to i}(t)$ where row $j$ is the evaluating observer and column $i$
    is the evaluated actor. Computes global Societal Status $\\mathcal{S}_i(t)$ via
    power iteration over the row-stochastic trust matrix.
    """

    def __init__(
        self,
        agent_ids: list[str] | tuple[str, ...],
        config: TrustGraphConfig | None = None,
    ) -> None:
        if not agent_ids:
            raise ValueError("VillageTrustGraph requires at least one agent_id.")
        self.config = config or TrustGraphConfig()
        self.agent_ids: tuple[str, ...] = tuple(sorted(set(agent_ids)))
        self._id_to_idx: dict[str, int] = {aid: i for i, aid in enumerate(self.agent_ids)}
        n = len(self.agent_ids)

        # Initialize T[j, i] = initial_trust for j != i, and 0.0 on diagonal (no self-trust inflation)
        self._matrix: np.ndarray = np.full((n, n), self.config.initial_trust, dtype=np.float64)
        np.fill_diagonal(self._matrix, 0.0)

        self._status: np.ndarray = np.full(n, 1.0 / n, dtype=np.float64)
        self.last_iterations: int = 0
        self.last_residual: float = 0.0
        self.compute_societal_status()

    @property
    def num_agents(self) -> int:
        return len(self.agent_ids)

    @property
    def trust_matrix(self) -> np.ndarray:
        """Returns a read-only copy of the $N \\times N$ trust matrix $\\mathbf{T}(t)$."""
        return self._matrix.copy()

    def get_trust(self, observer_id: str, target_id: str) -> float:
        """Returns $T_{\\text{observer} \\to \\text{target}}(t) \\in [0, 1]$."""
        if observer_id == target_id:
            return 1.0
        j = self._id_to_idx[observer_id]
        i = self._id_to_idx[target_id]
        return float(self._matrix[j, i])

    def set_trust(self, observer_id: str, target_id: str, value: float) -> None:
        """Explicitly sets $T_{j \\to i}$ clamped to $[0, 1]$."""
        if observer_id == target_id:
            return
        j = self._id_to_idx[observer_id]
        i = self._id_to_idx[target_id]
        self._matrix[j, i] = float(np.clip(value, 0.0, 1.0))

    def record_interaction(
        self,
        observer_id: str,
        actor_id: str,
        event_type: str,
    ) -> float:
        """Updates $T_{j \\to i}(t+1)$ following an observed interaction (§2.2.1):

        $$T_{j \\to i}(t+1) = \\text{clip}\\left((1 - \\lambda_{\\text{decay}}) T_{j \\to i}(t) + \\Delta T_{j \\to i},\\; 0,\\; 1\\right)$$
        """
        if observer_id == actor_id:
            return 1.0
        j = self._id_to_idx[observer_id]
        i = self._id_to_idx[actor_id]
        t_cur = float(self._matrix[j, i])

        if event_type == "coop":
            delta = self.config.alpha_coop * (1.0 - t_cur)
        elif event_type == "defect":
            delta = -self.config.beta_defect * t_cur
        elif event_type in ("crime_caught", "crime"):
            delta = -self.config.gamma_crime
        else:
            delta = 0.0

        t_next = (1.0 - self.config.lambda_decay) * t_cur + delta
        t_clamped = float(np.clip(t_next, 0.0, 1.0))
        self._matrix[j, i] = t_clamped
        return t_clamped

    def record_public_event(
        self,
        actor_id: str,
        event_type: str,
        witness_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Updates trust toward actor_id across all witnesses (or entire village if public)."""
        observers = witness_ids if witness_ids is not None else self.agent_ids
        for obs_id in observers:
            if obs_id in self._id_to_idx and obs_id != actor_id:
                self.record_interaction(obs_id, actor_id, event_type)

    def propagate_gossip(
        self,
        speaker_id: str,
        listener_id: str,
        subject_id: str,
    ) -> float:
        """Reputation gossip propagation: speaker shares opinion of subject with listener.

        Listener blends speaker's rating weighted by listener's trust in speaker:
        $$T_{L \\to S}^{\\text{new}} = (1 - \\eta) T_{L \\to S} + \\eta T_{S_p \\to S}, \\quad \\eta = w_{\\text{gossip}} \\cdot T_{L \\to S_p}$$
        """
        if listener_id in (speaker_id, subject_id) or speaker_id == subject_id:
            return self.get_trust(listener_id, subject_id)

        trust_in_speaker = self.get_trust(listener_id, speaker_id)
        speaker_opinion = self.get_trust(speaker_id, subject_id)
        listener_current = self.get_trust(listener_id, subject_id)

        eta = self.config.gossip_weight * trust_in_speaker
        blended = float(np.clip((1.0 - eta) * listener_current + eta * speaker_opinion, 0.0, 1.0))
        self.set_trust(listener_id, subject_id, blended)
        return blended

    def build_row_stochastic_matrix(self) -> np.ndarray:
        """Constructs the row-stochastic normalized trust matrix $\\mathbf{\\tilde{T}}$.

        If an agent $j$ has zero total outgoing trust (dangling node), row $j$ is
        replaced by the uniform distribution $\\frac{1}{N}\\mathbf{1}^T$.
        """
        n = self.num_agents
        mat = np.clip(self._matrix.copy(), 0.0, 1.0)
        np.fill_diagonal(mat, 0.0)
        row_sums = mat.sum(axis=1, keepdims=True)
        uniform_row = np.full((1, n), 1.0 / n, dtype=np.float64)
        tilde_t = np.where(row_sums > 1e-12, mat / np.maximum(row_sums, 1e-12), uniform_row)
        return tilde_t

    def compute_societal_status(
        self,
        step_callback: Callable[[PageRankStepEvent], None] | None = None,
    ) -> dict[str, float]:
        """Computes global Societal Status $\\mathcal{S}_i(t)$ via power iteration (§2.2.2):

        $$\\mathbf{s} = \\frac{1 - d}{N} \\mathbf{1} + d \\, \\mathbf{\\tilde{T}}^T \\mathbf{s}$$

        Guarantees $\\mathcal{S}_i \\in [0, 1]$ and $\\sum_{i=1}^N \\mathcal{S}_i = 1$.
        """
        n = self.num_agents
        if n == 1:
            self._status = np.array([1.0], dtype=np.float64)
            self.last_iterations = 1
            self.last_residual = 0.0
            return {self.agent_ids[0]: 1.0}

        tilde_t = self.build_row_stochastic_matrix()
        d = self.config.damping_factor
        teleport = np.full(n, (1.0 - d) / n, dtype=np.float64)
        s = np.full(n, 1.0 / n, dtype=np.float64)

        residual = 1.0
        it = 0
        for it in range(1, self.config.pagerank_max_iter + 1):
            s_next = teleport + d * (tilde_t.T @ s)
            # Normalize to guard against floating-point drift
            s_next = np.maximum(s_next, 0.0)
            s_next /= s_next.sum()
            residual = float(np.sum(np.abs(s_next - s)))
            s = s_next

            if step_callback is not None:
                step_callback(
                    PageRankStepEvent(
                        iteration=it,
                        l1_residual=residual,
                        status_vector={
                            aid: round(float(s[idx]), 6) for idx, aid in enumerate(self.agent_ids)
                        },
                    )
                )

            if residual <= self.config.pagerank_tol:
                break

        self._status = s
        self.last_iterations = it
        self.last_residual = residual
        return {aid: float(s[idx]) for idx, aid in enumerate(self.agent_ids)}

    def get_status(self, agent_id: str) -> float:
        """Returns cached PageRank Societal Status $\\mathcal{S}_i(t)$."""
        idx = self._id_to_idx[agent_id]
        return float(self._status[idx])

    def get_status_map(self) -> dict[str, float]:
        """Returns dictionary mapping `agent_id -> S_i(t)`."""
        return {aid: float(self._status[idx]) for idx, aid in enumerate(self.agent_ids)}

    def to_network_payload(self) -> dict[str, Any]:
        """Serializes the trust graph and status hierarchy for UI visualization."""
        edges: list[dict[str, Any]] = []
        for j, src in enumerate(self.agent_ids):
            for i, dst in enumerate(self.agent_ids):
                if j == i:
                    continue
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "trust": round(float(self._matrix[j, i]), 4),
                    }
                )
        leaderboard = sorted(
            [
                {"agent_id": aid, "status": round(float(self._status[idx]), 5)}
                for idx, aid in enumerate(self.agent_ids)
            ],
            key=lambda x: -float(x["status"]),  # type: ignore[arg-type]
        )
        return {
            "agent_ids": list(self.agent_ids),
            "status_map": {
                aid: round(float(self._status[idx]), 5) for idx, aid in enumerate(self.agent_ids)
            },
            "leaderboard": leaderboard,
            "edges": edges,
            "pagerank_iterations": self.last_iterations,
            "pagerank_residual": self.last_residual,
        }
