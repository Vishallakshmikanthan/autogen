"""Directed Multi-Dimensional Relationship Store for AUTOGEN Villagers (§3.5).

Tracks directed dyadic relationships across three orthogonal dimensions:
- affinity: liking / emotional warmth in [0, 1]
- respect: competence / standing appraisal in [0, 1]
- familiarity: cumulative interaction frequency in [0, 1]
(Note: 'trust' is maintained in VillageTrustGraph for institutional integration).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Relationship(BaseModel):
    """Directed relational state from agent_i toward agent_j (§3.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    affinity: float = Field(default=0.50, ge=0.0, le=1.0, description="Interpersonal warmth and liking")
    respect: float = Field(default=0.50, ge=0.0, le=1.0, description="Appraisal of competence and authority")
    familiarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Frequency and depth of prior contact")

    def with_interaction(self, quality: float, eta: float = 0.15) -> Relationship:
        """Saturating relational update following an interaction (§3.5):

        affinity <- affinity + eta * quality * (1 - affinity) if quality > 0 else affinity + eta * quality * affinity
        familiarity <- min(1.0, familiarity + 0.05)
        """
        aff = self.affinity
        if quality >= 0.0:
            aff_next = aff + eta * quality * (1.0 - aff)
        else:
            aff_next = aff + eta * quality * aff

        fam_next = min(1.0, self.familiarity + 0.05)
        return self.model_copy(
            update={
                "affinity": float(round(np.clip(aff_next, 0.0, 1.0), 4)),
                "familiarity": float(round(fam_next, 4)),
            }
        )

    def with_decay(self, factor: float = 0.999) -> Relationship:
        """Slow temporal decay of familiarity when out of contact."""
        fam_next = max(0.0, self.familiarity * factor)
        return self.model_copy(update={"familiarity": float(round(fam_next, 4))})


class VillageRelationshipStore:
    """Manages directed dyadic relationships among all active villagers."""

    def __init__(self) -> None:
        self._relations: dict[tuple[str, str], Relationship] = {}

    def get_relationship(self, source_id: str, target_id: str) -> Relationship:
        """Returns directed relationship (source_id -> target_id), defaulting to neutral."""
        if source_id == target_id:
            return Relationship(affinity=1.0, respect=1.0, familiarity=1.0)
        return self._relations.get((source_id, target_id), Relationship())

    def get_affinity(self, source_id: str, target_id: str) -> float:
        """Returns directed affinity scalar in [0, 1]."""
        return self.get_relationship(source_id, target_id).affinity

    def record_interaction(self, source_id: str, target_id: str, quality: float) -> None:
        """Updates directed relationship following a shared event or greeting."""
        if source_id == target_id:
            return
        rel = self.get_relationship(source_id, target_id)
        self._relations[(source_id, target_id)] = rel.with_interaction(quality)

    def record_dyadic_interaction(self, agent_a: str, agent_b: str, quality: float) -> None:
        """Updates both directed edges (A->B and B->A) synchronously."""
        self.record_interaction(agent_a, agent_b, quality)
        self.record_interaction(agent_b, agent_a, quality)

    def step_decay(self) -> None:
        """Decays familiarity across all relationships once per tick."""
        for pair, rel in list(self._relations.items()):
            self._relations[pair] = rel.with_decay()

    def get_all_for_agent(self, agent_id: str) -> dict[str, dict[str, float]]:
        """Returns serialized relationship dictionary for telemetry and inspection (§7)."""
        res: dict[str, dict[str, float]] = {}
        for (src, tgt), rel in self._relations.items():
            if src == agent_id:
                res[tgt] = {
                    "affinity": rel.affinity,
                    "respect": rel.respect,
                    "familiarity": rel.familiarity,
                }
        return res

