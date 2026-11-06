"""Sleep Consolidation, Dispositions, and Nightly Relational Reflection (§4.4).

Processes daily episodic experiences into durable, explainable Dispositions:
- tracks net valence, grudges, fondness, confidence, and citing evidence IDs
- executes asymmetric harm weighting, Neuroticism-modulated grudge retention, and forgiveness
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.personality import Traits


class Disposition(BaseModel):
    """A durable, explainable relational attitude toward a specific villager (§4.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: str
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Cumulative appraisal in [-1, 1]")
    grudge: float = Field(default=0.0, ge=0.0, le=1.0, description="Durable resentment from past harm")
    fondness: float = Field(default=0.0, ge=0.0, le=1.0, description="Warmth and gratitude from past assistance")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Certainty derived from sample size n")
    evidence_ids: tuple[str, ...] = ()
    evidence_count: int = 0


class MemoryEpisode(BaseModel):
    """A tagged episodic record ingested during reflection (§4.1, §4.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    step: int
    other_id: str
    valence: float = Field(ge=-1.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    is_harm: bool = False
    is_help: bool = False
    description: str = ""


class ReflectionEngine:
    """Executes sleep-time consolidation of memory episodes into Dispositions (§4.4)."""

    @staticmethod
    def compute_grudge_decay(traits: Traits) -> float:
        """Returns per-tick grudge retention factor:

        $$\\text{decay} = \\min(0.999, \\ 0.995 + 0.004 N - 0.003 A)$$
        High Neuroticism persists grudges; high Agreeableness accelerates forgiveness.
        """
        decay = 0.995 + 0.004 * traits.neuroticism - 0.003 * traits.agreeableness
        return float(round(min(0.999, max(0.950, decay)), 5))

    @classmethod
    def decay_grudges(
        cls,
        dispositions: Mapping[str, Disposition],
        traits: Traits,
    ) -> dict[str, Disposition]:
        """Applies per-tick exponential decay to existing grudges."""
        decay = cls.compute_grudge_decay(traits)
        updated: dict[str, Disposition] = {}

        for target_id, d in dispositions.items():
            if d.grudge <= 1e-4:
                updated[target_id] = d
                continue
            next_g = float(round(d.grudge * decay, 4))
            if next_g < 0.005:
                next_g = 0.0
            updated[target_id] = d.model_copy(update={"grudge": next_g})

        return updated

    @classmethod
    def reflect_episodes(
        cls,
        *,
        unreflected: list[MemoryEpisode],
        current_dispositions: Mapping[str, Disposition],
        traits: Traits,
        max_evidence_per_target: int = 8,
    ) -> dict[str, Disposition]:
        """Integrates unreflected episodes into Dispositions using exact §4.4 equations:

        $$v \\leftarrow 0.85 v + 0.15 \\bar{v}$$
        $$g \\leftarrow \\text{clip}(g + 0.6 \\cdot \\text{harm} - 0.4 \\cdot \\text{help} \\cdot \\mathbb{1}[g > 0], 0, 1)$$
        $$\\text{fondness} \\leftarrow \\text{clip}(\\text{fondness} + 0.6 \\cdot \\text{help} - 0.2 \\cdot \\text{harm}, 0, 1)$$
        $$\\text{conf} = 1 - e^{-0.4 n}$$
        """
        # Group episodes by other_id
        grouped: dict[str, list[MemoryEpisode]] = {}
        for ep in unreflected:
            if ep.other_id:
                grouped.setdefault(ep.other_id, []).append(ep)

        updated_dispositions = dict(current_dispositions)

        for other_id, episodes in grouped.items():
            # Top-k by importance
            sorted_eps = sorted(episodes, key=lambda e: e.importance, reverse=True)[:max_evidence_per_target]

            total_weight = sum(e.importance for e in sorted_eps)
            if total_weight > 0.0:
                bar_v = sum(e.valence * e.importance for e in sorted_eps) / total_weight
            else:
                bar_v = sum(e.valence for e in sorted_eps) / max(1, len(sorted_eps))

            harm = sum(e.importance for e in sorted_eps if e.is_harm or e.valence < -0.20)
            help_ = sum(e.importance for e in sorted_eps if e.is_help or e.valence > 0.20)
            n_new = len(sorted_eps)

            existing = updated_dispositions.get(other_id) or Disposition(target_id=other_id)
            n_total = existing.evidence_count + n_new

            # 1. Net valence integration (0.85 / 0.15)
            next_v = 0.85 * existing.valence + 0.15 * bar_v

            # 2. Grudge update (harm weighted higher than help)
            g_delta = 0.60 * harm
            if existing.grudge > 0.0:
                g_delta -= 0.40 * help_
            next_g = float(np.clip(existing.grudge + g_delta, 0.0, 1.0))

            # 3. Fondness update
            f_delta = 0.60 * help_ - 0.20 * harm
            next_f = float(np.clip(existing.fondness + f_delta, 0.0, 1.0))

            # 4. Confidence saturation
            conf = float(1.0 - math.exp(-0.40 * n_total))

            # Merge and cap evidence ids to last 16 to guarantee strictly bounded memory
            new_ev_ids = [e.episode_id for e in sorted_eps if e.episode_id]
            merged_ids = tuple(list(existing.evidence_ids) + new_ev_ids)[-16:]

            updated_dispositions[other_id] = Disposition(
                target_id=other_id,
                valence=float(round(np.clip(next_v, -1.0, 1.0), 4)),
                grudge=float(round(next_g, 4)),
                fondness=float(round(next_f, 4)),
                confidence=float(round(np.clip(conf, 0.0, 1.0), 4)),
                evidence_ids=merged_ids,
                evidence_count=n_total,
            )

        return updated_dispositions

    @staticmethod
    def render_thought(disposition: Disposition, other_name: str) -> str:
        """Renders an explainable thought summary based on dispositions and evidence IDs."""
        if disposition.grudge > 0.50:
            ev_str = ", ".join(disposition.evidence_ids[-3:]) or "recent actions"
            return f"I resent {other_name} for past harms (evidence: {ev_str})."
        if disposition.fondness > 0.50:
            ev_str = ", ".join(disposition.evidence_ids[-3:]) or "past kindness"
            return f"{other_name} has proven generous and dependable (evidence: {ev_str})."
        return f"{other_name} is neutral in my standing."
