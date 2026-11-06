"""Sleep Consolidation, Dispositions, and Nightly Relational Reflection (§4.4).

Processes daily episodic experiences into durable, explainable Dispositions:
- tracks net valence, grudges, fondness, confidence, and citing evidence IDs
- executes asymmetric harm weighting, Neuroticism-modulated grudge retention, and forgiveness
- implements situational attribution without ground-truth leakage and deterministic crime pattern detection
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.personality import Traits
from autogen.simulation.environment.village.village_grid import VillageActionType

if TYPE_CHECKING:
    pass


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
    """Executes sleep-time consolidation of memory items into Dispositions (§4.4)."""

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

    @staticmethod
    def should_reflect(
        unreflected: Sequence[Any],
        completed_action: VillageActionType | None = None,
    ) -> tuple[bool, str]:
        """Determines whether a reflection consolidation pass should trigger (§4.4):

        1. REST_HUT completion ("sleep_consolidation")
        2. Accumulated unreflected importance sum $\\ge 2.5$ ("importance_threshold")
        3. An acute critical event with $I \\ge 0.85$ ("acute_event")
        """
        if not unreflected:
            return (False, "none")

        if completed_action == VillageActionType.REST_HUT:
            return (True, "sleep_consolidation")

        total_imp = sum(getattr(m, "importance", 0.5) for m in unreflected)
        if total_imp >= 2.50:
            return (True, "importance_threshold")

        if any(getattr(m, "importance", 0.0) >= 0.85 for m in unreflected):
            return (True, "acute_event")

        return (False, "none")

    @staticmethod
    def compute_situational_blame(
        traits: Traits,
        is_drought: bool = False,
        granary_wheat: float = 100.0,
    ) -> float:
        """Computes situational attribution factor $\\text{blame} \\in [0.5, 1.0]$ using public signals (§4.4):

        $$\\sigma_{pub} = \\mathrm{clip}(\\mathbb{1}[\\text{drought}] \\cdot 0.5 + (1.0 - \\text{wheat}/100) \\cdot 0.5, \\ 0.0, \\ 1.0)$$
        $$\\text{blame} = 1.0 - 0.50 \\cdot A \\cdot \\sigma_{pub}$$
        """
        drought_factor = 0.50 if is_drought else 0.0
        scarcity_factor = max(0.0, min(1.0, (100.0 - granary_wheat) / 100.0)) * 0.50
        sigma_pub = min(1.0, max(0.0, drought_factor + scarcity_factor))
        blame = 1.0 - 0.50 * traits.agreeableness * sigma_pub
        return float(round(blame, 4))

    @classmethod
    def check_forgiveness(
        cls,
        *,
        disposition: Disposition,
        traits: Traits,
        witnessed_community_service: bool,
        steps_since_harm: int,
        rng: np.random.Generator,
    ) -> tuple[Disposition, bool]:
        """Executes a stochastic forgiveness check following visible rehabilitation (§4.4):

        $$P_f = \\mathrm{clip}(0.20 + 0.50 A + 0.30 \\cdot \\text{sinc}, \\ 0.0, \\ 1.0) \\cdot (1 - e^{-\\Delta t / 48})$$
        On success, cuts existing grudge in half.
        """
        if disposition.grudge <= 0.01:
            return disposition, False

        sinc = 1.0 if witnessed_community_service else 0.40
        time_factor = 1.0 - math.exp(-float(max(0, steps_since_harm)) / 48.0)
        p_forgive = float(np.clip(0.20 + 0.50 * traits.agreeableness + 0.30 * sinc, 0.0, 1.0) * time_factor)

        if float(rng.random()) < p_forgive:
            next_g = float(round(disposition.grudge * 0.50, 4))
            updated = disposition.model_copy(update={"grudge": next_g})
            return updated, True
        return disposition, False

    @classmethod
    def reflect_episodes(
        cls,
        *,
        unreflected: list[MemoryEpisode],
        current_dispositions: Mapping[str, Disposition],
        traits: Traits,
        max_evidence_per_target: int = 8,
        is_drought: bool = False,
        granary_wheat: float = 100.0,
    ) -> dict[str, Disposition]:
        """Integrates unreflected episodes into Dispositions (§4.4)."""
        blame = cls.compute_situational_blame(traits, is_drought, granary_wheat)

        # Group episodes by other_id
        grouped: dict[str, list[MemoryEpisode]] = {}
        for ep in unreflected:
            if ep.other_id:
                grouped.setdefault(ep.other_id, []).append(ep)

        updated_dispositions = dict(current_dispositions)

        for other_id, episodes in grouped.items():
            sorted_eps = sorted(episodes, key=lambda e: e.importance, reverse=True)[:max_evidence_per_target]

            total_weight = sum(e.importance for e in sorted_eps)
            if total_weight > 0.0:
                bar_v = sum(e.valence * e.importance for e in sorted_eps) / total_weight
            else:
                bar_v = sum(e.valence for e in sorted_eps) / max(1, len(sorted_eps))

            # Multiply harm by situational blame (Agreeable villagers forgive crisis thieves)
            harm = sum(e.importance * blame for e in sorted_eps if e.is_harm or e.valence < -0.20)
            help_ = sum(e.importance for e in sorted_eps if e.is_help or e.valence > 0.20)
            n_new = len(sorted_eps)

            existing = updated_dispositions.get(other_id) or Disposition(target_id=other_id)
            n_total = existing.evidence_count + n_new

            # 1. Net valence integration (0.85 / 0.15)
            next_v = 0.85 * existing.valence + 0.15 * bar_v

            # 2. Grudge update
            g_delta = 0.60 * harm
            if existing.grudge > 0.0:
                g_delta -= 0.40 * help_
            next_g = float(np.clip(existing.grudge + g_delta, 0.0, 1.0))

            # 3. Fondness update
            f_delta = 0.60 * help_ - 0.20 * harm
            next_f = float(np.clip(existing.fondness + f_delta, 0.0, 1.0))

            # 4. Confidence saturation
            conf = float(1.0 - math.exp(-0.40 * n_total))

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

    @classmethod
    def reflect_memory_items(
        cls,
        *,
        unreflected: Sequence[Any],
        current_dispositions: Mapping[str, Disposition],
        traits: Traits,
        self_id: str,
        max_evidence_per_target: int = 8,
        is_drought: bool = False,
        granary_wheat: float = 100.0,
    ) -> tuple[dict[str, Disposition], list[dict[str, Any]]]:
        """Consolidates structured MemoryItems into Dispositions and reflection syntheses (§4.4)."""
        blame = cls.compute_situational_blame(traits, is_drought, granary_wheat)

        # Group by participating target peer
        grouped: dict[str, list[Any]] = {}
        for m in unreflected:
            for actor in getattr(m, "actors", ()):
                if actor != self_id:
                    grouped.setdefault(actor, []).append(m)

        updated_dispositions = dict(current_dispositions)
        generated_reflections: list[dict[str, Any]] = []

        for other_id, items in sorted(grouped.items()):
            sorted_items = sorted(items, key=lambda x: getattr(x, "importance", 0.5), reverse=True)[
                :max_evidence_per_target
            ]

            total_weight = sum(getattr(m, "importance", 0.5) for m in sorted_items)
            if total_weight > 0.0:
                bar_v = sum(getattr(m, "valence", 0.0) * getattr(m, "importance", 0.5) for m in sorted_items) / total_weight
            else:
                bar_v = sum(getattr(m, "valence", 0.0) for m in sorted_items) / max(1, len(sorted_items))

            harm = sum(
                getattr(m, "importance", 0.5) * blame
                for m in sorted_items
                if "theft" in getattr(m, "tags", ()) or "crime" in getattr(m, "tags", ()) or getattr(m, "valence", 0.0) < -0.20
            )
            help_ = sum(
                getattr(m, "importance", 0.5)
                for m in sorted_items
                if "kindness" in getattr(m, "tags", ()) or "trade" in getattr(m, "tags", ()) or getattr(m, "valence", 0.0) > 0.20
            )
            n_new = len(sorted_items)

            existing = updated_dispositions.get(other_id) or Disposition(target_id=other_id)
            n_total = existing.evidence_count + n_new

            next_v = 0.85 * existing.valence + 0.15 * bar_v

            g_delta = 0.60 * harm
            if existing.grudge > 0.0:
                g_delta -= 0.40 * help_
            next_g = float(np.clip(existing.grudge + g_delta, 0.0, 1.0))

            f_delta = 0.60 * help_ - 0.20 * harm
            next_f = float(np.clip(existing.fondness + f_delta, 0.0, 1.0))

            conf = float(1.0 - math.exp(-0.40 * n_total))

            new_ev_ids = [getattr(m, "mid", "") for m in sorted_items if getattr(m, "mid", "")]
            merged_ids = tuple(list(existing.evidence_ids) + new_ev_ids)[-16:]

            new_disp = Disposition(
                target_id=other_id,
                valence=float(round(np.clip(next_v, -1.0, 1.0), 4)),
                grudge=float(round(next_g, 4)),
                fondness=float(round(next_f, 4)),
                confidence=float(round(np.clip(conf, 0.0, 1.0), 4)),
                evidence_ids=merged_ids,
                evidence_count=n_total,
            )
            updated_dispositions[other_id] = new_disp

            # Synthesize reflection thought memory
            thought = cls.render_thought(new_disp, other_id)
            ref_payload = {
                "kind": "reflection",
                "actors": (self_id, other_id),
                "summary": thought,
                "tags": ("reflection", f"disposition_{other_id}"),
                "valence": new_disp.valence,
                "importance": float(round(0.60 + 0.30 * new_disp.confidence, 4)),
                "evidence_of": new_ev_ids,
            }
            generated_reflections.append(ref_payload)

        return updated_dispositions, generated_reflections

    @staticmethod
    def detect_crime_patterns(
        items: Sequence[Any],
        agent_id: str,
        step: int,
    ) -> list[dict[str, Any]]:
        """Detects recurring spatial/temporal crime clusters (§4.4):

        At $\\ge 3$ crime observations in a specific (location, is_night) bin,
        synthesizes a high-level world belief memory.
        """
        counts: dict[tuple[str, bool], list[str]] = {}
        for m in items:
            tags = getattr(m, "tags", ())
            if "theft" in tags or "crime" in tags or "crime_caught" in tags:
                loc = getattr(m, "location", "village_square") or "village_square"
                hour = (getattr(m, "step", 0) * 0.5) % 24.0
                is_night = hour < 6.0 or hour >= 20.0
                counts.setdefault((loc, is_night), []).append(getattr(m, "mid", f"mem_{step}"))

        beliefs: list[dict[str, Any]] = []
        for (loc, is_night), mids in counts.items():
            if len(mids) >= 3:
                time_str = "night" if is_night else "day"
                summary = f"Thefts cluster near {loc} during {time_str} hours (occurrences: {len(mids)})"
                beliefs.append(
                    {
                        "kind": "belief",
                        "actors": (agent_id,),
                        "location": loc,
                        "summary": summary,
                        "tags": ("crime_pattern", "theft_risk", loc),
                        "importance": float(min(1.0, 0.60 + 0.10 * len(mids))),
                        "evidence_of": tuple(mids),
                    }
                )
        return beliefs

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
