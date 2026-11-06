"""Salience Scoring, Habituation, and Limited Attention Budget Engine (§2.3, §2.4).

Filters raw sensory percepts into high-priority Alerts and peripheral Notices
using multi-modal salience scoring, habituation decay, and personality gains.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.personality import Traits
from autogen.simulation.environment.village.events import Modality, PerceivedEvent


class Alert(BaseModel):
    """A high-salience sensory alert capable of interrupting ongoing intentions (§2.4, §2.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    percept: PerceivedEvent
    salience: float = Field(ge=0.0)
    threat: float = Field(default=0.0, ge=0.0, le=1.0, description="Threat magnitude in [0, 1]")
    eligible_step: int = Field(ge=0, description="Step at which alert becomes actionable (latency model)")
    is_startle: bool = False


THETA_NOTICE: float = 0.25
THETA_ALERT: float = 0.55
THETA_STARTLE: float = 1.10
ATTN_BUDGET: int = 2
CHARACTERISTIC_LENGTH_TILES: float = 6.0


class HabituationTracker:
    """Tracks per-event-kind habituation $h_{kind} \\in [0, 1]$ to suppress repetitive stimuli (§2.3)."""

    def __init__(self, repeat_increment: float = 0.30, decay_rate: float = 0.90) -> None:
        self.repeat_increment = repeat_increment
        self.decay_rate = decay_rate
        self.habituation: dict[str, float] = {}

    def get_habituation(self, kind: str) -> float:
        """Returns current habituation level $h_{kind} \\in [0, 1]$."""
        return float(self.habituation.get(kind, 0.0))

    def update_tick(self, observed_kinds: set[str]) -> None:
        """Updates habituation: increases on repeats, decays for unobserved stimuli."""
        all_kinds = set(self.habituation.keys()).union(observed_kinds)
        for kind in all_kinds:
            current_h = self.habituation.get(kind, 0.0)
            if kind in observed_kinds:
                # h <- h + 0.3 * (1 - h)
                next_h = current_h + self.repeat_increment * (1.0 - current_h)
            else:
                # h <- 0.9 * h
                next_h = current_h * self.decay_rate

            if next_h < 1e-4:
                self.habituation.pop(kind, None)
            else:
                self.habituation[kind] = float(np.clip(next_h, 0.0, 1.0))


def compute_modality_gain(modality: Modality) -> float:
    """Returns modality gain $g_{mod}$ (§2.3)."""
    if modality == Modality.AUDITORY:
        return 1.0
    if modality == Modality.VISUAL:
        return 0.80
    if modality == Modality.ENVIRONMENTAL:
        return 1.20
    if modality == Modality.SOCIAL:
        return 1.0
    return 1.0


def compute_salience(
    percept: PerceivedEvent,
    traits: Traits,
    arousal: float,
    habituation: float,
    relevance: float = 0.0,
) -> float:
    """Computes exact mathematical salience $S_i(p)$ (§2.3):

    $$S_i(p) = g_{\\text{mod}} \\cdot I(p) \\cdot \\text{prox}(p) \\cdot (1 - h_{\\text{kind}}) \\cdot (1 + R_i) \\cdot (1 + A_i^+) \\cdot G_{\\text{trait}}$$
    """
    ev = percept.event
    g_mod = compute_modality_gain(ev.modality)
    intensity = ev.intensity

    # Proximity: thunder and global events use prox = 1.0
    if ev.is_global or percept.distance <= 1e-6:
        prox = 1.0
    else:
        prox = float(math.exp(-percept.distance / CHARACTERISTIC_LENGTH_TILES))

    # Novelty (1 - h_kind)
    novelty = float(np.clip(1.0 - habituation, 0.0, 1.0))
    if novelty <= 1e-9:
        return 0.0

    # Relevance boost (1 + R_i)
    rel_factor = 1.0 + float(max(0.0, relevance))

    # Arousal boost (1 + A_i^+)
    a_plus = float(np.clip(arousal, 0.0, 1.0))
    arousal_factor = 1.0 + a_plus

    # Personality trait gain G_trait
    if ev.valence_hint < -0.20:
        # Threat stimulus -> scaled by Neuroticism
        g_trait = 1.0 + 0.50 * traits.neuroticism
    elif ev.kind == "CRY_OUT":
        # Distress call -> scaled by Agreeableness (empathy)
        g_trait = 1.0 + 0.50 * traits.agreeableness
    else:
        # Novel non-threat stimulus -> scaled by Openness
        g_trait = 1.0 + 0.50 * traits.openness

    salience = g_mod * intensity * prox * novelty * rel_factor * arousal_factor * g_trait
    return float(round(salience, 4))


class SalienceAttentionEngine:
    """Evaluates sensory percepts against attention thresholds and budget (§2.3, §2.4)."""

    def __init__(self, habituation_tracker: HabituationTracker | None = None) -> None:
        self.tracker = habituation_tracker or HabituationTracker()

    def process_percepts(
        self,
        *,
        percepts: tuple[PerceivedEvent, ...],
        traits: Traits,
        arousal: float,
        step: int,
        relevance_map: dict[str, float] | None = None,
    ) -> tuple[list[Alert], list[PerceivedEvent], dict[str, float]]:
        """Scores percepts, applies attention budget, updates habituation, and generates Alerts and Notices."""
        if not percepts:
            self.tracker.update_tick(set())
            return [], [], {}

        rel_map = relevance_map or {}
        scored: list[tuple[float, PerceivedEvent]] = []
        salience_by_id: dict[str, float] = {}

        for p in percepts:
            kind = p.event.kind
            h = self.tracker.get_habituation(kind)
            r = rel_map.get(p.event.event_id, 0.0)
            if p.event.source_id and p.event.source_id in rel_map:
                r += rel_map[p.event.source_id]

            s = compute_salience(p, traits, arousal, h, relevance=r)
            scored.append((s, p))
            salience_by_id[p.event.event_id] = s

        # Sort descending by salience
        scored.sort(key=lambda t: t[0], reverse=True)

        alerts: list[Alert] = []
        notices: list[PerceivedEvent] = []
        observed_kinds: set[str] = set()

        for s, p in scored[:ATTN_BUDGET]:
            observed_kinds.add(p.event.kind)
            if s >= THETA_ALERT:
                is_startle = s >= THETA_STARTLE
                # Startle has latency 0 (immediate), standard alert has latency 1 tick
                latency = 0 if is_startle else 1
                threat = float(np.clip(-p.event.valence_hint, 0.0, 1.0))
                alerts.append(
                    Alert(
                        percept=p,
                        salience=s,
                        threat=threat,
                        eligible_step=step + latency,
                        is_startle=is_startle,
                    )
                )
            elif s >= THETA_NOTICE:
                notices.append(p)

        # Update habituation for the tick
        self.tracker.update_tick(observed_kinds)

        return alerts, notices, salience_by_id
