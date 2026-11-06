"""Unit and mathematical oracle tests for Multi-Modal Salience Attention Filtering (§2.3, §5)."""

from __future__ import annotations

from autogen.intelligence.village.attention import (
    ATTN_BUDGET,
    THETA_ALERT,
    THETA_STARTLE,
    HabituationTracker,
    SalienceAttentionEngine,
    compute_salience,
)
from autogen.intelligence.village.personality import Traits
from autogen.simulation.environment.village.events import (
    Modality,
    PerceivedEvent,
    WorldEvent,
)
from autogen.simulation.state.types import Position


def test_thunder_walkthrough_mathematical_oracle() -> None:
    """Verifies §2.6 Thunder Walkthrough exact hand-computed mathematical oracle:

    Farmer (N=0.7, Arousal=0.10):
    S = 1.2 * 0.9 * 1.0 * (1 - 0) * (1 + 0) * (1 + 0.10) * (1 + 0.35) = 1.6038 > THETA_STARTLE (1.1)
    """
    thunder_event = WorldEvent(
        event_id="ev_thunder_1",
        step=1,
        kind="THUNDER",
        modality=Modality.ENVIRONMENTAL,
        position=Position(x=0, y=0),
        intensity=0.90,
        valence_hint=-0.80,
        is_global=True,
    )
    perceived_thunder = PerceivedEvent(
        event=thunder_event,
        distance=0.0,
        audible=True,
        visible=True,
    )

    farmer_traits = Traits(neuroticism=0.70)
    salience = compute_salience(
        percept=perceived_thunder,
        traits=farmer_traits,
        arousal=0.10,
        habituation=0.0,
        relevance=0.0,
    )

    expected = 1.2 * 0.90 * 1.0 * 1.0 * 1.0 * 1.10 * (1.0 + 0.5 * 0.70)
    assert abs(salience - round(expected, 4)) < 1e-4
    assert salience > THETA_STARTLE


def test_zero_novelty_yields_zero_salience() -> None:
    """Invariant: Complete habituation (h=1.0) strictly forces salience to zero."""
    event = WorldEvent(
        event_id="ev_loud_noise",
        step=1,
        kind="CRY_OUT",
        modality=Modality.AUDITORY,
        position=Position(x=2, y=2),
        intensity=1.0,
        valence_hint=-0.5,
    )
    percept = PerceivedEvent(event=event, distance=2.0)
    traits = Traits()

    salience = compute_salience(percept, traits, arousal=0.5, habituation=1.0)
    assert salience == 0.0


def test_attention_budget_is_never_exceeded() -> None:
    """Invariant: Attention engine produces at most ATTN_BUDGET (2) alerts even with 10 high-intensity events."""
    engine = SalienceAttentionEngine()
    percepts: list[PerceivedEvent] = []

    for k in range(10):
        ev = WorldEvent(
            event_id=f"ev_alarm_{k}",
            step=1,
            kind="CRY_OUT",
            modality=Modality.AUDITORY,
            position=Position(x=1, y=1),
            intensity=0.95,
            valence_hint=-0.7,
        )
        percepts.append(PerceivedEvent(event=ev, distance=1.0))

    alerts, notices, _ = engine.process_percepts(
        percepts=tuple(percepts),
        traits=Traits(neuroticism=0.8),
        arousal=0.5,
        step=1,
    )

    assert len(alerts) <= ATTN_BUDGET
    assert len(alerts) == 2
    assert all(a.salience >= THETA_ALERT for a in alerts)


def test_habituation_update_and_decay_dynamics() -> None:
    """Verifies habituation increases upon repeat and decays exponentially on absence (§2.3)."""
    tracker = HabituationTracker()
    assert tracker.get_habituation("THUNDER") == 0.0

    # Repeat 1: h = 0 + 0.3 * (1 - 0) = 0.30
    tracker.update_tick({"THUNDER"})
    assert abs(tracker.get_habituation("THUNDER") - 0.30) < 1e-4

    # Repeat 2: h = 0.30 + 0.3 * (1 - 0.30) = 0.51
    tracker.update_tick({"THUNDER"})
    assert abs(tracker.get_habituation("THUNDER") - 0.51) < 1e-4

    # Absence tick: h = 0.51 * 0.9 = 0.459
    tracker.update_tick(set())
    assert abs(tracker.get_habituation("THUNDER") - 0.459) < 1e-4
