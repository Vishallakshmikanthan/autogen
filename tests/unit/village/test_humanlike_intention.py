"""Unit and mathematical oracle tests for Intention Arbitration and Switching Costs (§2.5, §5)."""

from __future__ import annotations

from autogen.intelligence.village.attention import Alert
from autogen.intelligence.village.intention import (
    Intention,
    compute_reaction_utility,
    compute_switching_cost,
    should_interrupt,
)
from autogen.intelligence.village.personality import Traits
from autogen.simulation.environment.village.events import (
    Modality,
    PerceivedEvent,
    WorldEvent,
)
from autogen.simulation.environment.village.village_grid import VillageActionType
from autogen.simulation.state.types import Position


def test_interruption_stickiness_and_conscientiousness() -> None:
    """Invariant: High Conscientiousness and high progress make an intention significantly harder to interrupt."""
    ev = WorldEvent(
        event_id="ev_cry",
        step=1,
        kind="CRY_OUT",
        modality=Modality.AUDITORY,
        position=Position(x=4, y=4),
        intensity=0.8,
        valence_hint=-0.4,
    )
    percept = PerceivedEvent(event=ev, distance=4.0)
    alert = Alert(percept=percept, salience=0.85, threat=0.40, eligible_step=1)

    low_c_traits = Traits(conscientiousness=0.20)
    high_c_traits = Traits(conscientiousness=0.95)

    intention_low_prog = Intention(kind="WORK_FIELD", commitment=0.70, progress=0.20)
    intention_high_prog = Intention(kind="WORK_FIELD", commitment=0.70, progress=0.85)

    # Low-C agent with low progress gets interrupted by 0.85 salience
    assert should_interrupt(alert, intention_low_prog, low_c_traits.conscientiousness) is True

    # High-C agent with high progress (>0.80) protects task from interruption
    assert should_interrupt(alert, intention_high_prog, high_c_traits.conscientiousness) is False


def test_no_oscillation_under_alternating_stimuli_bound() -> None:
    """Invariant: Intention switches over 20 consecutive ticks <= 4 due to hysteresis (§5)."""
    current_intention = Intention(kind="WORK_FIELD", commitment=0.90, progress=0.10)
    traits = Traits(conscientiousness=0.70)
    switch_count = 0

    # Simulate 20 ticks of fluctuating borderline stimuli (salience 0.60 vs 0.20)
    for t in range(20):
        intensity = 0.65 if (t % 2 == 0) else 0.25
        ev = WorldEvent(
            event_id=f"ev_{t}",
            step=t,
            kind="CRY_OUT" if (t % 2 == 0) else "GREETING",
            modality=Modality.AUDITORY,
            position=Position(x=2, y=2),
            intensity=intensity,
            valence_hint=-0.3 if (t % 2 == 0) else 0.2,
        )
        alert = Alert(
            percept=PerceivedEvent(event=ev, distance=2.0),
            salience=0.60 if (t % 2 == 0) else 0.20,
            threat=0.3 if (t % 2 == 0) else 0.0,
            eligible_step=t,
        )

        if should_interrupt(alert, current_intention, traits.conscientiousness):
            current_intention = Intention(kind="INVESTIGATE_NOISE", started_step=t, commitment=0.85)
            switch_count += 1
        else:
            current_intention = current_intention.with_decayed_commitment(0.98)

    assert switch_count <= 4


def test_reaction_utility_and_switching_cost_modulation() -> None:
    """Verifies that u_react awards bonuses for matching actions, and off-intention actions pay C_switch."""
    ev = WorldEvent(
        event_id="ev_thunder",
        step=1,
        kind="THUNDER",
        modality=Modality.ENVIRONMENTAL,
        position=Position(x=0, y=0),
        intensity=0.95,
        valence_hint=-0.80,
        is_global=True,
    )
    alert = Alert(percept=PerceivedEvent(event=ev, distance=0.0), salience=1.5, threat=0.8, eligible_step=1)

    traits = Traits(neuroticism=0.7)
    u_react_shelter = compute_reaction_utility(VillageActionType.SHELTER, [alert], traits)
    u_react_harvest = compute_reaction_utility(VillageActionType.HARVEST_FIELD, [alert], traits)

    assert u_react_shelter > 1.0
    assert u_react_harvest == 0.0

    # Switching cost
    shelter_intention = Intention(kind="SHELTER", commitment=0.90)
    cost_shelter = compute_switching_cost(VillageActionType.SHELTER, shelter_intention, switching_cost_scale=1.2)
    cost_harvest = compute_switching_cost(VillageActionType.HARVEST_FIELD, shelter_intention, switching_cost_scale=1.2)

    # In-intention action pays 0 switching cost
    assert cost_shelter == 0.0
    # Off-intention action incurs positive switching cost
    assert cost_harvest > 2.0
