"""Intention Layer, Commitment Decay, Interruption Arbitration, and Switching Costs (§2.5).

Provides temporal commitment to goals, arbitration against sudden high-salience Alerts,
and utility modulation ($u_{react}$ and $C_{switch}$) as specified in Module M22-H.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.attention import THETA_STARTLE, Alert
from autogen.intelligence.village.personality import Traits
from autogen.simulation.environment.village.village_grid import VillageActionType
from autogen.simulation.state.types import Position


class Intention(BaseModel):
    """An active behavioral goal with temporal commitment and interruptibility (§2.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = "ROUTINE"  # "ROUTINE", "SHELTER", "INVESTIGATE_NOISE", "GREET", "WORK_FIELD", "REST_HUT", "FLEE"
    goal_zone: str = ""
    target_id: str | None = None
    target_pos: Position | None = None
    started_step: int = 0
    commitment: float = Field(default=1.0, ge=0.0, le=1.0, description="Commitment level decaying over time")
    interruptible: bool = True
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Completion fraction in [0, 1]")

    def with_progress(self, progress_delta: float) -> Intention:
        """Returns updated Intention with incremented progress."""
        next_p = float(np.clip(self.progress + progress_delta, 0.0, 1.0))
        return self.model_copy(update={"progress": next_p})

    def with_decayed_commitment(self, factor: float = 0.98) -> Intention:
        """Decays commitment over time if uncompleted."""
        next_c = float(np.clip(self.commitment * factor, 0.05, 1.0))
        return self.model_copy(update={"commitment": next_c})


HYSTERESIS: float = 0.15
LAMBDA_ATT: float = 4.0
KAPPA_SW: float = 3.0


def should_interrupt(
    alert: Alert,
    intention: Intention,
    conscientiousness: float,
) -> bool:
    """Evaluates whether an alert possesses sufficient urgency to preempt the current intention (§2.5):

    $$\\text{urgency} = S \\cdot (1 + \\text{threat})$$
    $$\\text{stickiness} = c \\cdot (1 + 0.4 C) + 0.3 \\cdot \\mathbb{1}[\\text{progress} > 0.8]$$
    $$\\text{Interrupt if: } \\text{urgency} > \\text{stickiness} + 0.15$$
    """
    if not intention.interruptible and alert.salience < THETA_STARTLE:
        return False

    urgency = alert.salience * (1.0 + alert.threat)
    stickiness = intention.commitment * (1.0 + 0.40 * conscientiousness)
    if intention.progress > 0.80:
        stickiness += 0.30

    return urgency > (stickiness + HYSTERESIS)


def action_serves_intention(action: VillageActionType, intention: Intention) -> bool:
    """Determines whether a candidate action directly advances the active intention."""
    k = intention.kind.upper()

    if k in ("ROUTINE", "IDLE"):
        return True

    if k == "SHELTER":
        return action in (
            VillageActionType.SHELTER,
            VillageActionType.REST_HUT,
            VillageActionType.STAY,
            VillageActionType.MOVE_NORTH,
            VillageActionType.MOVE_SOUTH,
            VillageActionType.MOVE_EAST,
            VillageActionType.MOVE_WEST,
        )

    if k == "INVESTIGATE_NOISE":
        return action in (
            VillageActionType.INVESTIGATE_NOISE,
            VillageActionType.MOVE_NORTH,
            VillageActionType.MOVE_SOUTH,
            VillageActionType.MOVE_EAST,
            VillageActionType.MOVE_WEST,
            VillageActionType.STAY,
        )

    if k == "GREET":
        return action in (
            VillageActionType.GREET,
            VillageActionType.STAY,
        )

    if k in ("WORK_FIELD", "HARVEST"):
        return action in (
            VillageActionType.HARVEST_FIELD,
            VillageActionType.TITHE_GRANARY,
            VillageActionType.MOVE_NORTH,
            VillageActionType.MOVE_SOUTH,
            VillageActionType.MOVE_EAST,
            VillageActionType.MOVE_WEST,
            VillageActionType.STAY,
        )

    if k in ("REST_HUT", "REST", "SLEEP"):
        return action in (
            VillageActionType.REST_HUT,
            VillageActionType.STAY,
        )

    return True


def compute_reaction_utility(
    action: VillageActionType,
    alerts: list[Alert],
    traits: Traits,
    affinity_to_source: float = 0.50,
) -> float:
    """Computes $u_{react}(a | \\text{alerts})$ bonus for reactive actions addressing alerts (§2.5):

    - SHELTER responds to THUNDER / weather threats (gain boosted by Neuroticism)
    - INVESTIGATE_NOISE responds to cries / auditory alerts (gain boosted by Openness)
    - GREET responds to ARRIVAL / social proximity (gain boosted by Extraversion & affinity)
    """
    if not alerts:
        return 0.0

    bonus = 0.0
    for alert in alerts:
        ev = alert.percept.event
        s = alert.salience

        if action == VillageActionType.SHELTER:
            if ev.kind in ("THUNDER", "RAIN_ONSET") or alert.threat > 0.40:
                bonus += s * (1.0 + 0.50 * traits.neuroticism)

        elif action == VillageActionType.INVESTIGATE_NOISE:
            if ev.modality == ev.modality.AUDITORY or ev.kind in ("CRY_OUT", "CHASE"):
                bonus += s * (1.0 + 0.50 * traits.openness)

        elif action == VillageActionType.GREET:
            if ev.kind == "ARRIVAL" or ev.source_id is not None:
                bonus += s * (0.50 + 0.50 * affinity_to_source) * (1.0 + 0.50 * traits.extraversion)

    return float(round(bonus, 4))


def compute_switching_cost(
    action: VillageActionType,
    intention: Intention,
    switching_cost_scale: float,
    kappa_sw: float = KAPPA_SW,
) -> float:
    """Calculates switching penalty $C_{switch} = \\kappa_{sw} \\cdot c \\cdot \\text{scale} \\cdot \\mathbb{1}[a \\notin I]$ (§2.5)."""
    if action_serves_intention(action, intention):
        return 0.0

    cost = kappa_sw * intention.commitment * switching_cost_scale
    return float(round(cost, 4))
