"""Circadian Daily Routines and Role-Specific Time-of-Day Priors (§5.4).

Provides smooth Gaussian circular priors $\\pi_I(\\tau)$ biasing villager intention
arbitration according to the 24-hour day/night cycle and occupational roles.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.environment.village.village_grid import VillageRole


def circular_hour_distance(tau1: float, tau2: float) -> float:
    """Computes circular shortest distance between two 24-hour clock times $\\delta \\in [0, 12]$."""
    t1 = tau1 % 24.0
    t2 = tau2 % 24.0
    diff = abs(t1 - t2)
    return float(min(diff, 24.0 - diff))


class CircadianPrior(BaseModel):
    """Gaussian temporal prior $\\pi_I(\\tau) = A \\cdot \\exp(-\\delta^2 / (2\\sigma^2))$ for an intention ($5.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intention_kind: str
    peak_hour: float = Field(ge=0.0, lt=24.0, description="Peak clock hour mu in [0, 24)")
    sigma: float = Field(gt=0.0, description="Standard deviation in hours")
    amplitude: float = Field(default=2.5, ge=0.0, description="Peak utility boost amplitude A")

    def evaluate(self, time_of_day: float) -> float:
        """Evaluates circadian prior score at current simulation clock hour."""
        delta = circular_hour_distance(time_of_day, self.peak_hour)
        score = self.amplitude * math.exp(-(delta**2) / (2.0 * (self.sigma**2)))
        return float(round(score, 4))


# Standard baseline circadian priors (§5.4 table)
DEFAULT_CIRCADIAN_PRIORS: tuple[CircadianPrior, ...] = (
    CircadianPrior(intention_kind="WORK", peak_hour=9.0, sigma=2.5, amplitude=2.50),
    CircadianPrior(intention_kind="MARKET", peak_hour=13.0, sigma=1.5, amplitude=2.20),
    CircadianPrior(intention_kind="SOCIALIZE", peak_hour=18.0, sigma=1.5, amplitude=2.00),
    CircadianPrior(intention_kind="REST", peak_hour=1.0, sigma=3.0, amplitude=3.00),
)

# Role-specific tailored circadian schedules
ROLE_CIRCADIAN_SCHEDULES: dict[VillageRole, tuple[CircadianPrior, ...]] = {
    VillageRole.FARMER: (
        CircadianPrior(intention_kind="WORK", peak_hour=8.5, sigma=2.5, amplitude=3.20),
        CircadianPrior(intention_kind="MARKET", peak_hour=13.5, sigma=1.5, amplitude=2.00),
        CircadianPrior(intention_kind="SOCIALIZE", peak_hour=18.0, sigma=1.5, amplitude=2.00),
        CircadianPrior(intention_kind="REST", peak_hour=1.0, sigma=3.0, amplitude=3.20),
    ),
    VillageRole.MERCHANT: (
        CircadianPrior(intention_kind="WORK", peak_hour=10.0, sigma=2.0, amplitude=2.00),
        CircadianPrior(intention_kind="MARKET", peak_hour=13.0, sigma=2.0, amplitude=3.80),
        CircadianPrior(intention_kind="SOCIALIZE", peak_hour=18.0, sigma=2.0, amplitude=2.50),
        CircadianPrior(intention_kind="REST", peak_hour=1.0, sigma=3.0, amplitude=2.80),
    ),
    VillageRole.GUARD: (
        CircadianPrior(intention_kind="WORK", peak_hour=22.0, sigma=3.5, amplitude=3.50),  # Night watch
        CircadianPrior(intention_kind="WORK", peak_hour=10.0, sigma=2.0, amplitude=2.20),  # Morning inspection
        CircadianPrior(intention_kind="SOCIALIZE", peak_hour=16.0, sigma=1.5, amplitude=2.00),
        CircadianPrior(intention_kind="REST", peak_hour=5.0, sigma=2.5, amplitude=3.20),  # Day sleep
    ),
    VillageRole.WATER_CARRIER: (
        CircadianPrior(intention_kind="WORK", peak_hour=8.0, sigma=2.0, amplitude=3.50),  # Morning well rush
        CircadianPrior(intention_kind="WORK", peak_hour=15.0, sigma=2.0, amplitude=2.80),  # Afternoon distribution
        CircadianPrior(intention_kind="SOCIALIZE", peak_hour=18.0, sigma=2.0, amplitude=2.60),
        CircadianPrior(intention_kind="REST", peak_hour=1.0, sigma=3.0, amplitude=3.00),
    ),
    VillageRole.WANDERER: (
        CircadianPrior(intention_kind="WORK", peak_hour=10.5, sigma=3.0, amplitude=2.40),  # Forest foraging
        CircadianPrior(intention_kind="MARKET", peak_hour=14.0, sigma=2.0, amplitude=2.20),
        CircadianPrior(intention_kind="SOCIALIZE", peak_hour=18.5, sigma=2.0, amplitude=2.80),  # Bardic songs
        CircadianPrior(intention_kind="REST", peak_hour=1.5, sigma=3.5, amplitude=2.60),
    ),
}


def compute_circadian_priors(
    role: VillageRole,
    time_of_day: float,
) -> dict[str, float]:
    """Computes the mapping of intention_kind -> circadian score boost for a given role and hour (§5.4)."""
    priors = ROLE_CIRCADIAN_SCHEDULES.get(role, DEFAULT_CIRCADIAN_PRIORS)
    scores: dict[str, float] = {}
    for p in priors:
        current_score = scores.get(p.intention_kind, 0.0)
        eval_score = p.evaluate(time_of_day)
        # If multiple peaks exist for an intention (e.g. Guard shifts), take the maximal prior
        scores[p.intention_kind] = max(current_score, eval_score)
    return scores
