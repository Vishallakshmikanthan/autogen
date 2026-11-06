"""Unit and mathematical oracle tests for Circadian Daily Routines and Temporal Priors (§5.4)."""

from __future__ import annotations

from autogen.intelligence.village.routines import (
    DEFAULT_CIRCADIAN_PRIORS,
    circular_hour_distance,
    compute_circadian_priors,
)
from autogen.simulation.environment.village.village_grid import VillageRole


def test_circular_hour_distance_oracle() -> None:
    """Verifies §5 Oracle: Circular hour distance wraps around 24-hour midnight boundary correctly."""
    # Distance between 23.5h and 0.5h is 1.0h, not 23.0h
    assert circular_hour_distance(23.5, 0.5) == 1.0
    assert circular_hour_distance(0.5, 23.5) == 1.0

    # Opposite sides of clock (12h difference)
    assert circular_hour_distance(12.0, 0.0) == 12.0
    assert circular_hour_distance(6.0, 18.0) == 12.0

    # Short daytime difference
    assert circular_hour_distance(14.0, 15.5) == 1.5

    # Near midnight wrap (2.0h to 22.0h -> delta = 4.0h)
    assert circular_hour_distance(2.0, 22.0) == 4.0


def test_circadian_peak_maxima_oracle() -> None:
    """Verifies §5 Oracle: Evaluating circadian prior at mu yields exactly the peak amplitude A."""
    for prior in DEFAULT_CIRCADIAN_PRIORS:
        peak_val = prior.evaluate(prior.peak_hour)
        assert abs(peak_val - prior.amplitude) < 1e-3

        # Values 8 hours away should be substantially dampened
        off_val = prior.evaluate((prior.peak_hour + 8.0) % 24.0)
        assert off_val < (0.25 * prior.amplitude)


def test_role_schedule_customization_oracle() -> None:
    """Verifies §5 Oracle: Occupational roles exhibit tailored diurnal routines."""
    # Guard: night watch peak at 22:00
    guard_night_priors = compute_circadian_priors(VillageRole.GUARD, time_of_day=22.0)
    farmer_night_priors = compute_circadian_priors(VillageRole.FARMER, time_of_day=22.0)

    assert guard_night_priors.get("WORK", 0.0) > 3.0
    assert farmer_night_priors.get("WORK", 0.0) < 0.50

    # Farmer: morning agricultural peak at 08:30
    farmer_morning_priors = compute_circadian_priors(VillageRole.FARMER, time_of_day=8.5)
    assert farmer_morning_priors.get("WORK", 0.0) > 3.0

    # Merchant: market peak at 13:00
    merchant_midday_priors = compute_circadian_priors(VillageRole.MERCHANT, time_of_day=13.0)
    assert merchant_midday_priors.get("MARKET", 0.0) > 3.5
