"""Unit and mathematical oracle tests for Big Five Personality Mapping (§5, §7)."""

from __future__ import annotations

from autogen.intelligence.village.personality import (
    ARCHETYPE_TRAIT_CATALOG,
    CANONICAL_AGENT_TRAITS,
    Traits,
    pad_baseline_from,
    weights_from_traits,
)
from autogen.simulation.environment.village.village_grid import VillageRole


def test_personality_weight_monotonicity_oracle() -> None:
    """Invariant: Each derived weight is strictly monotonic in its governing trait dimension (§5)."""
    # 1. w_soc must strictly increase with Extraversion
    prev_w_soc = -1.0
    for e in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        t = Traits(extraversion=e)
        w, _ = weights_from_traits(t, VillageRole.FARMER)
        assert w.w_soc > prev_w_soc
        prev_w_soc = w.w_soc

    # 2. w_econ must strictly increase with Conscientiousness
    prev_w_econ = -1.0
    for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        t = Traits(conscientiousness=c)
        w, _ = weights_from_traits(t, VillageRole.FARMER)
        assert w.w_econ > prev_w_econ
        prev_w_econ = w.w_econ

    # 3. kappa must strictly increase with Neuroticism
    prev_kappa = -1.0
    for n in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        t = Traits(neuroticism=n)
        w, _ = weights_from_traits(t, VillageRole.FARMER)
        assert w.crisis_penalty_kappa > prev_kappa
        prev_kappa = w.crisis_penalty_kappa


def test_risk_tolerance_boundary_oracle() -> None:
    """Invariant: theta_risk is strictly bounded in [0.02, 0.90] across extreme corners of [0, 1]^5."""
    for o in [0.0, 1.0]:
        for c in [0.0, 1.0]:
            for a in [0.0, 1.0]:
                for n in [0.0, 1.0]:
                    t = Traits(openness=o, conscientiousness=c, agreeableness=a, neuroticism=n)
                    w, params = weights_from_traits(t)
                    assert 0.02 <= w.theta_risk <= 0.90
                    assert 0.01 <= params.contagion_gamma <= 0.50
                    assert 0.90 <= params.grudge_decay <= 0.9999


def test_pad_baseline_bounds_and_mehrabian_direction() -> None:
    """Verifies Mehrabian PAD baseline: Pleasure drops with N, rises with E."""
    low_n = Traits(extraversion=0.5, neuroticism=0.1)
    high_n = Traits(extraversion=0.5, neuroticism=0.9)

    p_low_n = pad_baseline_from(low_n)
    p_high_n = pad_baseline_from(high_n)

    # Lower neuroticism => higher baseline pleasure
    assert p_low_n[0] > p_high_n[0]
    # Higher neuroticism => higher baseline arousal
    assert p_high_n[1] > p_low_n[1]

    # All components bounded in [-1, 1]
    for p in (p_low_n, p_high_n):
        assert all(-1.0 <= v <= 1.0 for v in p)


def test_canonical_archetype_catalog_validity() -> None:
    """Verifies all 10 canonical archetypes exist with valid trait values."""
    assert len(ARCHETYPE_TRAIT_CATALOG) == 10
    assert "Diligent Farmer" in ARCHETYPE_TRAIT_CATALOG
    assert "Anxious Farmer" in ARCHETYPE_TRAIT_CATALOG
    assert "Gossipy Water-Carrier" in ARCHETYPE_TRAIT_CATALOG

    for _name, traits in ARCHETYPE_TRAIT_CATALOG.items():
        assert 0.0 <= traits.openness <= 1.0
        assert 0.0 <= traits.conscientiousness <= 1.0
        assert 0.0 <= traits.extraversion <= 1.0
        assert 0.0 <= traits.agreeableness <= 1.0
        assert 0.0 <= traits.neuroticism <= 1.0

    # Ensure canonical agent mapping covers 10 villagers
    assert len(CANONICAL_AGENT_TRAITS) == 10
    assert "villager_arun" in CANONICAL_AGENT_TRAITS
