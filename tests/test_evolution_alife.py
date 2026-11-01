"""Tests for Artificial Life dynamics, demographic conservation identity, and lineage tracking (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.evolution.alife import (
    EcosystemEngine,
    LineageTracker,
    OrganismTraits,
)


def test_demographic_conservation_identity() -> None:
    """Verifies invariant: N_{t+1} == N_t + B_t - D_t at every single simulation step."""
    engine = EcosystemEngine(
        initial_population=30,
        carrying_capacity=50,
        food_growth_rate=25.0,
        max_food=300.0,
        seed=101,
    )

    for step in range(30):
        _res = engine.step()
        record = engine.demographic_history[-1]
        assert record["n_end"] == record["n_start"] + record["births"] - record["deaths"], (
            f"Demographic violation at step {step}: "
            f"n_end={record['n_end']} != {record['n_start']} + {record['births']} - {record['deaths']}"
        )


def test_metabolic_tradeoff_invariants() -> None:
    """Verifies that faster organisms incur higher metabolic energy consumption."""
    fast_genotype = np.array([2.0, 0.0, 0.0, 0.0])  # High speed
    slow_genotype = np.array([-2.0, 0.0, 0.0, 0.0]) # Low speed

    fast_traits = OrganismTraits.from_genotype(fast_genotype)
    slow_traits = OrganismTraits.from_genotype(slow_genotype)

    assert fast_traits.speed > slow_traits.speed
    assert fast_traits.metabolic_rate > slow_traits.metabolic_rate


def test_lineage_tracker_mrca() -> None:
    """Verifies phylogenetic ancestry recording and Most Recent Common Ancestor (MRCA) queries."""
    tracker = LineageTracker()

    # Ancestor A
    tracker.record_birth("A", parent_ids=[], generation=0, step=0)
    # Offspring B and C from A
    tracker.record_birth("B", parent_ids=["A"], generation=1, step=5)
    tracker.record_birth("C", parent_ids=["A"], generation=1, step=6)
    # Offspring D from B
    tracker.record_birth("D", parent_ids=["B"], generation=2, step=12)

    # MRCA of D and C must be A
    mrca_dc = tracker.most_recent_common_ancestor("D", "C")
    assert mrca_dc == "A"

    # MRCA of D and B must be B
    mrca_db = tracker.most_recent_common_ancestor("D", "B")
    assert mrca_db == "B"

    # Ancestors of D
    ancestors_d = tracker.get_ancestors("D")
    assert "B" in ancestors_d
    assert "A" in ancestors_d


def test_carrying_capacity_bounds_population() -> None:
    """Verifies that population stabilizes and remains bounded by environmental carrying capacity."""
    engine = EcosystemEngine(
        initial_population=20,
        carrying_capacity=40,
        food_growth_rate=40.0,
        max_food=400.0,
        seed=42,
    )

    for _ in range(40):
        engine.step()

    # Living population should remain bounded near carrying capacity
    living = [org for org in engine.organisms if org.alive]
    assert len(living) <= 60  # Allows small transient overshoot without exponential explosion
