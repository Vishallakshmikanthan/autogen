"""Unit and oracle tests for Memory Stream, Importance Scoring, and Bounded Forgetting."""

import math

import pytest

from autogen.intelligence.village.memory_stream import (
    MemoryItem,
    MemoryStream,
    compute_importance,
    compute_strength,
)


class TestMemoryImportance:
    """Mathematical verification of episodic memory importance scoring."""

    def test_hand_computed_fixtures(self) -> None:
        """Verify hand-computed mathematical fixtures for importance formula:

        I = clip(0.35 |v| + 0.25 a+ + 0.20 s + 0.10 r + 0.10 c, 0.0, 1.0)
        """
        # Baseline minimum: all zero
        assert compute_importance(
            valence=0.0,
            arousal=0.0,
            self_involvement=0.0,
            rarity=0.0,
            consequence=0.0,
        ) == pytest.approx(0.00, abs=1e-5)

        # Baseline maximum: all 1.0 -> 0.35 + 0.25 + 0.20 + 0.10 + 0.10 = 1.00
        assert compute_importance(
            valence=1.0,
            arousal=1.0,
            self_involvement=1.0,
            rarity=1.0,
            consequence=1.0,
        ) == pytest.approx(1.00, abs=1e-5)

        # Negative extreme valence: |-1.0| = 1.0
        assert compute_importance(
            valence=-1.0,
            arousal=1.0,
            self_involvement=1.0,
            rarity=1.0,
            consequence=1.0,
        ) == pytest.approx(1.00, abs=1e-5)

        # Intermediate fixture: v=-0.5, a=0.4, s=0.5, r=0.5, c=0.2
        # 0.35(0.5) + 0.25(0.4) + 0.20(0.5) + 0.10(0.5) + 0.10(0.2)
        # = 0.175 + 0.10 + 0.10 + 0.05 + 0.02 = 0.445
        assert compute_importance(
            valence=-0.5,
            arousal=0.4,
            self_involvement=0.5,
            rarity=0.5,
            consequence=0.2,
        ) == pytest.approx(0.445, abs=1e-4)

    def test_negative_arousal_rectification(self) -> None:
        """Negative arousal (lethargy/calm) does not reduce importance below zero (a+ = max(0, a))."""
        score_neutral_arousal = compute_importance(
            valence=0.4,
            arousal=0.0,
            self_involvement=0.5,
            rarity=0.5,
            consequence=0.2,
        )
        score_negative_arousal = compute_importance(
            valence=0.4,
            arousal=-0.8,
            self_involvement=0.5,
            rarity=0.5,
            consequence=0.2,
        )
        assert score_neutral_arousal == pytest.approx(score_negative_arousal, abs=1e-5)

    def test_monotonicity(self) -> None:
        """Verify importance is strictly monotonic with respect to |v|, a+, s, r, and c."""
        base = compute_importance(
            valence=0.2,
            arousal=0.2,
            self_involvement=0.2,
            rarity=0.2,
            consequence=0.2,
        )
        higher_v = compute_importance(
            valence=0.6,
            arousal=0.2,
            self_involvement=0.2,
            rarity=0.2,
            consequence=0.2,
        )
        assert higher_v > base

        higher_s = compute_importance(
            valence=0.2,
            arousal=0.2,
            self_involvement=0.8,
            rarity=0.2,
            consequence=0.2,
        )
        assert higher_s > base

        higher_r = compute_importance(
            valence=0.2,
            arousal=0.2,
            self_involvement=0.2,
            rarity=0.9,
            consequence=0.2,
        )
        assert higher_r > base


class TestRetentionStrength:
    """Mathematical verification of Ebbinghaus-power retention strength."""

    def test_hand_computed_fixtures(self) -> None:
        """Verify hand-computed retention decay and rehearsal boost fixtures."""
        # Initial item at encoding time: dt = 0, n_acc = 0 -> S = I_m
        s_initial = compute_strength(
            importance=0.80,
            step=100,
            last_access_step=100,
            access_count=0,
            s_0=96.0,
        )
        assert s_initial == pytest.approx(0.80, abs=1e-5)

        # Decayed after 96 steps (1 halflife factor: dt=96, s0=96 -> exp(-1) ~= 0.367879)
        # S = 0.80 * exp(-1) * (1 + 0) = 0.80 * 0.36787944 = 0.2943035
        s_decayed = compute_strength(
            importance=0.80,
            step=196,
            last_access_step=100,
            access_count=0,
            s_0=96.0,
        )
        assert s_decayed == pytest.approx(0.80 * math.exp(-1.0), abs=1e-4)

        # Rehearsal boost: n_acc = 5, dt = 0
        # S = 0.80 * 1.0 * (1 + 0.30 * ln(6)) = 0.80 * (1 + 0.30 * 1.791759) = 0.80 * 1.537528 = 1.230022
        s_rehearsed = compute_strength(
            importance=0.80,
            step=100,
            last_access_step=100,
            access_count=5,
            s_0=96.0,
        )
        expected_boost = 0.80 * (1.0 + 0.30 * math.log(6.0))
        assert s_rehearsed == pytest.approx(expected_boost, abs=1e-4)

    def test_reflection_extended_decay(self) -> None:
        """Reflections have 3x longer half-life (s_0 * 3.0), decaying much slower."""
        s_normal = compute_strength(
            importance=0.90,
            step=200,
            last_access_step=100,
            access_count=0,
            is_reflection=False,
            s_0=96.0,
        )
        s_reflection = compute_strength(
            importance=0.90,
            step=200,
            last_access_step=100,
            access_count=0,
            is_reflection=True,
            s_0=96.0,
        )
        assert s_reflection > s_normal


class TestMemoryStreamBoundedCapacity:
    """Verification of capacity cap (300) and unreflected memory protection."""

    def test_unreflected_protection_invariant(self) -> None:
        """When capacity is exceeded, reflected memories must be evicted before unreflected memories."""
        stream = MemoryStream(agent_id="villager_1", capacity=300)

        # Add 300 unreflected memories (step 100, importance 0.50)
        for i in range(300):
            item = MemoryItem(
                mid=f"unreflected_{i}",
                step=100,
                kind="observation",
                summary=f"Unreflected item {i}",
                importance=0.50,
                last_access_step=100,
            )
            stream.add(item)

        assert len(stream) == 300
        assert len(stream.unreflected()) == 300

        # Add 50 reflected memories (step 100, importance 0.60, kind='reflection')
        for i in range(50):
            ref_item = MemoryItem(
                mid=f"reflected_{i}",
                step=100,
                kind="reflection",
                summary=f"Reflected item {i}",
                importance=0.60,
                last_access_step=100,
            )
            stream.add(ref_item)
            stream.mark_reflected([f"reflected_{i}"])

        assert len(stream) == 350
        # Trigger pruning at step 100
        evicted = stream.prune(current_step=100)

        # Capacity should now be strictly capped at 300
        assert len(stream) == 300
        assert len(evicted) == 50

        # All 50 evicted items must be from the reflected set!
        for i in range(50):
            assert f"reflected_{i}" in evicted

        # All 300 unreflected items remain completely intact!
        all_ids = {m.mid for m in stream.all_items()}
        for i in range(300):
            assert f"unreflected_{i}" in all_ids

    def test_create_and_add_rarity(self) -> None:
        """Repeated events with same tag decrease rarity and reduce importance."""
        stream = MemoryStream(agent_id="villager_1", capacity=100)
        m1 = stream.create_and_add(
            step=1,
            kind="observation",
            tags=("farm", "wheat"),
            valence=0.0,
            pad=(0.0, 0.0, 0.0),
            self_involvement=0.5,
            consequence=0.0,
        )
        # 1st event has rarity = 1 / (1 + 0) = 1.0
        # Now add 4 more farm/wheat events at step 2..5
        for s in range(2, 6):
            stream.create_and_add(
                step=s,
                kind="observation",
                tags=("farm", "wheat"),
                valence=0.0,
                pad=(0.0, 0.0, 0.0),
                self_involvement=0.5,
                consequence=0.0,
            )

        # 6th event at step 6: rarity should be 1 / (1 + 5) = 1/6
        m6 = stream.create_and_add(
            step=6,
            kind="observation",
            tags=("farm", "wheat"),
            valence=0.0,
            pad=(0.0, 0.0, 0.0),
            self_involvement=0.5,
            consequence=0.0,
        )
        assert m6.importance < m1.importance

    def test_digest_stability(self) -> None:
        """Verify digest produces a deterministic SHA-256 hex string and changes upon item changes."""
        stream = MemoryStream(agent_id="villager_1", capacity=10)
        stream.add(
            MemoryItem(
                mid="m1",
                step=1,
                kind="observation",
                importance=0.5,
                last_access_step=1,
            )
        )
        d1 = stream.digest()
        assert isinstance(d1, str) and len(d1) == 64
        assert stream.digest() == d1

        # Adding another item changes the digest deterministically
        stream.add(
            MemoryItem(
                mid="m2",
                step=2,
                kind="observation",
                importance=0.6,
                last_access_step=2,
            )
        )
        d2 = stream.digest()
        assert d2 != d1
