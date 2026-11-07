"""Unit and oracle tests for Lossy Gossip Transmission and Provenance Dynamics."""

import numpy as np
import pytest

from autogen.intelligence.village.gossip import (
    Belief,
    BeliefStore,
    GossipEngine,
    compute_acceptance_probability,
    compute_listener_credibility,
    compute_speaker_spin,
    merge_corroborated_confidence,
)


class TestGossipTransmissionMath:
    """Mathematical verification of gossip spin, credibility, and acceptance probability."""

    def test_speaker_spin_hand_computed_fixture(self) -> None:
        """Verify hand-computed speaker spin exaggeration:

        spin = 1.0 + 0.25 * max(0, a_s) + 0.20 * dislike
        where dislike = max(0, (0.50 - affinity) * 2.0)
        """
        # Neutral: a=0.0, aff=0.5 -> dislike=0.0 -> spin=1.00
        assert compute_speaker_spin(arousal=0.0, affinity_to_subject=0.50) == pytest.approx(1.00, abs=1e-4)

        # High arousal (0.8) + strong dislike (aff=0.10 -> dislike = (0.5 - 0.1)*2 = 0.8)
        # spin = 1.0 + 0.25(0.8) + 0.20(0.8) = 1.0 + 0.20 + 0.16 = 1.36
        spin = compute_speaker_spin(arousal=0.80, affinity_to_subject=0.10)
        assert spin == pytest.approx(1.36, abs=1e-4)

    def test_listener_credibility_fixture(self) -> None:
        """Verify listener source credibility evaluation:

        cred = 0.50 * trust + 0.30 * status + 0.20 * affinity
        """
        # trust=0.80, status=0.60, affinity=0.70
        # cred = 0.50(0.80) + 0.30(0.60) + 0.20(0.70) = 0.40 + 0.18 + 0.14 = 0.72
        cred = compute_listener_credibility(
            trust_in_speaker=0.80,
            speaker_status=0.60,
            affinity_for_speaker=0.70,
        )
        assert cred == pytest.approx(0.72, abs=1e-4)

    def test_acceptance_probability_monotonicity(self) -> None:
        """Higher credibility and higher consistency strictly increase acceptance probability."""
        p_low = compute_acceptance_probability(credibility=0.30, consistency=0.50, openness=0.50)
        p_high = compute_acceptance_probability(credibility=0.80, consistency=0.90, openness=0.50)
        assert p_high > p_low
        assert 0.0 <= p_low <= 1.0
        assert 0.0 <= p_high <= 1.0


class TestProvenanceAndCorroboration:
    """Verification of rumor echo suppression and multi-source Noisy-OR fusion."""

    def test_echo_loop_suppression_invariant(self) -> None:
        """If listener is already in source chain, transmission MUST be rejected immediately."""
        circulating_belief = Belief(
            belief_id="b1",
            subject="bheem",
            claim="stole_grain",
            valence=-0.60,
            confidence=0.85,
            source_chain=("arun", "imran", "kavya"),  # Arun is the originator
            first_hand=False,
            heard_at_step=5,
        )

        rng = np.random.default_rng(42)
        # Kavya attempts to tell Arun (who originated the claim)
        res, accepted, reason = GossipEngine.transmit(
            speaker_id="kavya",
            listener_id="arun",
            belief=circulating_belief,
            speaker_arousal=0.5,
            speaker_affinity_to_subject=0.2,
            listener_trust_in_speaker=0.9,
            listener_status_of_speaker=0.8,
            listener_affinity_for_speaker=0.8,
            listener_openness=0.9,
            listener_prior_belief=None,
            rng=rng,
            step=10,
        )

        assert accepted is False
        assert res is None
        assert reason == "loop_echo_suppression"

    def test_noisy_or_independent_corroboration(self) -> None:
        """Two independent source chains combine confidence via Noisy-OR:

        c_merged = 1 - (1 - c_prior)(1 - c_incoming * cred)
        """
        # Prior: c=0.60. Incoming: c=0.50, cred=1.0 -> effective = 0.50
        # c_merged = 1 - (1 - 0.60)(1 - 0.50) = 1 - (0.40)(0.50) = 1 - 0.20 = 0.80
        merged = merge_corroborated_confidence(
            prior_confidence=0.60,
            incoming_confidence=0.50,
            credibility=1.00,
        )
        assert merged == pytest.approx(0.80, abs=1e-4)

        # Three independent reports with confidence 0.50 each
        # Step 1: 0.50
        # Step 2: 1 - (0.5 * 0.5) = 0.75
        step2 = merge_corroborated_confidence(0.50, 0.50, 1.0)
        assert step2 == pytest.approx(0.75, abs=1e-4)
        # Step 3: 1 - (0.25 * 0.5) = 0.875
        step3 = merge_corroborated_confidence(step2, 0.50, 1.0)
        assert step3 == pytest.approx(0.875, abs=1e-4)

    def test_belief_store_digest(self) -> None:
        """BeliefStore maintains deterministic SHA-256 digest of stored beliefs."""
        store = BeliefStore("arun")
        store.put(
            Belief(
                belief_id="b1",
                subject="bheem",
                claim="stole_grain",
                valence=-0.5,
                confidence=0.8,
                source_chain=("arun",),
                heard_at_step=2,
            )
        )
        d1 = store.digest()
        assert isinstance(d1, str) and len(d1) == 64
        assert store.digest() == d1

        # Adding another belief changes digest deterministically
        store.put(
            Belief(
                belief_id="b2",
                subject="lata",
                claim="generous",
                valence=0.7,
                confidence=0.9,
                source_chain=("lata",),
                heard_at_step=3,
            )
        )
        d2 = store.digest()
        assert d2 != d1
