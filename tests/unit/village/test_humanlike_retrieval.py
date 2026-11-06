"""Unit and oracle tests for Multimodal Episodic Memory Retrieval Engine."""

import math

import pytest

from autogen.intelligence.village.memory_retrieval import (
    MemoryQuery,
    retrieve_top_k,
    score_memory,
)
from autogen.intelligence.village.memory_stream import MemoryItem, MemoryStream


class TestMemoryRetrievalScoring:
    """Mathematical verification of composite retrieval scoring components."""

    def test_hand_computed_score_breakdown(self) -> None:
        """Verify hand-computed composite retrieval score and component breakdown:

        score = w_r * exp(-dt / tau_r) + w_i * I_m + w_q * J(m, q) + w_m * (1 - d_pad / 2*sqrt(3))
        """
        mem = MemoryItem(
            mid="mem_hand_calc",
            step=50,
            last_access_step=50,
            importance=0.80,
            kind="observation",
            actors=("v1",),
            location="granary",
            tags=("wheat", "food"),
            pad=(0.2, 0.4, 0.1),
        )

        query = MemoryQuery(
            step=74,  # delta_t = 24.0 -> exp(-24/24) = exp(-1.0) ~= 0.3678794
            actors=("v1", "v2"),
            location="granary",
            tags=("wheat",),
            pad=(0.2, 0.4, 0.1),  # d_pad = 0.0 -> mood = 1.0
        )

        breakdown = score_memory(mem, query)

        # Expected components:
        # Recency: exp(-1) ~= 0.3679
        assert breakdown.recency_score == pytest.approx(math.exp(-1.0), abs=1e-4)

        # Importance: 0.80
        assert breakdown.importance_score == pytest.approx(0.80, abs=1e-4)

        # Relevance: Jaccard overlap
        # mem_set = {"v1", "granary", "wheat", "food"} (size 4)
        # query_set = {"v1", "v2", "granary", "wheat"} (size 4)
        # intersection = {"v1", "granary", "wheat"} (size 3)
        # union = {"v1", "v2", "granary", "wheat", "food"} (size 5)
        # Jaccard = 3/5 = 0.60
        assert breakdown.relevance_score == pytest.approx(0.60, abs=1e-4)

        # Mood: d_pad = 0 -> 1.0
        assert breakdown.mood_score == pytest.approx(1.00, abs=1e-4)

        # Total = 1.0(0.3678794) + 1.0(0.80) + 1.5(0.60) + 0.4(1.00)
        #       = 0.3678794 + 0.80 + 0.90 + 0.40 = 2.4678794
        assert breakdown.total_score == pytest.approx(2.4679, abs=1e-4)

    def test_mood_congruence_priming_invariant(self) -> None:
        """When an agent is distressed/fearful, fear-congruent memories must score higher

        than joy-congruent memories of identical importance and recency (Bower, 1981).
        """
        # Memory A: Joyful event
        mem_joy = MemoryItem(
            mid="mem_joy",
            step=100,
            last_access_step=100,
            importance=0.70,
            kind="observation",
            tags=("common",),
            pad=(0.80, 0.20, 0.40),  # Joy
        )
        # Memory B: Fearful event
        mem_fear = MemoryItem(
            mid="mem_fear",
            step=100,
            last_access_step=100,
            importance=0.70,
            kind="observation",
            tags=("common",),
            pad=(-0.64, 0.60, -0.43),  # Fear
        )

        # Query in Fear state
        fear_query = MemoryQuery(
            step=100,
            tags=("common",),
            pad=(-0.64, 0.60, -0.43),  # Agent currently feeling Fear
        )

        bd_joy = score_memory(mem_joy, fear_query)
        bd_fear = score_memory(mem_fear, fear_query)

        # Recency and importance are identical
        assert bd_joy.recency_score == bd_fear.recency_score
        assert bd_joy.importance_score == bd_fear.importance_score
        assert bd_joy.relevance_score == bd_fear.relevance_score

        # Fear memory has higher mood congruence and higher total score
        assert bd_fear.mood_score > bd_joy.mood_score
        assert bd_fear.total_score > bd_joy.total_score


class TestTopKRetrieval:
    """Verification of top-k ranking, rehearsal side-effects, and deterministic tie-breaking."""

    def test_rehearsal_side_effect_invariant(self) -> None:
        """rehearse=False must leave stream items untouched, while rehearse=True updates access count."""
        stream = MemoryStream(agent_id="meera", capacity=50)
        m = MemoryItem(
            mid="meera:1:1",
            step=1,
            last_access_step=1,
            importance=0.75,
            kind="observation",
            tags=("harvest",),
            access_count=0,
        )
        stream.add(m)

        query = MemoryQuery(step=20, tags=("harvest",))

        # Read-only retrieval
        results_no_rehearse = retrieve_top_k(stream, query, k=1, rehearse=False)
        assert len(results_no_rehearse) == 1
        stored_item = stream.get("meera:1:1")
        assert stored_item is not None
        assert stored_item.access_count == 0
        assert stored_item.last_access_step == 1

        # Rehearsal retrieval
        results_rehearse = retrieve_top_k(stream, query, k=1, rehearse=True)
        assert len(results_rehearse) == 1
        rehearsed_item = stream.get("meera:1:1")
        assert rehearsed_item is not None
        assert rehearsed_item.access_count == 1
        assert rehearsed_item.last_access_step == 20

    def test_peer_interaction_retrieval(self) -> None:
        """When interacting with Bheem, memories involving Bheem rank highest."""
        stream = MemoryStream(agent_id="meera", capacity=50)
        # Memory 1: General farming
        m_farm = MemoryItem(
            mid="meera:1:1",
            step=10,
            last_access_step=10,
            importance=0.60,
            kind="observation",
            location="farmland",
            tags=("wheat",),
        )
        # Memory 2: Bheem trade
        m_bheem = MemoryItem(
            mid="meera:1:2",
            step=10,
            last_access_step=10,
            importance=0.60,
            kind="observation",
            actors=("bheem",),
            tags=("trade", "wheat"),
        )
        stream.add(m_farm)
        stream.add(m_bheem)

        query = MemoryQuery(step=15, actors=("bheem",))
        results = retrieve_top_k(stream, query, k=1, rehearse=False)
        assert len(results) == 1
        top_item, bd = results[0]
        assert top_item.mid == "meera:1:2"
        assert "bheem" in top_item.actors

    def test_deterministic_tie_breaking(self) -> None:
        """Ties in total score are broken deterministically by mid ascending."""
        stream = MemoryStream(agent_id="v1", capacity=10)
        m_b = MemoryItem(
            mid="v1:10:b",
            step=10,
            last_access_step=10,
            importance=0.50,
            kind="observation",
            tags=("work",),
        )
        m_a = MemoryItem(
            mid="v1:10:a",
            step=10,
            last_access_step=10,
            importance=0.50,
            kind="observation",
            tags=("work",),
        )
        stream.add(m_b)
        stream.add(m_a)

        query = MemoryQuery(step=10, tags=("work",))
        results = retrieve_top_k(stream, query, k=2, rehearse=False)
        assert len(results) == 2
        # Both have identical scores, 'v1:10:a' comes before 'v1:10:b'
        assert results[0][0].mid == "v1:10:a"
        assert results[1][0].mid == "v1:10:b"
