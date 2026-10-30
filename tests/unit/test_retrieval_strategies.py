"""Unit tests for memory retrieval strategies against analytical oracles."""

from dataclasses import dataclass

from autogen.memory.retrieval import (
    CompositeRetrieval,
    CosineSimilarityRetrieval,
    ProximityRetrieval,
    RecencyRetrieval,
)
from autogen.simulation.state.types import Position


@dataclass
class MockMemoryItem:
    item_id: str
    step: int
    position: Position
    confidence: float
    feature_vector: tuple[float, ...] = ()


def test_recency_retrieval():
    items = [
        MockMemoryItem("a", step=10, position=Position(0, 0), confidence=1.0),
        MockMemoryItem("b", step=50, position=Position(0, 0), confidence=1.0),
        MockMemoryItem("c", step=30, position=Position(0, 0), confidence=1.0),
    ]

    retriever = RecencyRetrieval()
    res = retriever.retrieve(items, k=2)

    assert len(res) == 2
    assert res[0].item_id == "b"  # step 50
    assert res[1].item_id == "c"  # step 30


def test_proximity_retrieval():
    items = [
        MockMemoryItem("far", step=1, position=Position(10, 10), confidence=1.0),
        MockMemoryItem("close", step=1, position=Position(2, 3), confidence=1.0),
        MockMemoryItem("mid", step=1, position=Position(5, 5), confidence=1.0),
    ]

    retriever = ProximityRetrieval()
    # Query position: (2, 2)
    # dist to close: |2-2| + |3-2| = 1
    # dist to mid: |5-2| + |5-2| = 6
    # dist to far: |10-2| + |10-2| = 16
    res = retriever.retrieve(items, query={"position": Position(2, 2)}, k=3)

    assert [item.item_id for item in res] == ["close", "mid", "far"]


def test_cosine_similarity_retrieval_oracles():
    """Verify cosine similarity against exact mathematical vector projections."""
    items = [
        MockMemoryItem("identical", 1, Position(0, 0), 1.0, feature_vector=(1.0, 0.0)),
        MockMemoryItem("orthogonal", 1, Position(0, 0), 1.0, feature_vector=(0.0, 1.0)),
        MockMemoryItem("forty_five", 1, Position(0, 0), 1.0, feature_vector=(1.0, 1.0)),
    ]

    retriever = CosineSimilarityRetrieval()
    query_vec = (1.0, 0.0)

    res = retriever.retrieve(items, query={"vector": query_vec}, k=3)

    # Identical vector cos(0) = 1.0
    assert res[0].item_id == "identical"
    # Forty-five degree vector cos(pi/4) = 1/sqrt(2) approx 0.707
    assert res[1].item_id == "forty_five"
    # Orthogonal vector cos(pi/2) = 0.0
    assert res[2].item_id == "orthogonal"


def test_composite_retrieval():
    items = [
        MockMemoryItem("close_stale", step=5, position=Position(1, 1), confidence=0.2),
        MockMemoryItem("fresh_close", step=100, position=Position(1, 2), confidence=0.9),
        MockMemoryItem("far_fresh", step=100, position=Position(20, 20), confidence=1.0),
    ]

    retriever = CompositeRetrieval(weight_proximity=0.5, weight_confidence=0.3, weight_recency=0.2)
    res = retriever.retrieve(
        items,
        query={"position": Position(1, 1), "current_step": 100},
        k=2,
    )

    # fresh_close has both high proximity and high confidence/recency
    assert res[0].item_id == "fresh_close"
