"""Unit tests for Central Algorithm Registry and Metadata Catalog (SYS-19, Module M21)."""

import pytest

from autogen.intelligence.registry import (
    AlgorithmFamily,
    AlgorithmMetadata,
    AlgorithmRegistry,
    populate_default_catalog,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry starts fresh and repopulates standard catalog."""
    AlgorithmRegistry.clear()
    populate_default_catalog()
    yield


def test_registry_populated():
    """Verify that default algorithms are registered across multiple families."""
    count = AlgorithmRegistry.count()
    assert count >= 8

    search_algs = AlgorithmRegistry.get_by_family(AlgorithmFamily.SEARCH)
    assert len(search_algs) >= 2
    assert any(a.algorithm_id == "astar" for a in search_algs)

    hybrid_algs = AlgorithmRegistry.get_by_family(AlgorithmFamily.HYBRID)
    assert len(hybrid_algs) >= 1
    assert any(a.algorithm_id == "hybrid_cognitive_architecture" for a in hybrid_algs)


def test_duplicate_registration_rejected():
    """Verify that registering an already registered algorithm raises ValueError."""
    meta = AlgorithmMetadata(
        algorithm_id="astar",
        name="A* Graph Search",
        family=AlgorithmFamily.SEARCH,
        doc_path="docs/algorithms/search/informed_search.md",
        complexity_time="O(b^d)",
        complexity_space="O(b^d)",
    )
    with pytest.raises(ValueError, match="already registered"):
        AlgorithmRegistry.register(meta)


def test_retrieve_by_id():
    """Verify metadata lookup and case-insensitivity."""
    meta = AlgorithmRegistry.get("ASTAR")
    assert meta.algorithm_id == "astar"
    assert meta.family == AlgorithmFamily.SEARCH
    assert meta.citations[0].author_or_source == "Russell & Norvig"


def test_unknown_algorithm_raises_key_error():
    """Verify that requesting an unknown algorithm raises KeyError."""
    with pytest.raises(KeyError, match="not found in AlgorithmRegistry"):
        AlgorithmRegistry.get("quantum_teleportation_search")
