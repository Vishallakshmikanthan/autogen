"""Unit tests for SearchProblem protocol, SearchNode, and SearchResult contracts (Module M06)."""

from __future__ import annotations

from autogen.intelligence.search.contracts import (
    SearchNode,
    SearchResult,
    SearchStepEvent,
    reconstruct_path,
)


def test_search_node_and_f_cost():
    """Verifies node lineage, depth, and f-cost evaluation f(n) = g(n) + h(n)."""
    root: SearchNode[tuple[int, int], str] = SearchNode(
        state=(0, 0), g_cost=0.0, h_cost=5.0, depth=0
    )
    assert root.f_cost == 5.0
    assert root.parent is None
    assert root.action is None

    child: SearchNode[tuple[int, int], str] = SearchNode(
        state=(0, 1),
        parent=root,
        action="DOWN",
        g_cost=1.5,
        h_cost=4.0,
        depth=1,
    )
    assert child.f_cost == 5.5
    assert child.depth == 1
    assert child.parent is root


def test_reconstruct_path():
    """Verifies backward pointer traversal to construct forward actions and states."""
    n0: SearchNode[str, str] = SearchNode(state="A", g_cost=0.0)
    n1: SearchNode[str, str] = SearchNode(state="B", parent=n0, action="GO_B", g_cost=1.0)
    n2: SearchNode[str, str] = SearchNode(state="C", parent=n1, action="GO_C", g_cost=2.0)

    actions, states = reconstruct_path(n2)
    assert actions == ["GO_B", "GO_C"]
    assert states == ["A", "B", "C"]


def test_search_step_event_and_result():
    """Verifies SearchStepEvent and SearchResult container fields."""
    event: SearchStepEvent[tuple[int, int], str] = SearchStepEvent(
        algorithm="A*",
        step=1,
        current_state=(1, 2),
        frontier_states=[(1, 3), (2, 2)],
        explored_states=[(1, 2)],
        f_cost=4.0,
        g_cost=2.0,
        nodes_expanded=1,
    )
    assert event.algorithm == "A*"
    assert len(event.frontier_states) == 2

    res = SearchResult(
        success=True,
        path=["RIGHT", "DOWN"],
        states=[(0, 0), (1, 0), (1, 1)],
        total_cost=2.0,
        nodes_expanded=5,
        nodes_generated=8,
        max_frontier_size=3,
        runtime_seconds=0.001,
    )
    assert res.success is True
    assert res.total_cost == 2.0
    assert len(res.path) == 2
