"""Unit tests for local and stochastic search algorithms (Hill Climbing, Simulated Annealing, Beam) (Module M06)."""

from __future__ import annotations

from autogen.intelligence.search import (
    GridSearchProblem,
    HillClimbingSearch,
    LocalBeamSearch,
    SimulatedAnnealingSearch,
)
from autogen.simulation.state.types import Position


def _make_open_grid(w: int = 8, h: int = 8) -> GridSearchProblem:
    grid = [[1.0 for _ in range(w)] for _ in range(h)]
    return GridSearchProblem.from_cost_matrix(
        grid,
        start=Position(0, 0),
        goal=Position(w - 1, h - 1),
        heuristic_type="manhattan",
    )


def test_hill_climbing_monotonic_surface():
    """Hill climbing rapidly finds the global optimum on a monotonic Manhattan slope."""
    problem = _make_open_grid(8, 8)
    hc = HillClimbingSearch(max_steps=50)
    result = hc.solve(problem)

    assert result.success is True
    assert result.states[-1] == Position(7, 7)
    assert len(result.path) == 14  # 7 + 7 steps


def test_simulated_annealing_cooling_and_convergence():
    """Simulated Annealing cools temperature and reaches the target on grid."""
    problem = _make_open_grid(5, 5)
    sa = SimulatedAnnealingSearch(
        initial_temperature=10.0,
        cooling_rate=0.995,
        min_temperature=1e-3,
        max_steps=3000,
        seed=42,
    )
    result = sa.solve(problem)

    assert result.success is True
    assert result.states[-1] == Position(4, 4)


def test_local_beam_search_k_states():
    """Local Beam Search tracks k parallel candidates and successfully reaches goal."""
    problem = _make_open_grid(8, 8)
    beam = LocalBeamSearch(beam_width=4, max_steps=100)
    result = beam.solve(problem)

    assert result.success is True
    assert result.states[-1] == Position(7, 7)
