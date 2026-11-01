"""Integration test suite for Gate 10 showpiece scenarios (SYS-19, Gate 10, Module M22).

Validates that all seven canonical research showpieces execute deterministically,
return valid Pydantic ShowpieceRunResult objects, and satisfy their physical/algorithmic invariants.
"""

from __future__ import annotations

from scripts.run_showpiece_ablation_arena import run_ablation_arena
from scripts.run_showpiece_algorithm_duel import run_algorithm_duel
from scripts.run_showpiece_evolution_of_tribes import run_evolution_of_tribes
from scripts.run_showpiece_ghost_in_the_fog import run_ghost_in_the_fog
from scripts.run_showpiece_the_changing_world import run_the_changing_world
from scripts.run_showpiece_time_machine import run_time_machine
from scripts.run_showpiece_tragedy_of_the_commons import run_tragedy_of_the_commons


def test_showpiece_1_ghost_in_the_fog():
    res = run_ghost_in_the_fog(seed=42, fast=True)
    assert res.metrics["hybrid_survival_mean"] >= res.metrics["reflex_survival_mean"]
    assert res.metrics["initial_entropy_bits"] > 0.0


def test_showpiece_2_algorithm_duel():
    res = run_algorithm_duel(seed=42, fast=True)
    assert res.metrics["astar_expansions"] <= res.metrics["ucs_expansions"]
    assert res.metrics["optimal_cost"] > 0.0


def test_showpiece_3_tragedy_of_the_commons():
    res = run_tragedy_of_the_commons(seed=42, fast=True)
    assert res.metrics["prosocial_depletion_tick"] >= res.metrics["selfish_depletion_tick"]
    assert res.metrics["prosocial_total_harvest"] >= res.metrics["selfish_total_harvest"]


def test_showpiece_4_evolution_of_tribes():
    res = run_evolution_of_tribes(seed=42, fast=True)
    assert res.metrics["final_generation_fitness"] >= res.metrics["initial_generation_fitness"]
    assert res.metrics["num_clusters_k"] == 3.0


def test_showpiece_5_ablation_arena():
    res = run_ablation_arena(seed=42, fast=True)
    assert res.metrics["full_baseline_mean_return"] is not None
    assert len(res.traces) >= 4


def test_showpiece_6_time_machine():
    res = run_time_machine(seed=42, fast=True)
    assert res.metrics["final_divergence_distance"] >= 1.0
    assert res.metrics["total_divergence_steps"] > 0


def test_showpiece_7_the_changing_world():
    res = run_the_changing_world(seed=42, fast=True)
    assert res.metrics["spring_return_hybrid"] > res.metrics["drought_return_hybrid"]
    assert res.metrics["autumn_return_hybrid"] > res.metrics["drought_return_hybrid"]


def test_determinism_across_runs():
    """Verify bit-for-bit identical metrics when running twice with identical seed."""
    res1 = run_algorithm_duel(seed=999, fast=True)
    res2 = run_algorithm_duel(seed=999, fast=True)
    assert res1.metrics == res2.metrics
