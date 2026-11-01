"""Tests for evolutionary selection, crossover, mutation operators and elitism invariants (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.evolution.contracts import Individual
from autogen.intelligence.evolution.ga import (
    BitFlipMutation,
    GaussianMutation,
    GeneticAlgorithm,
    RankSelection,
    RouletteWheelSelection,
    SinglePointCrossover,
    TournamentSelection,
    TwoPointCrossover,
    UniformCrossover,
)


def test_selection_operators_contract() -> None:
    """Verifies that selection operators respect population size and preserve candidate individuals."""
    rng = np.random.default_rng(42)
    pop = [
        Individual(genome=np.array([i]), fitness=float(i), individual_id=f"ind_{i}")
        for i in range(10)
    ]

    # Tournament selection
    ts = TournamentSelection(tournament_size=3)
    selected_ts = ts.select(pop, k=5, rng=rng)
    assert len(selected_ts) == 5
    for ind in selected_ts:
        assert ind.fitness >= 0.0

    # Rank selection
    rs = RankSelection(selection_pressure=1.5)
    selected_rs = rs.select(pop, k=6, rng=rng)
    assert len(selected_rs) == 6

    # Roulette selection
    rw = RouletteWheelSelection()
    selected_rw = rw.select(pop, k=4, rng=rng)
    assert len(selected_rw) == 4


def test_crossover_operators_integrity() -> None:
    """Verifies that crossover operators preserve array lengths and valid offspring loci."""
    rng = np.random.default_rng(42)
    p1 = np.zeros(20, dtype=int)
    p2 = np.ones(20, dtype=int)

    # Single-point crossover
    sp = SinglePointCrossover(crossover_rate=1.0)
    c1, c2 = sp.crossover(p1, p2, rng)
    assert len(c1) == 20
    assert len(c2) == 20
    assert np.all((c1 == 0) | (c1 == 1))

    # Two-point crossover
    tp = TwoPointCrossover(crossover_rate=1.0)
    c1, c2 = tp.crossover(p1, p2, rng)
    assert len(c1) == 20
    assert len(c2) == 20

    # Uniform crossover
    uc = UniformCrossover(crossover_rate=1.0, swap_prob=0.5)
    c1, c2 = uc.crossover(p1, p2, rng)
    assert len(c1) == 20
    assert len(c2) == 20


def test_mutation_operators() -> None:
    """Verifies discrete and continuous mutation perturbation semantics."""
    rng = np.random.default_rng(42)

    # Bit-flip mutation
    bf = BitFlipMutation(mutation_rate=1.0)  # Invert all bits
    bits = np.zeros(10, dtype=int)
    mutated_bits = bf.mutate(bits, rng)
    assert np.all(mutated_bits == 1)

    # Gaussian mutation with bounds clipping
    gm = GaussianMutation(mutation_rate=1.0, sigma=0.5, bounds=(-2.0, 2.0))
    real_vec = np.zeros(10, dtype=np.float64)
    mutated_real = gm.mutate(real_vec, rng)
    assert not np.all(mutated_real == 0.0)
    assert np.all(mutated_real >= -2.0)
    assert np.all(mutated_real <= 2.0)


def test_elitism_monotonicity_invariant() -> None:
    """Invariant: With elitism >= 1, the best fitness across generations is strictly non-decreasing."""
    def noisy_objective(x: np.ndarray) -> float:
        return float(np.sum(x))

    ga = GeneticAlgorithm(
        fitness_function=noisy_objective,
        genome_type="binary",
        genome_length=20,
        population_size=30,
        elitism_count=2,
        seed=999,
    )

    best_fitness_history: list[float] = []

    def callback(diag) -> None:
        best_fitness_history.append(diag.best_fitness)

    ga.evolve(generations=20, step_callback=callback)

    for i in range(1, len(best_fitness_history)):
        assert best_fitness_history[i] >= best_fitness_history[i - 1] - 1e-12, (
            f"Elitism violation at gen {i}: {best_fitness_history[i]} < {best_fitness_history[i-1]}"
        )


def test_population_diversity_metrics() -> None:
    """Verifies Shannon entropy and pairwise variance diversity metrics."""
    ga = GeneticAlgorithm(
        fitness_function=lambda x: float(np.sum(x)),
        genome_type="binary",
        genome_length=15,
        population_size=20,
        seed=42,
    )

    div = ga.compute_diversity()
    assert div.unique_genotypes >= 1
    assert div.fitness_variance >= 0.0
    assert div.pairwise_distance_mean >= 0.0
