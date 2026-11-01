"""Tests for discrete and continuous evolutionary benchmark functions and optimizers (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.evolution.benchmarks import (
    Ackley,
    OneMax,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from autogen.intelligence.evolution.differential_evolution import DifferentialEvolution
from autogen.intelligence.evolution.evolution_strategies import EvolutionStrategy
from autogen.intelligence.evolution.ga import GeneticAlgorithm
from autogen.intelligence.evolution.pso import ParticleSwarmOptimization


def test_benchmark_function_evaluations() -> None:
    """Verifies that mathematical benchmark functions evaluate correctly at global optima."""
    # OneMax
    onemax = OneMax(length=10)
    assert onemax(np.ones(10)) == 10.0
    assert onemax(np.zeros(10)) == 0.0

    # Sphere
    sphere = Sphere(dimension=5)
    assert sphere.raw_cost(np.zeros(5)) == 0.0
    assert sphere(np.zeros(5)) == 0.0
    assert sphere(np.ones(5)) == -5.0

    # Rastrigin
    rastrigin = Rastrigin(dimension=5)
    assert rastrigin.raw_cost(np.zeros(5)) == 0.0
    assert rastrigin(np.zeros(5)) == 0.0

    # Rosenbrock
    rosenbrock = Rosenbrock(dimension=5)
    assert rosenbrock.raw_cost(np.ones(5)) == 0.0
    assert rosenbrock(np.ones(5)) == 0.0

    # Ackley
    ackley = Ackley(dimension=5)
    assert abs(ackley.raw_cost(np.zeros(5))) < 1e-12
    assert abs(ackley(np.zeros(5))) < 1e-12


def test_ga_onemax_convergence() -> None:
    """Verifies that GeneticAlgorithm achieves 100% ones on OneMax benchmark."""
    length = 40
    onemax = OneMax(length=length)

    ga = GeneticAlgorithm(
        fitness_function=onemax,
        genome_type="binary",
        genome_length=length,
        population_size=60,
        elitism_count=2,
        seed=123,
    )

    diag = ga.evolve(generations=40)
    best = ga.get_best()

    assert best.fitness == float(length)
    assert diag.best_fitness == float(length)
    assert np.all(best.genome == 1)


def test_differential_evolution_sphere() -> None:
    """Verifies Differential Evolution converges near zero on the continuous Sphere benchmark."""
    sphere = Sphere(dimension=5)
    de = DifferentialEvolution(
        fitness_function=sphere,
        dimension=5,
        population_size=30,
        bounds=(-5.0, 5.0),
        mutation_factor=0.7,
        crossover_rate=0.8,
        seed=42,
    )

    diag = de.evolve(generations=50)
    best = de.get_best()

    # Fitness is -raw_cost, so near zero is > -0.05
    assert best.fitness > -0.05
    assert diag.best_fitness > -0.05
    assert np.all(np.abs(best.genome) < 0.3)


def test_pso_sphere_convergence() -> None:
    """Verifies Particle Swarm Optimization converges on the Sphere benchmark."""
    sphere = Sphere(dimension=5)
    pso = ParticleSwarmOptimization(
        fitness_function=sphere,
        dimension=5,
        swarm_size=30,
        bounds=(-5.0, 5.0),
        seed=42,
    )

    diag = pso.evolve(generations=35)
    best = pso.get_best()

    assert best.fitness > -0.1
    assert diag.best_fitness > -0.1


def test_evolution_strategy_sphere_convergence() -> None:
    """Verifies (mu + lambda)-ES converges on the Sphere benchmark with self-adaptive step sizes."""
    sphere = Sphere(dimension=5)
    es = EvolutionStrategy(
        fitness_function=sphere,
        dimension=5,
        mu=10,
        lambda_=40,
        plus_selection=True,
        bounds=(-5.0, 5.0),
        seed=42,
    )

    diag = es.evolve(generations=35)
    best = es.get_best()

    assert best.fitness > -0.1
    assert diag.best_fitness > -0.1
