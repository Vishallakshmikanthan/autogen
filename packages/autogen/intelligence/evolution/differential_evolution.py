"""Differential Evolution (DE/rand/1/bin) continuous vector optimizer (SYS-15, Module M16)."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import numpy as np

from autogen.intelligence.evolution.contracts import (
    DiversityMetrics,
    EvolutionDiagnostics,
    FitnessFunction,
    Individual,
)


class DifferentialEvolution:
    """DE/rand/1/bin continuous vector optimizer implemented from scratch in pure NumPy."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        dimension: int = 10,
        population_size: int = 40,
        bounds: tuple[float, float] = (-5.12, 5.12),
        mutation_factor: float = 0.7,  # F
        crossover_rate: float = 0.8,   # CR
        seed: int = 42,
    ) -> None:
        self.fitness_function = fitness_function
        self.dimension = dimension
        self.population_size = max(4, population_size)
        self.bounds = bounds
        self.f = mutation_factor
        self.cr = crossover_rate
        self.rng = np.random.default_rng(seed)

        self.generation = 0
        self.evaluations = 0
        self.population: list[Individual] = self._init_population()
        self.history: list[EvolutionDiagnostics] = []

    def _init_population(self) -> list[Individual]:
        population: list[Individual] = []
        low, high = self.bounds
        for _ in range(self.population_size):
            genome = self.rng.uniform(low, high, size=self.dimension)
            fit = float(self.fitness_function(genome))
            self.evaluations += 1
            ind = Individual(
                genome=genome,
                fitness=fit,
                raw_fitness=fit,
                individual_id=str(uuid.uuid4())[:8],
                age=0,
            )
            population.append(ind)
        return population

    def compute_diversity(self) -> DiversityMetrics:
        """Computes Euclidean and variance diversity across population vectors."""
        fitnesses = np.array([ind.fitness for ind in self.population])
        variance = float(np.var(fitnesses))
        genomes = np.array([ind.genome for ind in self.population], dtype=np.float64)
        n = len(genomes)

        sample_size = min(30, n)
        idx = self.rng.choice(n, size=sample_size, replace=False)
        sampled = genomes[idx]
        diffs = sampled[:, np.newaxis, :] - sampled[np.newaxis, :, :]
        dists = np.sqrt(np.sum(np.square(diffs), axis=-1))
        upper = dists[np.triu_indices(sample_size, k=1)]
        mean_dist = float(np.mean(upper)) if len(upper) > 0 else 0.0

        return DiversityMetrics(
            shannon_entropy=0.0,
            pairwise_distance_mean=mean_dist,
            unique_genotypes=n,
            fitness_variance=variance,
            active_species=1,
        )

    def step(self) -> EvolutionDiagnostics:
        """Performs one generational pass of mutation, crossover, and greedy selection."""
        self.generation += 1
        low, high = self.bounds
        next_population: list[Individual] = []

        for i, target in enumerate(self.population):
            # Select 3 distinct random candidates different from target index i
            candidates = [idx for idx in range(self.population_size) if idx != i]
            r1, r2, r3 = self.rng.choice(candidates, size=3, replace=False)

            x_r1 = self.population[r1].genome
            x_r2 = self.population[r2].genome
            x_r3 = self.population[r3].genome

            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            donor = x_r1 + self.f * (x_r2 - x_r3)

            # Bounds clamping
            donor = np.clip(donor, low, high)

            # Crossover: binomial trial vector u
            j_rand = int(self.rng.integers(0, self.dimension))
            crossover_mask = (self.rng.random(size=self.dimension) <= self.cr)
            crossover_mask[j_rand] = True  # Ensure at least one coordinate comes from donor

            trial_genome = np.where(crossover_mask, donor, target.genome)
            trial_fitness = float(self.fitness_function(trial_genome))
            self.evaluations += 1

            # Selection: greedy replacement (maximization convention)
            if trial_fitness >= target.fitness:
                trial_ind = Individual(
                    genome=trial_genome,
                    fitness=trial_fitness,
                    raw_fitness=trial_fitness,
                    individual_id=str(uuid.uuid4())[:8],
                    parent_ids=[target.individual_id, self.population[r1].individual_id],
                    age=0,
                )
                next_population.append(trial_ind)
            else:
                target_copy = target.clone()
                target_copy.age += 1
                next_population.append(target_copy)

        self.population = next_population
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        fitnesses = [ind.fitness for ind in self.population]
        best_fit = float(np.max(fitnesses))
        mean_fit = float(np.mean(fitnesses))
        worst_fit = float(np.min(fitnesses))
        diversity = self.compute_diversity()

        diag = EvolutionDiagnostics(
            generation=self.generation,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            worst_fitness=worst_fit,
            evaluations=self.evaluations,
            diversity=diversity,
            metadata={"best_genome": self.population[0].genome.copy()},
        )
        self.history.append(diag)
        return diag

    def evolve(
        self,
        generations: int = 50,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Executes DE optimization across the specified number of generations."""
        last_diag = None
        for _ in range(generations):
            diag = self.step()
            last_diag = diag
            if step_callback is not None:
                step_callback(diag)
        assert last_diag is not None
        return last_diag

    def get_best(self) -> Individual:
        """Returns the best individual solution found so far."""
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        return self.population[0].clone()
