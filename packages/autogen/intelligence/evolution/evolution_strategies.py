"""Evolution Strategies (mu + lambda and mu, lambda) with self-adaptive step sizes (SYS-15, Module M16)."""

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


class EvolutionStrategy:
    """(mu + lambda) and (mu, lambda) Evolution Strategies with log-normal self-adaptive step sizes."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        dimension: int = 10,
        mu: int = 15,          # Number of parents
        lambda_: int = 100,    # Number of offspring
        plus_selection: bool = True,  # True: (mu + lambda), False: (mu, lambda)
        bounds: tuple[float, float] = (-5.12, 5.12),
        initial_sigma: float = 0.5,
        min_sigma: float = 1e-5,
        seed: int = 42,
    ) -> None:
        if not plus_selection and lambda_ < mu:
            raise ValueError(f"(mu, lambda) selection requires lambda ({lambda_}) >= mu ({mu})")

        self.fitness_function = fitness_function
        self.dimension = dimension
        self.mu = mu
        self.lambda_ = lambda_
        self.plus_selection = plus_selection
        self.bounds = bounds
        self.min_sigma = min_sigma
        self.rng = np.random.default_rng(seed)

        # Standard ES learning rates
        self.tau0 = 1.0 / np.sqrt(2.0 * dimension)
        self.tau = 1.0 / np.sqrt(2.0 * np.sqrt(dimension))

        self.generation = 0
        self.evaluations = 0

        # Population of parents: each individual stores genome x and metadata['sigma']
        self.parents: list[Individual] = self._init_parents(initial_sigma)
        self.history: list[EvolutionDiagnostics] = []

    def _init_parents(self, initial_sigma: float) -> list[Individual]:
        parents: list[Individual] = []
        low, high = self.bounds
        for _ in range(self.mu):
            x = self.rng.uniform(low, high, size=self.dimension)
            sigma = np.full(self.dimension, initial_sigma, dtype=np.float64)
            fit = float(self.fitness_function(x))
            self.evaluations += 1

            ind = Individual(
                genome=x,
                fitness=fit,
                raw_fitness=fit,
                individual_id=str(uuid.uuid4())[:8],
                age=0,
                metadata={"sigma": sigma},
            )
            parents.append(ind)
        parents.sort(key=lambda ind: ind.fitness, reverse=True)
        return parents

    def compute_diversity(self) -> DiversityMetrics:
        """Computes diversity of parent population."""
        genomes = np.array([p.genome for p in self.parents], dtype=np.float64)
        fitnesses = np.array([p.fitness for p in self.parents])
        mean_p = np.mean(genomes, axis=0)
        spread = float(np.mean(np.sqrt(np.sum(np.square(genomes - mean_p), axis=1))))

        return DiversityMetrics(
            shannon_entropy=0.0,
            pairwise_distance_mean=spread,
            unique_genotypes=len(self.parents),
            fitness_variance=float(np.var(fitnesses)),
            active_species=1,
        )

    def step(self) -> EvolutionDiagnostics:
        """Executes one generation of mutation, evaluation, and (mu + lambda) / (mu, lambda) selection."""
        self.generation += 1
        low, high = self.bounds
        offspring: list[Individual] = []

        for _ in range(self.lambda_):
            # Select random parent uniformly
            parent_idx = int(self.rng.integers(0, self.mu))
            parent = self.parents[parent_idx]

            parent_x = parent.genome
            parent_sigma = parent.metadata["sigma"]

            # Self-adaptation of step sizes
            xi_0 = self.rng.normal(0.0, 1.0)
            xi_j = self.rng.normal(0.0, 1.0, size=self.dimension)
            child_sigma = parent_sigma * np.exp(self.tau0 * xi_0 + self.tau * xi_j)
            child_sigma = np.maximum(self.min_sigma, child_sigma)

            # Mutate candidate solution
            z = self.rng.normal(0.0, 1.0, size=self.dimension)
            child_x = parent_x + child_sigma * z
            child_x = np.clip(child_x, low, high)

            child_fit = float(self.fitness_function(child_x))
            self.evaluations += 1

            child = Individual(
                genome=child_x,
                fitness=child_fit,
                raw_fitness=child_fit,
                individual_id=str(uuid.uuid4())[:8],
                parent_ids=[parent.individual_id],
                age=0,
                metadata={"sigma": child_sigma},
            )
            offspring.append(child)

        # Selection pool
        if self.plus_selection:
            # (mu + lambda) selection
            candidate_pool = self.parents + offspring
            for p in self.parents:
                p.age += 1
        else:
            # (mu, lambda) selection
            candidate_pool = offspring

        candidate_pool.sort(key=lambda ind: ind.fitness, reverse=True)
        self.parents = candidate_pool[: self.mu]

        fitnesses = [p.fitness for p in self.parents]
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
            metadata={"best_genome": self.parents[0].genome.copy()},
        )
        self.history.append(diag)
        return diag

    def evolve(
        self,
        generations: int = 50,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Runs ES optimization across generations."""
        last_diag = None
        for _ in range(generations):
            diag = self.step()
            last_diag = diag
            if step_callback is not None:
                step_callback(diag)
        assert last_diag is not None
        return last_diag

    def get_best(self) -> Individual:
        """Returns the best solution found."""
        return self.parents[0].clone()
