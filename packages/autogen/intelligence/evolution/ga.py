"""Genetic Algorithm engine with modular selection, crossover, and mutation operators (SYS-15, Module M16)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from autogen.intelligence.evolution.contracts import (
    CrossoverOperator,
    DiversityMetrics,
    EvolutionDiagnostics,
    FitnessFunction,
    Individual,
    MutationOperator,
    SelectionOperator,
)


class RouletteWheelSelection:
    """Fitness proportionate selection with linear baseline shifting for non-positive fitness."""

    def select(
        self,
        population: list[Individual],
        k: int,
        rng: np.random.Generator,
    ) -> list[Individual]:
        n = len(population)
        if n == 0 or k <= 0:
            return []

        fitnesses = np.array([ind.fitness for ind in population], dtype=np.float64)
        min_fit = float(np.min(fitnesses))
        # Shift fitness so all values are strictly positive
        shifted = fitnesses - min_fit + 1e-6
        total = np.sum(shifted)
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(n) / n
        else:
            probs = shifted / total

        indices = rng.choice(n, size=k, p=probs)
        return [population[i].clone() for i in indices]


class TournamentSelection:
    """Tournament selection of size k picking the highest fitness individual with probability p."""

    def __init__(self, tournament_size: int = 3, selection_prob: float = 1.0) -> None:
        self.tournament_size = max(1, tournament_size)
        self.selection_prob = selection_prob

    def select(
        self,
        population: list[Individual],
        k: int,
        rng: np.random.Generator,
    ) -> list[Individual]:
        n = len(population)
        if n == 0 or k <= 0:
            return []

        selected: list[Individual] = []
        for _ in range(k):
            candidates_idx = rng.choice(n, size=min(self.tournament_size, n), replace=False)
            candidates = [population[i] for i in candidates_idx]
            candidates.sort(key=lambda ind: ind.fitness, reverse=True)

            if rng.random() <= self.selection_prob:
                selected.append(candidates[0].clone())
            else:
                selected.append(candidates[-1].clone())
        return selected


class RankSelection:
    """Linear rank selection allocating probability linearly from worst (rank 0) to best (rank N-1)."""

    def __init__(self, selection_pressure: float = 1.5) -> None:
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))

    def select(
        self,
        population: list[Individual],
        k: int,
        rng: np.random.Generator,
    ) -> list[Individual]:
        n = len(population)
        if n == 0 or k <= 0:
            return []

        # Sort population by fitness ascending
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        s = self.selection_pressure
        # P(rank i) = (2 - s)/N + 2*i*(s - 1)/(N*(N - 1))
        if n == 1:
            probs = np.array([1.0])
        else:
            ranks = np.arange(n, dtype=np.float64)
            probs = (2.0 - s) / n + (2.0 * ranks * (s - 1.0)) / (n * (n - 1.0))
            probs = np.maximum(0.0, probs)
            probs /= np.sum(probs)

        indices = rng.choice(n, size=k, p=probs)
        return [sorted_pop[i].clone() for i in indices]


class SinglePointCrossover:
    """1-point crossover splitting parental arrays at a single uniform crossover point."""

    def __init__(self, crossover_rate: float = 0.9) -> None:
        self.crossover_rate = crossover_rate

    def crossover(
        self,
        parent1: Any,
        parent2: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        p1 = np.asarray(parent1)
        p2 = np.asarray(parent2)
        if len(p1) <= 1 or rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        point = int(rng.integers(1, len(p1)))
        offspring1 = np.concatenate([p1[:point], p2[point:]])
        offspring2 = np.concatenate([p2[:point], p1[point:]])
        return offspring1, offspring2


class TwoPointCrossover:
    """2-point crossover swapping internal chromosomal segments between two cut points."""

    def __init__(self, crossover_rate: float = 0.9) -> None:
        self.crossover_rate = crossover_rate

    def crossover(
        self,
        parent1: Any,
        parent2: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        p1 = np.asarray(parent1)
        p2 = np.asarray(parent2)
        length = len(p1)
        if length <= 2 or rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        pts = sorted(int(x) for x in rng.choice(np.arange(1, length), size=2, replace=False))
        pt1, pt2 = pts[0], pts[1]

        offspring1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
        offspring2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
        return offspring1, offspring2


class UniformCrossover:
    """Uniform crossover where each locus is independently swapped with probability p_swap."""

    def __init__(self, crossover_rate: float = 0.9, swap_prob: float = 0.5) -> None:
        self.crossover_rate = crossover_rate
        self.swap_prob = swap_prob

    def crossover(
        self,
        parent1: Any,
        parent2: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        p1 = np.asarray(parent1)
        p2 = np.asarray(parent2)
        if rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        mask = rng.random(size=len(p1)) < self.swap_prob
        offspring1 = np.where(mask, p2, p1)
        offspring2 = np.where(mask, p1, p2)
        return offspring1, offspring2


class BitFlipMutation:
    """Discrete binary mutation flipping bit values with probability mutation_rate."""

    def __init__(self, mutation_rate: float = 0.01) -> None:
        self.mutation_rate = mutation_rate

    def mutate(self, genome: Any, rng: np.random.Generator) -> Any:
        arr = np.asarray(genome).copy()
        mask = rng.random(size=len(arr)) < self.mutation_rate
        arr[mask] = 1 - arr[mask]
        return arr


class GaussianMutation:
    """Continuous real-valued mutation adding zero-mean Gaussian perturbation."""

    def __init__(
        self,
        mutation_rate: float = 0.1,
        sigma: float = 0.1,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.bounds = bounds

    def mutate(self, genome: Any, rng: np.random.Generator) -> Any:
        arr = np.asarray(genome, dtype=np.float64).copy()
        mask = rng.random(size=len(arr)) < self.mutation_rate
        noise = rng.normal(0.0, self.sigma, size=len(arr))
        arr[mask] += noise[mask]
        if self.bounds is not None:
            arr = np.clip(arr, self.bounds[0], self.bounds[1])
        return arr


class GeneticAlgorithm:
    """Standard and real-coded Genetic Algorithm with elitism and glass-box diversity tracking."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        genome_type: str = "binary",  # "binary" or "real"
        genome_length: int = 50,
        population_size: int = 100,
        bounds: tuple[float, float] = (-5.12, 5.12),
        selection: SelectionOperator | None = None,
        crossover: CrossoverOperator | None = None,
        mutation: MutationOperator | None = None,
        elitism_count: int = 2,
        seed: int = 42,
    ) -> None:
        self.fitness_function = fitness_function
        self.genome_type = genome_type
        self.genome_length = genome_length
        self.population_size = max(4, population_size)
        self.bounds = bounds
        self.elitism_count = max(0, min(elitism_count, population_size))
        self.rng = np.random.default_rng(seed)

        self.selection = selection or TournamentSelection(tournament_size=3)
        self.crossover = crossover or UniformCrossover(crossover_rate=0.8)
        if mutation is not None:
            self.mutation = mutation
        elif genome_type == "binary":
            self.mutation = BitFlipMutation(mutation_rate=1.0 / max(1, genome_length))
        else:
            self.mutation = GaussianMutation(
                mutation_rate=0.1,
                sigma=(bounds[1] - bounds[0]) * 0.05,
                bounds=bounds,
            )

        self.generation = 0
        self.evaluations = 0
        self.population: list[Individual] = self._init_population()
        self._evaluate_population(self.population)
        self.history: list[EvolutionDiagnostics] = []

    def _init_population(self) -> list[Individual]:
        population: list[Individual] = []
        for _ in range(self.population_size):
            genome: np.ndarray
            if self.genome_type == "binary":
                genome = self.rng.integers(0, 2, size=self.genome_length)
            else:
                genome = self.rng.uniform(self.bounds[0], self.bounds[1], size=self.genome_length)

            ind = Individual(
                genome=genome,
                individual_id=str(uuid.uuid4())[:8],
                age=0,
            )
            population.append(ind)
        return population

    def _evaluate_population(self, individuals: list[Individual]) -> None:
        for ind in individuals:
            if ind.fitness == float("-inf"):
                ind.fitness = float(self.fitness_function(ind.genome))
                ind.raw_fitness = ind.fitness
                self.evaluations += 1

    def compute_diversity(self) -> DiversityMetrics:
        """Computes population diversity: Shannon entropy, pairwise Euclidean distance, and variance."""
        fitnesses = np.array([ind.fitness for ind in self.population])
        variance = float(np.var(fitnesses))

        genomes = np.array([ind.genome for ind in self.population], dtype=np.float64)
        n = len(genomes)

        # Unique genotypes
        unique_count = len(np.unique(genomes, axis=0))

        # Mean pairwise distance
        if n > 1:
            # Subsample pairwise distance if n is large to maintain O(N) evaluation
            sample_size = min(30, n)
            indices = self.rng.choice(n, size=sample_size, replace=False)
            sampled = genomes[indices]
            diffs = sampled[:, np.newaxis, :] - sampled[np.newaxis, :, :]
            dists = np.sqrt(np.sum(np.square(diffs), axis=-1))
            upper_tri = dists[np.triu_indices(sample_size, k=1)]
            mean_dist = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        else:
            mean_dist = 0.0

        # Shannon entropy across fitness distribution (discretized into 10 bins)
        if len(fitnesses) > 1 and np.max(fitnesses) > np.min(fitnesses):
            hist, _ = np.histogram(fitnesses, bins=10)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        return DiversityMetrics(
            shannon_entropy=entropy,
            pairwise_distance_mean=mean_dist,
            unique_genotypes=unique_count,
            fitness_variance=variance,
            active_species=1,
        )

    def step(self) -> EvolutionDiagnostics:
        """Performs one evolutionary generation with selection, crossover, mutation, and elitism."""
        self.generation += 1

        # Sort population descending by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # 1. Elitism: preserve top individuals unmodified
        next_gen: list[Individual] = []
        for i in range(self.elitism_count):
            elite = self.population[i].clone()
            elite.age += 1
            next_gen.append(elite)

        # 2. Reproduce until population capacity is met
        num_needed = self.population_size - len(next_gen)
        # Select parents
        parents = self.selection.select(self.population, k=num_needed, rng=self.rng)

        # Pairwise mating
        offspring_list: list[Individual] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i + 1] if (i + 1) < len(parents) else parents[0]

            child1_genome, child2_genome = self.crossover.crossover(p1.genome, p2.genome, self.rng)
            child1_genome = self.mutation.mutate(child1_genome, self.rng)
            child2_genome = self.mutation.mutate(child2_genome, self.rng)

            c1 = Individual(
                genome=child1_genome,
                individual_id=str(uuid.uuid4())[:8],
                parent_ids=[p1.individual_id, p2.individual_id],
                age=0,
            )
            offspring_list.append(c1)
            if len(offspring_list) < num_needed:
                c2 = Individual(
                    genome=child2_genome,
                    individual_id=str(uuid.uuid4())[:8],
                    parent_ids=[p1.individual_id, p2.individual_id],
                    age=0,
                )
                offspring_list.append(c2)

        self._evaluate_population(offspring_list)
        next_gen.extend(offspring_list)
        self.population = next_gen
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
        """Runs the evolutionary loop for specified generations."""
        last_diag = None
        for _ in range(generations):
            diag = self.step()
            last_diag = diag
            if step_callback is not None:
                step_callback(diag)
        assert last_diag is not None
        return last_diag

    def get_best(self) -> Individual:
        """Returns current best individual in the population."""
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        return self.population[0].clone()
