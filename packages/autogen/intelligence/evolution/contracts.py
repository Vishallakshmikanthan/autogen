"""Evolutionary computation and Artificial Life contracts, protocols, and data structures (SYS-15, Module M16)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Individual:
    """A single candidate solution possessing a genome, scalar fitness, and evolutionary metadata."""

    genome: Any
    fitness: float = float("-inf")
    raw_fitness: float = float("-inf")
    age: int = 0
    species_id: int = 0
    individual_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> Individual:
        """Creates a deep copy of the individual."""
        import copy
        return Individual(
            genome=copy.deepcopy(self.genome),
            fitness=self.fitness,
            raw_fitness=self.raw_fitness,
            age=self.age,
            species_id=self.species_id,
            individual_id=self.individual_id,
            parent_ids=list(self.parent_ids),
            metadata=dict(self.metadata),
        )


@dataclass
class DiversityMetrics:
    """Genotypic and phenotypic diversity measurements of a population."""

    shannon_entropy: float = 0.0
    pairwise_distance_mean: float = 0.0
    unique_genotypes: int = 0
    fitness_variance: float = 0.0
    active_species: int = 1


@dataclass
class EvolutionDiagnostics:
    """Summary diagnostics produced at each evolutionary generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    evaluations: int
    diversity: DiversityMetrics = field(default_factory=DiversityMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageRecord:
    """Phylogenetic node tracking ancestor-descendant relationships across generations."""

    individual_id: str
    parent_ids: list[str]
    generation: int
    birth_step: int
    death_step: int | None = None
    fitness: float = 0.0
    mutations: list[str] = field(default_factory=list)
    traits: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class FitnessFunction(Protocol):
    """Protocol for objective functions assigning scalar fitness to candidate genomes."""

    def __call__(self, genome: Any) -> float:
        """Evaluates genome and returns scalar fitness (higher is better)."""
        ...


@runtime_checkable
class SelectionOperator(Protocol):
    """Protocol for parental selection mechanisms."""

    def select(
        self,
        population: list[Individual],
        k: int,
        rng: np.random.Generator,
    ) -> list[Individual]:
        """Selects k individuals from the population according to selection pressure."""
        ...


@runtime_checkable
class CrossoverOperator(Protocol):
    """Protocol for genetic recombination of two parents."""

    def crossover(
        self,
        parent1: Any,
        parent2: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        """Produces two offspring genomes through genetic recombination."""
        ...


@runtime_checkable
class MutationOperator(Protocol):
    """Protocol for genome mutation."""

    def mutate(
        self,
        genome: Any,
        rng: np.random.Generator,
    ) -> Any:
        """Returns a mutated copy of the input genome."""
        ...


@runtime_checkable
class EvolutionAlgorithm(Protocol):
    """Protocol for evolutionary optimization algorithms."""

    def step(self) -> EvolutionDiagnostics:
        """Executes a single generational step and returns diagnostic metrics."""
        ...

    def evolve(
        self,
        generations: int = 50,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Runs the evolutionary loop for specified generations."""
        ...

    def get_best(self) -> Individual:
        """Returns the best individual found across all generations."""
        ...
