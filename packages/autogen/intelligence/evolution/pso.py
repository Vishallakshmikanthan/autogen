"""Particle Swarm Optimization (PSO) continuous optimizer (SYS-15, Module M16)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from autogen.intelligence.evolution.contracts import (
    DiversityMetrics,
    EvolutionDiagnostics,
    FitnessFunction,
    Individual,
)


@dataclass
class Particle:
    """A single particle in the swarm maintaining current position, velocity, and personal best."""

    position: np.ndarray
    velocity: np.ndarray
    fitness: float
    best_position: np.ndarray
    best_fitness: float
    particle_id: str


class ParticleSwarmOptimization:
    """Canonical Particle Swarm Optimization implemented from scratch in pure NumPy."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        dimension: int = 10,
        swarm_size: int = 40,
        bounds: tuple[float, float] = (-5.12, 5.12),
        inertia_weight: float = 0.729,
        cognitive_coeff: float = 1.49445,  # c1
        social_coeff: float = 1.49445,     # c2
        v_max_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.fitness_function = fitness_function
        self.dimension = dimension
        self.swarm_size = max(4, swarm_size)
        self.bounds = bounds
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff
        self.rng = np.random.default_rng(seed)

        low, high = bounds
        span = high - low
        self.v_max = span * v_max_ratio

        self.generation = 0
        self.evaluations = 0

        self.particles: list[Particle] = []
        self.global_best_position = np.zeros(dimension)
        self.global_best_fitness = float("-inf")
        self._init_swarm()
        self.history: list[EvolutionDiagnostics] = []

    def _init_swarm(self) -> None:
        low, high = self.bounds
        for _ in range(self.swarm_size):
            pos = self.rng.uniform(low, high, size=self.dimension)
            vel = self.rng.uniform(-self.v_max, self.v_max, size=self.dimension)
            fit = float(self.fitness_function(pos))
            self.evaluations += 1

            p_id = str(uuid.uuid4())[:8]
            particle = Particle(
                position=pos,
                velocity=vel,
                fitness=fit,
                best_position=pos.copy(),
                best_fitness=fit,
                particle_id=p_id,
            )
            self.particles.append(particle)

            if fit > self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_position = pos.copy()

    def compute_diversity(self) -> DiversityMetrics:
        """Computes swarm positional variance and spread."""
        positions = np.array([p.position for p in self.particles])
        fitnesses = np.array([p.fitness for p in self.particles])
        mean_pos = np.mean(positions, axis=0)
        spread = float(np.mean(np.sqrt(np.sum(np.square(positions - mean_pos), axis=1))))

        return DiversityMetrics(
            shannon_entropy=0.0,
            pairwise_distance_mean=spread,
            unique_genotypes=len(self.particles),
            fitness_variance=float(np.var(fitnesses)),
            active_species=1,
        )

    def step(self) -> EvolutionDiagnostics:
        """Updates particle velocities, positions, and personal/global bests."""
        self.generation += 1
        low, high = self.bounds

        for p in self.particles:
            r1 = self.rng.random(size=self.dimension)
            r2 = self.rng.random(size=self.dimension)

            # Velocity update
            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)
            p.velocity = self.w * p.velocity + cognitive + social

            # Velocity clamping
            p.velocity = np.clip(p.velocity, -self.v_max, self.v_max)

            # Position update
            p.position += p.velocity

            # Boundary clamping with velocity dampening
            mask_low = p.position < low
            mask_high = p.position > high
            p.position = np.clip(p.position, low, high)
            p.velocity[mask_low | mask_high] *= -0.5

            # Evaluation
            fit = float(self.fitness_function(p.position))
            self.evaluations += 1
            p.fitness = fit

            # Update personal best
            if fit > p.best_fitness:
                p.best_fitness = fit
                p.best_position = p.position.copy()

            # Update global best
            if fit > self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_position = p.position.copy()

        fitnesses = [p.fitness for p in self.particles]
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
            metadata={"best_genome": self.global_best_position.copy()},
        )
        self.history.append(diag)
        return diag

    def evolve(
        self,
        generations: int = 50,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Runs the PSO swarm iteration across specified generations."""
        last_diag = None
        for _ in range(generations):
            diag = self.step()
            last_diag = diag
            if step_callback is not None:
                step_callback(diag)
        assert last_diag is not None
        return last_diag

    def get_best(self) -> Individual:
        """Returns the global best individual discovered by the swarm."""
        return Individual(
            genome=self.global_best_position.copy(),
            fitness=self.global_best_fitness,
            raw_fitness=self.global_best_fitness,
            individual_id="pso_gbest",
        )
