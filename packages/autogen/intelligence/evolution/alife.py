"""Artificial Life ecosystem dynamics, genotype-phenotype mappings, and lineage tracking (SYS-15, Module M16)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autogen.intelligence.evolution.contracts import (
    DiversityMetrics,
    LineageRecord,
)


@dataclass
class OrganismTraits:
    """Phenotypic traits mapped from organism genotype with biological trade-offs."""

    speed: float = 1.0
    vision_range: int = 2
    reproduction_threshold: float = 60.0
    foraging_efficiency: float = 1.0
    metabolic_rate: float = 1.0

    @classmethod
    def from_genotype(cls, genotype: np.ndarray) -> OrganismTraits:
        """Decodes real-valued genotype vector into bounded phenotypic traits."""
        g = np.asarray(genotype, dtype=np.float64).ravel()
        if len(g) < 4:
            padded = np.zeros(4)
            padded[: len(g)] = g
            g = padded

        # Trait bounds mapping
        speed = float(np.clip(1.0 + 0.5 * g[0], 0.5, 2.5))
        vision = int(np.clip(round(2.0 + 1.0 * g[1]), 1, 5))
        repro_thresh = float(np.clip(60.0 + 15.0 * g[2], 30.0, 120.0))
        foraging = float(np.clip(1.0 + 0.3 * g[3], 0.5, 2.0))

        # Biological trade-off: higher speed and higher vision dramatically increase basal metabolism
        metabolic_burn = 0.5 + 0.2 * (speed**2) + 0.1 * (vision**1.2)

        return cls(
            speed=speed,
            vision_range=vision,
            reproduction_threshold=repro_thresh,
            foraging_efficiency=foraging,
            metabolic_rate=float(metabolic_burn),
        )


@dataclass
class ALifeIndividual:
    """An autonomous living organism within the ecosystem simulation."""

    individual_id: str
    genotype: np.ndarray
    traits: OrganismTraits
    energy: float = 50.0
    age: int = 0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    alive: bool = True
    birth_step: int = 0
    death_step: int | None = None
    offspring_count: int = 0


class LineageTracker:
    """Tracks phylogenetic ancestor-descendant DAGs across evolutionary time."""

    def __init__(self) -> None:
        self.records: dict[str, LineageRecord] = {}

    def record_birth(
        self,
        individual_id: str,
        parent_ids: list[str],
        generation: int,
        step: int,
        traits: dict[str, float] | None = None,
        mutations: list[str] | None = None,
    ) -> LineageRecord:
        record = LineageRecord(
            individual_id=individual_id,
            parent_ids=list(parent_ids),
            generation=generation,
            birth_step=step,
            traits=traits or {},
            mutations=mutations or [],
        )
        self.records[individual_id] = record
        return record

    def record_death(self, individual_id: str, step: int, final_fitness: float = 0.0) -> None:
        if individual_id in self.records:
            rec = self.records[individual_id]
            rec.death_step = step
            rec.fitness = final_fitness

    def get_ancestors(self, individual_id: str) -> list[str]:
        """Returns ordered list of all ancestor IDs using breadth-first traversal."""
        ancestors: list[str] = []
        queue = [individual_id]
        visited = {individual_id}

        while queue:
            curr = queue.pop(0)
            if curr in self.records:
                for pid in self.records[curr].parent_ids:
                    if pid not in visited:
                        visited.add(pid)
                        ancestors.append(pid)
                        queue.append(pid)
        return ancestors

    def most_recent_common_ancestor(self, id1: str, id2: str) -> str | None:
        """Finds Most Recent Common Ancestor (MRCA) between two individuals."""
        ancestors1 = set(self.get_ancestors(id1))
        ancestors1.add(id1)

        # BFS on id2's ancestry to find earliest matching ancestor
        queue = [id2]
        visited = {id2}
        while queue:
            curr = queue.pop(0)
            if curr in ancestors1:
                return curr
            if curr in self.records:
                for pid in self.records[curr].parent_ids:
                    if pid not in visited:
                        visited.add(pid)
                        queue.append(pid)
        return None


class EcosystemEngine:
    """Simulates artificial life population dynamics, carrying capacity, and demographic conservation."""

    def __init__(
        self,
        initial_population: int = 50,
        carrying_capacity: int = 200,
        food_growth_rate: float = 40.0,
        max_food: float = 500.0,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.carrying_capacity = carrying_capacity
        self.food_growth_rate = food_growth_rate
        self.max_food = max_food
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.current_food = max_food * 0.5
        self.lineage_tracker = LineageTracker()

        self.organisms: list[ALifeIndividual] = []
        self._init_population(initial_population)

        # Invariant verification tracking
        self.demographic_history: list[dict[str, int]] = []

    def _init_population(self, count: int) -> None:
        for _ in range(count):
            genotype = self.rng.normal(0.0, 0.5, size=4)
            traits = OrganismTraits.from_genotype(genotype)
            org_id = str(uuid.uuid4())[:8]

            org = ALifeIndividual(
                individual_id=org_id,
                genotype=genotype,
                traits=traits,
                energy=50.0,
                age=0,
                generation=0,
                birth_step=0,
            )
            self.organisms.append(org)
            self.lineage_tracker.record_birth(
                individual_id=org_id,
                parent_ids=[],
                generation=0,
                step=0,
                traits={"speed": traits.speed, "metabolism": traits.metabolic_rate},
            )

    def compute_diversity(self) -> DiversityMetrics:
        """Calculates genotypic diversity across living organisms."""
        living = [org for org in self.organisms if org.alive]
        if not living:
            return DiversityMetrics()

        genotypes = np.array([org.genotype for org in living], dtype=np.float64)
        mean_g = np.mean(genotypes, axis=0)
        spread = float(np.mean(np.sqrt(np.sum(np.square(genotypes - mean_g), axis=1))))

        # Shannon entropy of speed trait (binned into 5 bins)
        speeds = [org.traits.speed for org in living]
        hist, _ = np.histogram(speeds, bins=5, range=(0.5, 2.5))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

        return DiversityMetrics(
            shannon_entropy=entropy,
            pairwise_distance_mean=spread,
            unique_genotypes=len(living),
            fitness_variance=float(np.var(speeds)),
            active_species=1,
        )

    def step(self) -> dict[str, Any]:
        """Executes one ecosystem time step with demographic conservation: N_{t+1} == N_t + B_t - D_t."""
        self.step_count += 1
        n_start = len([org for org in self.organisms if org.alive])

        # 1. Regrow food with logistic constraint
        food_regen = self.food_growth_rate * (1.0 - self.current_food / self.max_food)
        self.current_food = min(self.max_food, max(0.0, self.current_food + food_regen))

        # 2. Living organisms forage and burn metabolism
        living = [org for org in self.organisms if org.alive]
        num_living = len(living)

        births = 0
        deaths = 0
        new_offspring: list[ALifeIndividual] = []

        if num_living > 0:
            food_per_capita = self.current_food / num_living
            self.current_food = max(0.0, self.current_food - self.current_food * 0.7)

            for org in living:
                org.age += 1
                # Ingest food multiplied by foraging efficiency
                ingested = food_per_capita * org.traits.foraging_efficiency
                org.energy += ingested

                # Deduct metabolic cost
                org.energy -= org.traits.metabolic_rate

                # Overcrowding / density-dependent mortality pressure
                if num_living > self.carrying_capacity:
                    excess_ratio = (num_living - self.carrying_capacity) / self.carrying_capacity
                    if self.rng.random() < excess_ratio * 0.5:
                        org.energy = 0.0

                # Check death
                if org.energy <= 0.0:
                    org.alive = False
                    org.death_step = self.step_count
                    self.lineage_tracker.record_death(org.individual_id, self.step_count, final_fitness=float(org.age))
                    deaths += 1
                    continue

                # Check reproduction
                if org.energy >= org.traits.reproduction_threshold:
                    # Parent splits energy with child
                    child_energy = org.energy * 0.4
                    org.energy -= child_energy
                    org.offspring_count += 1

                    # Mutate child genotype
                    child_genotype = org.genotype.copy()
                    mask = self.rng.random(size=len(child_genotype)) < self.mutation_rate
                    noise = self.rng.normal(0.0, self.mutation_sigma, size=len(child_genotype))
                    child_genotype[mask] += noise[mask]

                    child_traits = OrganismTraits.from_genotype(child_genotype)
                    child_id = str(uuid.uuid4())[:8]

                    child = ALifeIndividual(
                        individual_id=child_id,
                        genotype=child_genotype,
                        traits=child_traits,
                        energy=child_energy,
                        age=0,
                        generation=org.generation + 1,
                        parent_ids=[org.individual_id],
                        birth_step=self.step_count,
                    )
                    new_offspring.append(child)
                    self.lineage_tracker.record_birth(
                        individual_id=child_id,
                        parent_ids=[org.individual_id],
                        generation=child.generation,
                        step=self.step_count,
                        traits={"speed": child_traits.speed, "metabolism": child_traits.metabolic_rate},
                    )
                    births += 1

        self.organisms.extend(new_offspring)
        # Retain only organisms alive or recently deceased to keep state bounded
        self.organisms = [org for org in self.organisms if org.alive or (self.step_count - (org.death_step or 0)) < 10]

        n_end = len([org for org in self.organisms if org.alive])

        # VERIFY CONSERVATION IDENTITY: N_{t+1} == N_t + B_t - D_t
        assert n_end == n_start + births - deaths, (
            f"Demographic conservation violation! N_start={n_start}, B={births}, D={deaths}, N_end={n_end}"
        )

        demo_record = {
            "step": self.step_count,
            "n_start": n_start,
            "births": births,
            "deaths": deaths,
            "n_end": n_end,
            "food": int(self.current_food),
        }
        self.demographic_history.append(demo_record)

        return {
            "step": self.step_count,
            "population": n_end,
            "births": births,
            "deaths": deaths,
            "food": self.current_food,
            "diversity": self.compute_diversity(),
        }
