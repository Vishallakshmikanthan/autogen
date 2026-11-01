"""Evolutionary computation and Artificial Life module (SYS-15, Module M16)."""

from __future__ import annotations

from autogen.intelligence.evolution.alife import (
    ALifeIndividual,
    EcosystemEngine,
    LineageTracker,
    OrganismTraits,
)
from autogen.intelligence.evolution.benchmarks import (
    Ackley,
    OneMax,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from autogen.intelligence.evolution.contracts import (
    CrossoverOperator,
    DiversityMetrics,
    EvolutionAlgorithm,
    EvolutionDiagnostics,
    FitnessFunction,
    Individual,
    LineageRecord,
    MutationOperator,
    SelectionOperator,
)
from autogen.intelligence.evolution.differential_evolution import DifferentialEvolution
from autogen.intelligence.evolution.evolution_strategies import EvolutionStrategy
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
from autogen.intelligence.evolution.gp import (
    GeneticProgramming,
    GPFunctionNode,
    GPNode,
    GPTerminalNode,
    GPTreeGenerator,
)
from autogen.intelligence.evolution.neat_lite import (
    ConnectionGene,
    InnovationTracker,
    NEATGenome,
    NEATLite,
    NodeGene,
    compute_compatibility_distance,
    neat_crossover,
)
from autogen.intelligence.evolution.neuroevolution import (
    NeuroEvolutionEngine,
    clone_model_with_weights,
    extract_weights,
    inject_weights,
)
from autogen.intelligence.evolution.pso import Particle, ParticleSwarmOptimization

__all__ = [
    "Ackley",
    "ALifeIndividual",
    "BitFlipMutation",
    "ConnectionGene",
    "CrossoverOperator",
    "DifferentialEvolution",
    "DiversityMetrics",
    "EcosystemEngine",
    "EvolutionAlgorithm",
    "EvolutionDiagnostics",
    "EvolutionStrategy",
    "FitnessFunction",
    "GPFunctionNode",
    "GPNode",
    "GPTerminalNode",
    "GPTreeGenerator",
    "GaussianMutation",
    "GeneticAlgorithm",
    "GeneticProgramming",
    "Individual",
    "InnovationTracker",
    "LineageRecord",
    "LineageTracker",
    "MutationOperator",
    "NEATGenome",
    "NEATLite",
    "NodeGene",
    "NeuroEvolutionEngine",
    "OneMax",
    "OrganismTraits",
    "Particle",
    "ParticleSwarmOptimization",
    "RankSelection",
    "Rastrigin",
    "Rosenbrock",
    "RouletteWheelSelection",
    "SelectionOperator",
    "SinglePointCrossover",
    "Sphere",
    "TournamentSelection",
    "TwoPointCrossover",
    "UniformCrossover",
    "clone_model_with_weights",
    "compute_compatibility_distance",
    "extract_weights",
    "inject_weights",
    "neat_crossover",
]
