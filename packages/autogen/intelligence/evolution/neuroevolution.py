"""Neuroevolution: weight optimization of fixed-topology neural networks (SYS-15, Module M16)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from autogen.intelligence.evolution.contracts import (
    EvolutionAlgorithm,
    EvolutionDiagnostics,
    FitnessFunction,
)
from autogen.intelligence.evolution.differential_evolution import DifferentialEvolution
from autogen.intelligence.evolution.evolution_strategies import EvolutionStrategy
from autogen.intelligence.evolution.ga import GeneticAlgorithm
from autogen.intelligence.evolution.pso import ParticleSwarmOptimization
from autogen.intelligence.neural.mlp import NeuralNetwork


class _ObjectiveWrapper:
    def __init__(self, template: NeuralNetwork, eval_fn: Callable[[NeuralNetwork], float]) -> None:
        self.template = template
        self.eval_fn = eval_fn

    def __call__(self, genome: Any) -> float:
        model = clone_model_with_weights(self.template, np.asarray(genome))
        return float(self.eval_fn(model))


def extract_weights(network: NeuralNetwork) -> np.ndarray:
    """Extracts all trainable parameters from network layers into a flat 1D numpy array."""
    flat_parts: list[np.ndarray] = []
    for layer in network.layers:
        if not hasattr(layer, "params"):
            continue
        params = layer.params
        for key in sorted(params.keys()):
            flat_parts.append(params[key].ravel())
    if not flat_parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(flat_parts)


def inject_weights(network: NeuralNetwork, flat_weights: np.ndarray) -> None:
    """Injects flat 1D weights back into the trainable parameters of network layers."""
    offset = 0
    for layer in network.layers:
        if not hasattr(layer, "params"):
            continue
        params = layer.params
        for key in sorted(params.keys()):
            arr = params[key]
            size = arr.size
            slice_w = flat_weights[offset : offset + size]
            arr[...] = slice_w.reshape(arr.shape)
            offset += size


def clone_model_with_weights(template: NeuralNetwork, flat_weights: np.ndarray) -> NeuralNetwork:
    """Creates a deep copy of template network and loads flat weights into it."""
    cloned = copy.deepcopy(template)
    inject_weights(cloned, flat_weights)
    return cloned


class NeuroEvolutionEngine:
    """Orchestrates evolutionary optimization of fixed-topology neural networks."""

    def __init__(
        self,
        template_network: NeuralNetwork,
        evaluation_fn: Callable[[NeuralNetwork], float],
        algorithm: str = "es",  # "ga", "de", "pso", or "es"
        population_size: int = 30,
        bounds: tuple[float, float] = (-3.0, 3.0),
        seed: int = 42,
    ) -> None:
        self.template_network = template_network
        self.evaluation_fn = evaluation_fn
        self.algorithm_name = algorithm.lower()
        self.population_size = population_size
        self.bounds = bounds
        self.seed = seed

        initial_weights = extract_weights(template_network)
        self.dimension = len(initial_weights)

        self.objective_fn: FitnessFunction = _ObjectiveWrapper(self.template_network, self.evaluation_fn)

        self.optimizer: EvolutionAlgorithm
        if self.algorithm_name == "ga":
            self.optimizer = GeneticAlgorithm(
                fitness_function=self.objective_fn,
                genome_type="real",
                genome_length=self.dimension,
                population_size=population_size,
                bounds=bounds,
                elitism_count=2,
                seed=seed,
            )
        elif self.algorithm_name == "de":
            self.optimizer = DifferentialEvolution(
                fitness_function=self.objective_fn,
                dimension=self.dimension,
                population_size=population_size,
                bounds=bounds,
                seed=seed,
            )
        elif self.algorithm_name == "pso":
            self.optimizer = ParticleSwarmOptimization(
                fitness_function=self.objective_fn,
                dimension=self.dimension,
                swarm_size=population_size,
                bounds=bounds,
                seed=seed,
            )
        elif self.algorithm_name == "es":
            self.optimizer = EvolutionStrategy(
                fitness_function=self.objective_fn,
                dimension=self.dimension,
                mu=max(4, population_size // 4),
                lambda_=population_size,
                bounds=bounds,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported neuroevolution optimizer: {algorithm}")

    def step(self) -> EvolutionDiagnostics:
        """Executes one generational update of network weight evolution."""
        return self.optimizer.step()

    def evolve(
        self,
        generations: int = 30,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Runs weight evolution for specified generations."""
        return self.optimizer.evolve(generations=generations, step_callback=step_callback)

    def get_best_network(self) -> tuple[NeuralNetwork, float]:
        """Returns the best neural network and its fitness score."""
        best_ind = self.optimizer.get_best()
        best_net = clone_model_with_weights(self.template_network, best_ind.genome)
        return best_net, best_ind.fitness
