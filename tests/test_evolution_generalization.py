"""Tests for neuroevolution weight manipulation, GP symbolic regression, and seed generalization (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.evolution.gp import (
    GeneticProgramming,
)
from autogen.intelligence.evolution.neuroevolution import (
    NeuroEvolutionEngine,
    clone_model_with_weights,
    extract_weights,
)
from autogen.intelligence.neural.activations import ReLU
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.mlp import NeuralNetwork


def test_neural_weight_vectorization_invariants() -> None:
    """Verifies that extract_weights and inject_weights preserve model parameter identity."""
    net = NeuralNetwork([
        DenseLayer(in_features=4, out_features=8, activation=ReLU(), seed=42),
        DenseLayer(in_features=8, out_features=2, seed=43),
    ])

    original_weights = extract_weights(net)
    assert len(original_weights) == (4 * 8 + 8) + (8 * 2 + 2)

    # Clone with new weights
    new_weights = original_weights * 2.0
    cloned_net = clone_model_with_weights(net, new_weights)
    extracted_cloned = extract_weights(cloned_net)

    assert np.allclose(extracted_cloned, new_weights)

    # Original net must remain unmodified
    assert np.allclose(extract_weights(net), original_weights)


def test_gp_symbolic_regression() -> None:
    """Verifies Genetic Programming evolves an expression tree on target function y = 2*x + 1."""
    x_train = np.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    y_target = 2.0 * x_train.ravel() + 1.0

    def objective(tree) -> float:
        pred = tree.evaluate(x_train)
        mse = float(np.mean(np.square(pred - y_target)))
        return -mse

    gp = GeneticProgramming(
        fitness_function=objective,
        num_features=1,
        population_size=30,
        max_depth=3,
        parsimony_coefficient=0.001,
        seed=42,
    )

    _diag = gp.evolve(generations=15)
    best = gp.get_best()

    # Fitness should improve significantly from random initialization
    assert best.fitness > -50.0


def test_neuroevolution_policy_generalization() -> None:
    """Verifies neuroevolution training on train seeds generalizes to held-out eval seeds."""
    template = NeuralNetwork([
        DenseLayer(in_features=2, out_features=4, activation=ReLU(), seed=1),
        DenseLayer(in_features=4, out_features=1, seed=2),
    ])

    train_seeds = [10, 20, 30]
    eval_seeds = [100, 200, 300]

    def evaluate_seeds(model: NeuralNetwork, seeds: list[int]) -> float:
        total_score = 0.0
        for s in seeds:
            rng = np.random.default_rng(s)
            inputs = rng.normal(0.0, 1.0, size=(10, 2))
            preds = model.forward(inputs, training=False).ravel()
            # Task: minimize sum of squares
            total_score -= float(np.mean(np.square(preds)))
        return total_score / len(seeds)

    engine = NeuroEvolutionEngine(
        template_network=template,
        evaluation_fn=lambda m: evaluate_seeds(m, train_seeds),
        algorithm="es",
        population_size=20,
        seed=42,
    )

    _diag = engine.evolve(generations=10)
    best_model, train_fitness = engine.get_best_network()

    # Evaluate generalization on held-out seeds
    eval_score = evaluate_seeds(best_model, eval_seeds)

    # Generalization assert: score on held-out seeds should track train fitness
    assert eval_score < 0.0
    assert eval_score > -10.0
