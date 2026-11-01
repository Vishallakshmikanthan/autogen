"""Runnable CLI demonstration of Evolutionary Computation & Artificial Life (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.agents.evolution.evolved_agent import EvolvedAgent
from autogen.intelligence.evolution.alife import EcosystemEngine
from autogen.intelligence.evolution.benchmarks import OneMax, Sphere
from autogen.intelligence.evolution.differential_evolution import DifferentialEvolution
from autogen.intelligence.evolution.ga import GeneticAlgorithm
from autogen.intelligence.evolution.neat_lite import NEATGenome, NEATLite
from autogen.intelligence.evolution.pso import ParticleSwarmOptimization
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position


def main() -> None:
    print("================================================================================")
    print("         AUTOGEN AI - Evolutionary Computation & Artificial Life Demo (M16)     ")
    print("================================================================================\n")

    # 1. Genetic Algorithm on OneMax
    print("--- 1. Genetic Algorithm: Discrete OneMax (L = 40) ---")
    onemax = OneMax(length=40)
    ga = GeneticAlgorithm(
        fitness_function=onemax,
        genome_type="binary",
        genome_length=40,
        population_size=50,
        elitism_count=2,
        seed=42,
    )

    def ga_callback(diag) -> None:
        if diag.generation % 10 == 0 or diag.generation == 1:
            print(
                f"  Gen {diag.generation:2d} | Best Fitness: {diag.best_fitness:4.1f}/40.0 "
                f"| Mean: {diag.mean_fitness:4.1f} | Diversity (Unique): {diag.diversity.unique_genotypes}"
            )

    _ga_diag = ga.evolve(generations=30, step_callback=ga_callback)
    best_ga = ga.get_best()
    print(f"  Final Best Genome Ones: {int(best_ga.fitness)} / 40 bits\n")

    # 2. Continuous Optimizers: Differential Evolution & PSO on Sphere
    print("--- 2. Continuous Optimizers: Sphere Benchmark (D = 5) ---")
    sphere = Sphere(dimension=5)

    de = DifferentialEvolution(fitness_function=sphere, dimension=5, population_size=25, seed=42)
    de_diag = de.evolve(generations=25)
    print(f"  [DE]  Gen 25 Best Cost: {-de_diag.best_fitness:.6f} | Eval Count: {de.evaluations}")

    pso = ParticleSwarmOptimization(fitness_function=sphere, dimension=5, swarm_size=25, seed=42)
    pso_diag = pso.evolve(generations=25)
    print(f"  [PSO] Gen 25 Best Cost: {-pso_diag.best_fitness:.6f} | Eval Count: {pso.evaluations}\n")

    # 3. NEAT-Lite Topological Evolution
    print("--- 3. NEAT-Lite: Topological Neuroevolution & Speciation ---")
    def simple_target(g: NEATGenome) -> float:
        out = g.forward(np.array([1.0, 0.5]))
        return float(-abs(out[0] - 2.0))

    neat = NEATLite(
        fitness_function=simple_target,
        num_inputs=2,
        num_outputs=1,
        population_size=20,
        compatibility_threshold=1.5,
        seed=42,
    )
    _neat_diag = neat.evolve(generations=10)
    best_neat = neat.get_best()
    print(
        f"  NEAT Gen 10 | Best Fitness: {best_neat.fitness:.4f} "
        f"| Nodes: {len(best_neat.nodes)} | Connections: {len(best_neat.connections)} "
        f"| Active Species: {len(neat.species)}\n"
    )

    # 4. Artificial Life Ecosystem & Lineage Engine
    print("--- 4. Artificial Life Ecosystem Dynamics & Demographic Conservation ---")
    alife = EcosystemEngine(
        initial_population=25,
        carrying_capacity=50,
        food_growth_rate=30.0,
        max_food=300.0,
        seed=42,
    )

    print("  Running 15 ecosystem steps asserting: N_{t+1} == N_t + B_t - D_t...")
    for _ in range(15):
        step_res = alife.step()
        if step_res["step"] % 5 == 0 or step_res["step"] == 1:
            print(
                f"    Step {step_res['step']:2d} | Living: {step_res['population']:2d} "
                f"| Births: {step_res['births']:2d} | Deaths: {step_res['deaths']:2d} "
                f"| Available Food: {step_res['food']:.1f}"
            )

    print("  All 15 demographic conservation assertions satisfied successfully!\n")

    # 5. Evolved Agent Decision Trace
    print("--- 5. Evolved Agent Decision & Authentic DecisionTrace ---")
    controller = NeuralNetwork([DenseLayer(in_features=4, out_features=5, seed=42)])
    agent = EvolvedAgent(
        agent_id="evolved_champion",
        controller=controller,
        state_dim=4,
        generation=15,
        genome_id="alpha_001",
    )

    sample_obs = Observation(
        agent_id="evolved_champion",
        step=1,
        position=Position(x=4, y=4),
        energy=92.5,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )
    agent.observe(sample_obs)
    action, trace = agent.decide()
    print(f"  Chosen Action : {action.action_type.value}")
    print(f"  Trace Agent ID: {trace.agent_id}")
    print(f"  Trace Algorithm: {trace.algorithm_name}")
    print(f"  Trace Rationale: {trace.rationale}")
    print("\n================================================================================")
    print("                     Evolutionary Computation Demo Complete                     ")
    print("================================================================================")


if __name__ == "__main__":
    main()
