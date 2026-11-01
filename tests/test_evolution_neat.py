"""Tests for NEAT-lite topological neuroevolution, speciation, and innovation tracking (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.evolution.neat_lite import (
    ConnectionGene,
    InnovationTracker,
    NEATGenome,
    NEATLite,
    NodeGene,
    compute_compatibility_distance,
    neat_crossover,
)


def test_innovation_tracker_consistency() -> None:
    """Verifies that InnovationTracker assigns consistent innovation numbers and node splits."""
    tracker = InnovationTracker()

    inv1 = tracker.get_connection_innovation(0, 2)
    inv2 = tracker.get_connection_innovation(0, 2)
    inv3 = tracker.get_connection_innovation(1, 2)

    assert inv1 == inv2  # Same connection must return identical innovation
    assert inv3 != inv1  # Different connection must get a new innovation

    node_id1, in_inv1, out_inv1 = tracker.get_node_split(0, 2)
    node_id2, in_inv2, out_inv2 = tracker.get_node_split(0, 2)

    assert node_id1 == node_id2
    assert in_inv1 == in_inv2
    assert out_inv1 == out_inv2


def test_compatibility_distance_calculation() -> None:
    """Verifies NEAT compatibility distance delta = c1*E/N + c2*D/N + c3*W_bar."""
    g1 = NEATGenome(num_inputs=2, num_outputs=1)
    g2 = NEATGenome(num_inputs=2, num_outputs=1)

    # Both have matching connection 1 with weights 1.0 and 2.0
    g1.connections[1] = ConnectionGene(in_node=0, out_node=2, weight=1.0, enabled=True, innovation_number=1)
    g2.connections[1] = ConnectionGene(in_node=0, out_node=2, weight=2.0, enabled=True, innovation_number=1)

    # g1 has excess connection 3
    g1.connections[3] = ConnectionGene(in_node=1, out_node=2, weight=0.5, enabled=True, innovation_number=3)

    # delta with c1=1, c2=1, c3=0.4:
    # Matching: inv 1, diff = 1.0 -> c3 * 1.0 = 0.4
    # Excess: inv 3 (count = 1) -> c1 * 1 / 1 = 1.0
    # Expected delta = 1.4
    dist = compute_compatibility_distance(g1, g2, c1=1.0, c2=1.0, c3=0.4)
    assert abs(dist - 1.4) < 1e-6


def test_neat_crossover_alignment() -> None:
    """Verifies that NEAT crossover inherits disjoint/excess genes from the fitter parent."""
    rng = np.random.default_rng(42)

    fitter = NEATGenome(num_inputs=2, num_outputs=1)
    fitter.fitness = 10.0
    fitter.connections[1] = ConnectionGene(0, 2, weight=2.0, enabled=True, innovation_number=1)
    fitter.connections[2] = ConnectionGene(1, 2, weight=3.0, enabled=True, innovation_number=2)

    weaker = NEATGenome(num_inputs=2, num_outputs=1)
    weaker.fitness = 5.0
    weaker.connections[1] = ConnectionGene(0, 2, weight=-1.0, enabled=True, innovation_number=1)
    weaker.connections[4] = ConnectionGene(0, 3, weight=9.9, enabled=True, innovation_number=4)

    child = neat_crossover(fitter, weaker, rng)

    # Child must inherit connection 2 from fitter parent, but NOT connection 4 from weaker parent
    assert 1 in child.connections
    assert 2 in child.connections
    assert 4 not in child.connections


def test_neat_forward_propagation() -> None:
    """Verifies feedforward evaluation of a minimal NEAT graph."""
    genome = NEATGenome(num_inputs=2, num_outputs=1)
    genome.nodes[0] = NodeGene(node_id=0, node_type="input")
    genome.nodes[1] = NodeGene(node_id=1, node_type="input")
    genome.nodes[2] = NodeGene(node_id=2, node_type="output", activation="linear")

    # Connect input 0 to output 2 with weight 0.5; input 1 with weight 1.5
    genome.connections[1] = ConnectionGene(0, 2, weight=0.5, enabled=True, innovation_number=1)
    genome.connections[2] = ConnectionGene(1, 2, weight=1.5, enabled=True, innovation_number=2)

    inputs = np.array([2.0, 3.0])
    output = genome.forward(inputs)

    # Expected: 2.0 * 0.5 + 3.0 * 1.5 = 1.0 + 4.5 = 5.5
    assert len(output) == 1
    assert abs(output[0] - 5.5) < 1e-4


def test_neat_lite_evolution_speciation() -> None:
    """Verifies that NEATLite maintains distinct species and improves objective fitness."""
    def xor_like_objective(g: NEATGenome) -> float:
        # Simple objective: maximize output on [1, 0] and minimize on [0, 0]
        out1 = g.forward(np.array([1.0, 0.0]))[0]
        out0 = g.forward(np.array([0.0, 0.0]))[0]
        return float(out1 - out0)

    neat = NEATLite(
        fitness_function=xor_like_objective,
        num_inputs=2,
        num_outputs=1,
        population_size=20,
        compatibility_threshold=1.5,
        seed=42,
    )

    diag = neat.evolve(generations=15)
    best = neat.get_best()

    assert diag.diversity.active_species >= 1
    assert best.fitness > -10.0
