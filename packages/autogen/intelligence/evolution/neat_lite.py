"""NEAT-lite: NeuroEvolution of Augmenting Topologies with speciation and innovation tracking (SYS-15, Module M16)."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from autogen.intelligence.evolution.contracts import (
    DiversityMetrics,
    EvolutionDiagnostics,
)


@dataclass
class NodeGene:
    """A node gene in a NEAT neural network graph."""

    node_id: int
    node_type: str  # "input", "hidden", "output"
    bias: float = 0.0
    activation: str = "tanh"  # "tanh", "sigmoid", "relu", "linear"


@dataclass
class ConnectionGene:
    """A directed, weighted synaptic connection gene with global innovation tracking."""

    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation_number: int


class InnovationTracker:
    """Global innovation number generator memoizing structural mutations across generations."""

    def __init__(self) -> None:
        self.current_innovation = 0
        self.current_node_id = 0
        self.connection_innovations: dict[tuple[int, int], int] = {}
        self.node_splits: dict[tuple[int, int], int] = {}

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        pair = (in_node, out_node)
        if pair not in self.connection_innovations:
            self.current_innovation += 1
            self.connection_innovations[pair] = self.current_innovation
        return self.connection_innovations[pair]

    def get_node_split(self, in_node: int, out_node: int) -> tuple[int, int, int]:
        """Returns (new_node_id, in_to_new_innov, new_to_out_innov) for splitting a connection."""
        pair = (in_node, out_node)
        if pair not in self.node_splits:
            self.current_node_id += 1
            new_node = self.current_node_id
            self.node_splits[pair] = new_node
        new_node = self.node_splits[pair]

        in_innov = self.get_connection_innovation(in_node, new_node)
        out_innov = self.get_connection_innovation(new_node, out_node)
        return new_node, in_innov, out_innov


class NEATGenome:
    """Topological neural network genome supporting structural mutations and speciation."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        genome_id: str = "",
    ) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genome_id = genome_id or str(uuid.uuid4())[:8]

        self.nodes: dict[int, NodeGene] = {}
        self.connections: dict[int, ConnectionGene] = {}  # Keyed by innovation number
        self.fitness: float = float("-inf")
        self.raw_fitness: float = float("-inf")
        self.species_id: int = 0

    def clone(self) -> NEATGenome:
        cloned = NEATGenome(self.num_inputs, self.num_outputs, genome_id=str(uuid.uuid4())[:8])
        cloned.nodes = {nid: copy.deepcopy(node) for nid, node in self.nodes.items()}
        cloned.connections = {inv: copy.deepcopy(conn) for inv, conn in self.connections.items()}
        cloned.fitness = self.fitness
        cloned.raw_fitness = self.raw_fitness
        cloned.species_id = self.species_id
        return cloned

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes feed-forward propagation through the arbitrary directed graph."""
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if len(x_arr) < self.num_inputs:
            padded = np.zeros(self.num_inputs)
            padded[: len(x_arr)] = x_arr
            x_arr = padded

        # Initialize node activations
        node_values: dict[int, float] = {}
        for nid in range(self.num_inputs):
            node_values[nid] = float(x_arr[nid])

        # Active connections
        active_conns = [c for c in self.connections.values() if c.enabled]

        # Iterative relaxation (handles arbitrary feedforward topologies and feed-forward DAGs)
        # 3 passes are sufficient for typical short-depth DAGs
        hidden_and_outputs = [nid for nid in self.nodes.keys() if nid >= self.num_inputs]
        for _ in range(4):
            for nid in hidden_and_outputs:
                node = self.nodes[nid]
                in_conns = [c for c in active_conns if c.out_node == nid]
                if not in_conns and node.node_type == "hidden":
                    continue
                z = node.bias + sum(node_values.get(c.in_node, 0.0) * c.weight for c in in_conns)

                if node.activation == "tanh":
                    val = float(np.tanh(z))
                elif node.activation == "sigmoid":
                    val = float(1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))))
                elif node.activation == "relu":
                    val = float(max(0.0, z))
                else:
                    val = float(z)
                node_values[nid] = val

        output_nodes = [
            nid for nid, node in self.nodes.items() if node.node_type == "output"
        ]
        output_nodes.sort()
        return np.array([node_values.get(nid, 0.0) for nid in output_nodes], dtype=np.float64)


def compute_compatibility_distance(
    g1: NEATGenome,
    g2: NEATGenome,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> float:
    """Calculates NEAT compatibility distance delta = c1*E/N + c2*D/N + c3*W_bar."""
    inv1 = set(g1.connections.keys())
    inv2 = set(g2.connections.keys())

    if not inv1 and not inv2:
        return 0.0

    matching = inv1.intersection(inv2)
    disjoint_and_excess = inv1.symmetric_difference(inv2)

    # Max innovation of g1 and g2
    max_inv1 = max(inv1) if inv1 else 0
    max_inv2 = max(inv2) if inv2 else 0

    excess_count = 0
    disjoint_count = 0
    min_max = min(max_inv1, max_inv2)
    for inv in disjoint_and_excess:
        if inv > min_max:
            excess_count += 1
        else:
            disjoint_count += 1

    # Average weight difference of matching genes
    if matching:
        w_diffs = [
            abs(g1.connections[inv].weight - g2.connections[inv].weight) for inv in matching
        ]
        w_bar = float(np.mean(w_diffs))
    else:
        w_bar = 0.0

    n = max(len(inv1), len(inv2), 1)
    if n < 20:
        n = 1  # Standard NEAT convention: N=1 for small genomes

    delta = (c1 * excess_count) / n + (c2 * disjoint_count) / n + c3 * w_bar
    return float(delta)


def neat_crossover(
    parent1: NEATGenome,
    parent2: NEATGenome,
    rng: np.random.Generator,
) -> NEATGenome:
    """NEAT innovation-aligned crossover: inherits disjoint/excess genes from fitter parent."""
    # Ensure p1 is fitter or equal
    if parent2.fitness > parent1.fitness:
        p1, p2 = parent2, parent1
    else:
        p1, p2 = parent1, parent2

    child = NEATGenome(p1.num_inputs, p1.num_outputs)

    # Nodes: inherit all from p1
    child.nodes = {nid: copy.deepcopy(node) for nid, node in p1.nodes.items()}

    # Connections:
    inv1 = set(p1.connections.keys())
    inv2 = set(p2.connections.keys())
    all_invs = inv1.union(inv2)

    for inv in all_invs:
        if inv in inv1 and inv in inv2:
            # Matching: pick randomly
            chosen_conn = p1.connections[inv] if rng.random() < 0.5 else p2.connections[inv]
            child.connections[inv] = copy.deepcopy(chosen_conn)
            # Re-enable disabled gene with small probability
            if not p1.connections[inv].enabled or not p2.connections[inv].enabled:
                if rng.random() < 0.25:
                    child.connections[inv].enabled = True
        elif inv in inv1:
            # Disjoint or excess from fitter parent
            child.connections[inv] = copy.deepcopy(p1.connections[inv])

    return child


class NEATLite:
    """NEAT-lite engine with speciation, structural mutation, and fitness sharing."""

    def __init__(
        self,
        fitness_function: Callable[[NEATGenome], float],
        num_inputs: int = 4,
        num_outputs: int = 2,
        population_size: int = 40,
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        weight_mutation_rate: float = 0.8,
        add_connection_rate: float = 0.1,
        add_node_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.fitness_function = fitness_function
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = max(4, population_size)
        self.compat_threshold = compatibility_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.weight_mut_rate = weight_mutation_rate
        self.add_conn_rate = add_connection_rate
        self.add_node_rate = add_node_rate
        self.rng = np.random.default_rng(seed)

        self.tracker = InnovationTracker()
        self.tracker.current_node_id = num_inputs + num_outputs - 1

        self.generation = 0
        self.evaluations = 0
        self.species: dict[int, list[NEATGenome]] = {}
        self.species_representatives: dict[int, NEATGenome] = {}
        self.next_species_id = 0

        self.population: list[NEATGenome] = self._init_population()
        self.history: list[EvolutionDiagnostics] = []

    def _init_population(self) -> list[NEATGenome]:
        pop: list[NEATGenome] = []
        for _ in range(self.population_size):
            genome = NEATGenome(self.num_inputs, self.num_outputs)
            # Setup input nodes
            for i in range(self.num_inputs):
                genome.nodes[i] = NodeGene(node_id=i, node_type="input", bias=0.0)
            # Setup output nodes
            for j in range(self.num_outputs):
                out_id = self.num_inputs + j
                genome.nodes[out_id] = NodeGene(node_id=out_id, node_type="output", bias=0.0)

            # Minimal initial connections: each input connects to each output
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    out_id = self.num_inputs + j
                    inv = self.tracker.get_connection_innovation(i, out_id)
                    w = float(self.rng.normal(0.0, 1.0))
                    genome.connections[inv] = ConnectionGene(
                        in_node=i,
                        out_node=out_id,
                        weight=w,
                        enabled=True,
                        innovation_number=inv,
                    )

            fit = float(self.fitness_function(genome))
            self.evaluations += 1
            genome.fitness = fit
            genome.raw_fitness = fit
            pop.append(genome)

        self._speciate(pop)
        return pop

    def _speciate(self, population: list[NEATGenome]) -> None:
        """Groups genomes into species based on compatibility distance."""
        new_species: dict[int, list[NEATGenome]] = {}

        for genome in population:
            assigned = False
            for sid, rep in self.species_representatives.items():
                dist = compute_compatibility_distance(genome, rep, self.c1, self.c2, self.c3)
                if dist < self.compat_threshold:
                    new_species.setdefault(sid, []).append(genome)
                    genome.species_id = sid
                    assigned = True
                    break

            if not assigned:
                self.next_species_id += 1
                sid = self.next_species_id
                self.species_representatives[sid] = genome.clone()
                new_species[sid] = [genome]
                genome.species_id = sid

        # Remove empty species and update representatives
        self.species = {sid: members for sid, members in new_species.items() if members}
        for sid, members in self.species.items():
            rep_idx = int(self.rng.integers(0, len(members)))
            self.species_representatives[sid] = members[rep_idx].clone()

    def _mutate_genome(self, genome: NEATGenome) -> None:
        """Applies weight perturbation, add connection, and add node mutations."""
        # 1. Weight mutation
        for conn in genome.connections.values():
            if self.rng.random() < self.weight_mut_rate:
                if self.rng.random() < 0.9:
                    conn.weight += float(self.rng.normal(0.0, 0.2))
                else:
                    conn.weight = float(self.rng.normal(0.0, 1.0))

        # 2. Add connection mutation
        if self.rng.random() < self.add_conn_rate:
            nodes = list(genome.nodes.keys())
            if len(nodes) >= 2:
                n1_idx = int(self.rng.integers(0, len(nodes)))
                n2_idx = int(self.rng.integers(0, len(nodes)))
                n1 = nodes[n1_idx]
                n2 = nodes[n2_idx]
                # Ensure n1 is not output, and n1 != n2
                if genome.nodes[n1].node_type != "output" and n1 != n2:
                    # Prevent backwards connection into input
                    if genome.nodes[n2].node_type != "input":
                        existing = [
                            c for c in genome.connections.values() if c.in_node == n1 and c.out_node == n2
                        ]
                        if not existing:
                            inv = self.tracker.get_connection_innovation(n1, n2)
                            w = float(self.rng.normal(0.0, 1.0))
                            genome.connections[inv] = ConnectionGene(
                                in_node=n1,
                                out_node=n2,
                                weight=w,
                                enabled=True,
                                innovation_number=inv,
                            )

        # 3. Add node mutation (split connection)
        if self.rng.random() < self.add_node_rate and genome.connections:
            enabled_conns = [c for c in genome.connections.values() if c.enabled]
            if enabled_conns:
                conn_idx = int(self.rng.integers(0, len(enabled_conns)))
                conn_to_split = enabled_conns[conn_idx]
                conn_to_split.enabled = False

                new_node_id, in_inv, out_inv = self.tracker.get_node_split(
                    conn_to_split.in_node, conn_to_split.out_node
                )
                if new_node_id not in genome.nodes:
                    genome.nodes[new_node_id] = NodeGene(
                        node_id=new_node_id,
                        node_type="hidden",
                        bias=0.0,
                    )

                # Connection 1: in_node -> new_node with weight 1.0
                genome.connections[in_inv] = ConnectionGene(
                    in_node=conn_to_split.in_node,
                    out_node=new_node_id,
                    weight=1.0,
                    enabled=True,
                    innovation_number=in_inv,
                )
                # Connection 2: new_node -> out_node with old weight
                genome.connections[out_inv] = ConnectionGene(
                    in_node=new_node_id,
                    out_node=conn_to_split.out_node,
                    weight=conn_to_split.weight,
                    enabled=True,
                    innovation_number=out_inv,
                )

    def step(self) -> EvolutionDiagnostics:
        """Executes one generation of NEAT speciation, reproduction, and fitness sharing."""
        self.generation += 1

        # 1. Compute adjusted fitness: f_i' = (f_i - min) / |species|
        min_fit = min(g.fitness for g in self.population)
        species_adjusted_sums: dict[int, float] = {}

        for sid, members in self.species.items():
            s_size = len(members)
            s_sum = 0.0
            for g in members:
                adj = max(0.001, (g.fitness - min_fit + 0.01)) / s_size
                s_sum += adj
            species_adjusted_sums[sid] = s_sum

        total_adjusted = sum(species_adjusted_sums.values())
        if total_adjusted <= 0 or not np.isfinite(total_adjusted):
            total_adjusted = 1.0

        # 2. Reproduce across species
        next_population: list[NEATGenome] = []

        # Elitism: champion of each species with >= 5 members is preserved
        for _sid, members in self.species.items():
            if len(members) >= 5:
                members.sort(key=lambda g: g.fitness, reverse=True)
                champ = members[0].clone()
                next_population.append(champ)

        # Fill remaining slots proportionally
        slots_remaining = self.population_size - len(next_population)
        for sid, members in self.species.items():
            share = species_adjusted_sums[sid] / total_adjusted
            offspring_count = int(round(share * slots_remaining))
            members.sort(key=lambda g: g.fitness, reverse=True)

            for _ in range(offspring_count):
                if len(next_population) >= self.population_size:
                    break
                p1_pool = members[: max(1, len(members) // 2)]
                p1_idx = int(self.rng.integers(0, len(p1_pool)))
                p1 = p1_pool[p1_idx]
                if len(members) > 1 and self.rng.random() < 0.75:
                    p2_pool = members[: max(1, len(members) // 2)]
                    p2_idx = int(self.rng.integers(0, len(p2_pool)))
                    p2 = p2_pool[p2_idx]
                    child = neat_crossover(p1, p2, self.rng)
                else:
                    child = p1.clone()

                self._mutate_genome(child)
                child.fitness = float(self.fitness_function(child))
                child.raw_fitness = child.fitness
                self.evaluations += 1
                next_population.append(child)

        # If any slots unfilled due to rounding, clone best overall
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        while len(next_population) < self.population_size:
            best_overall = self.population[0].clone()
            self._mutate_genome(best_overall)
            best_overall.fitness = float(self.fitness_function(best_overall))
            best_overall.raw_fitness = best_overall.fitness
            self.evaluations += 1
            next_population.append(best_overall)

        self.population = next_population
        self._speciate(self.population)

        self.population.sort(key=lambda g: g.fitness, reverse=True)
        fitnesses = [g.fitness for g in self.population]
        best_fit = float(np.max(fitnesses))
        mean_fit = float(np.mean(fitnesses))
        worst_fit = float(np.min(fitnesses))

        diag = EvolutionDiagnostics(
            generation=self.generation,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            worst_fitness=worst_fit,
            evaluations=self.evaluations,
            diversity=DiversityMetrics(
                active_species=len(self.species),
                fitness_variance=float(np.var(fitnesses)),
            ),
            metadata={
                "active_species": len(self.species),
                "best_nodes": len(self.population[0].nodes),
                "best_connections": len(self.population[0].connections),
            },
        )
        self.history.append(diag)
        return diag

    def evolve(
        self,
        generations: int = 25,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Runs NEAT evolutionary loop."""
        last_diag = None
        for _ in range(generations):
            diag = self.step()
            last_diag = diag
            if step_callback is not None:
                step_callback(diag)
        assert last_diag is not None
        return last_diag

    def get_best(self) -> NEATGenome:
        """Returns highest fitness genome."""
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        return self.population[0].clone()
