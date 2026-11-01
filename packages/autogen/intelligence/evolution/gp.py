"""Genetic Programming with Abstract Syntax Tree (AST) expression trees (SYS-15, Module M16)."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from autogen.intelligence.evolution.contracts import (
    DiversityMetrics,
    EvolutionDiagnostics,
    Individual,
)


@dataclass
class GPNode:
    """Base node in a Genetic Programming abstract syntax tree."""

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates node recursively given input matrix/vector X."""
        raise NotImplementedError

    def size(self) -> int:
        """Returns total number of nodes in this subtree."""
        raise NotImplementedError

    def depth(self) -> int:
        """Returns maximum depth of this subtree."""
        raise NotImplementedError

    def clone(self) -> GPNode:
        """Creates a deep copy of the subtree."""
        return copy.deepcopy(self)


@dataclass
class GPTerminalNode(GPNode):
    """Terminal node holding either a variable index (x[i]) or constant scalar value."""

    var_idx: int | None = None
    constant: float | None = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        n_samples = x_arr.shape[0]

        if self.var_idx is not None:
            return x_arr[:, self.var_idx]
        if self.constant is not None:
            return np.full(n_samples, self.constant, dtype=np.float64)
        return np.zeros(n_samples, dtype=np.float64)

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def __str__(self) -> str:
        if self.var_idx is not None:
            return f"x_{self.var_idx}"
        return f"{self.constant:.2f}"


@dataclass
class GPFunctionNode(GPNode):
    """Function node applying primitive mathematical operators to child subtrees."""

    name: str
    children: list[GPNode] = field(default_factory=list)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        child_vals = [c.evaluate(x) for c in self.children]

        if self.name == "add":
            return np.asarray(child_vals[0] + child_vals[1], dtype=np.float64)
        elif self.name == "sub":
            return np.asarray(child_vals[0] - child_vals[1], dtype=np.float64)
        elif self.name == "mul":
            return np.asarray(np.clip(child_vals[0] * child_vals[1], -1e9, 1e9), dtype=np.float64)
        elif self.name == "div":
            # Protected division
            denom = child_vals[1]
            safe_denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
            res = child_vals[0] / safe_denom
            return np.asarray(np.where(np.abs(denom) < 1e-6, 1.0, np.clip(res, -1e9, 1e9)), dtype=np.float64)
        elif self.name == "sin":
            return np.asarray(np.sin(child_vals[0]), dtype=np.float64)
        elif self.name == "cos":
            return np.asarray(np.cos(child_vals[0]), dtype=np.float64)
        elif self.name == "max":
            return np.asarray(np.maximum(child_vals[0], child_vals[1]), dtype=np.float64)
        elif self.name == "min":
            return np.asarray(np.minimum(child_vals[0], child_vals[1]), dtype=np.float64)
        elif self.name == "if_gt":
            # Ternary: if child0 > child1 then child2 else child3
            c0, c1, c2, c3 = child_vals[0], child_vals[1], child_vals[2], child_vals[3]
            return np.asarray(np.where(c0 > c1, c2, c3), dtype=np.float64)
        else:
            raise ValueError(f"Unknown GP function primitive: {self.name}")

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def __str__(self) -> str:
        if len(self.children) == 2 and self.name in ("add", "sub", "mul", "div"):
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.name]
            return f"({self.children[0]} {sym} {self.children[1]})"
        child_strs = ", ".join(str(c) for c in self.children)
        return f"{self.name}({child_strs})"


class GPTreeGenerator:
    """Generates random AST expression trees using Grow, Full, or Ramped Half-and-Half."""

    BINARY_FUNCTIONS = ["add", "sub", "mul", "div", "max", "min"]
    UNARY_FUNCTIONS = ["sin", "cos"]

    def __init__(
        self,
        num_features: int,
        constants: list[float] | None = None,
        max_depth: int = 4,
    ) -> None:
        self.num_features = num_features
        self.constants = constants or [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        self.max_depth = max_depth

    def random_terminal(self, rng: np.random.Generator) -> GPTerminalNode:
        if rng.random() < 0.6:
            var_idx = int(rng.integers(0, self.num_features))
            return GPTerminalNode(var_idx=var_idx)
        else:
            c = float(rng.choice(self.constants))
            return GPTerminalNode(constant=c)

    def build_tree(
        self,
        current_depth: int,
        max_depth: int,
        method: str,  # "full" or "grow"
        rng: np.random.Generator,
    ) -> GPNode:
        if current_depth >= max_depth:
            return self.random_terminal(rng)

        if method == "grow" and rng.random() < 0.35 and current_depth > 1:
            return self.random_terminal(rng)

        # Pick function
        if rng.random() < 0.8:
            fname = str(rng.choice(self.BINARY_FUNCTIONS))
            c1 = self.build_tree(current_depth + 1, max_depth, method, rng)
            c2 = self.build_tree(current_depth + 1, max_depth, method, rng)
            return GPFunctionNode(name=fname, children=[c1, c2])
        else:
            fname = str(rng.choice(self.UNARY_FUNCTIONS))
            c1 = self.build_tree(current_depth + 1, max_depth, method, rng)
            return GPFunctionNode(name=fname, children=[c1])


def collect_nodes(tree: GPNode) -> list[GPNode]:
    """Flattens all subtrees in prefix order."""
    nodes = [tree]
    if isinstance(tree, GPFunctionNode):
        for child in tree.children:
            nodes.extend(collect_nodes(child))
    return nodes


def replace_node(tree: GPNode, target: GPNode, replacement: GPNode) -> GPNode:
    """Replaces target node with replacement in a cloned tree."""
    if tree is target:
        return replacement.clone()

    if isinstance(tree, GPFunctionNode):
        new_children = []
        for child in tree.children:
            new_children.append(replace_node(child, target, replacement))
        return GPFunctionNode(name=tree.name, children=new_children)

    return tree.clone()


class GeneticProgramming:
    """Genetic Programming engine for symbolic regression and evolved control policies."""

    def __init__(
        self,
        fitness_function: Callable[[GPNode], float],
        num_features: int = 4,
        population_size: int = 50,
        max_depth: int = 4,
        parsimony_coefficient: float = 0.001,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        seed: int = 42,
    ) -> None:
        self.fitness_function = fitness_function
        self.num_features = num_features
        self.population_size = max(4, population_size)
        self.max_depth = max_depth
        self.parsimony = parsimony_coefficient
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

        self.tree_gen = GPTreeGenerator(num_features=num_features, max_depth=max_depth)
        self.generation = 0
        self.evaluations = 0

        self.population: list[Individual] = self._init_population()
        self.history: list[EvolutionDiagnostics] = []

    def _init_population(self) -> list[Individual]:
        population: list[Individual] = []
        # Ramped Half-and-Half initialization
        depths = list(range(2, self.max_depth + 1))
        for i in range(self.population_size):
            d = depths[i % len(depths)]
            method = "full" if (i % 2 == 0) else "grow"
            tree = self.tree_gen.build_tree(1, d, method, self.rng)

            raw_fit = float(self.fitness_function(tree))
            self.evaluations += 1
            penalized_fit = raw_fit - self.parsimony * tree.size()

            ind = Individual(
                genome=tree,
                fitness=penalized_fit,
                raw_fitness=raw_fit,
                individual_id=str(uuid.uuid4())[:8],
                age=0,
            )
            population.append(ind)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        return population

    def _tournament_select(self) -> Individual:
        cand_indices = self.rng.choice(len(self.population), size=min(self.tournament_size, len(self.population)), replace=False)
        candidates: list[Individual] = [self.population[int(i)] for i in cand_indices]
        candidates.sort(key=lambda ind: ind.fitness, reverse=True)
        return candidates[0]

    def _subtree_crossover(self, parent1: GPNode, parent2: GPNode) -> tuple[GPNode, GPNode]:
        nodes1 = collect_nodes(parent1)
        nodes2 = collect_nodes(parent2)

        idx1 = int(self.rng.integers(0, len(nodes1)))
        idx2 = int(self.rng.integers(0, len(nodes2)))
        t1 = nodes1[idx1]
        t2 = nodes2[idx2]

        c1 = replace_node(parent1, t1, t2)
        c2 = replace_node(parent2, t2, t1)

        # Check depth limit
        if c1.depth() > self.max_depth:
            c1 = parent1.clone()
        if c2.depth() > self.max_depth:
            c2 = parent2.clone()
        return c1, c2

    def _subtree_mutation(self, tree: GPNode) -> GPNode:
        nodes = collect_nodes(tree)
        idx = int(self.rng.integers(0, len(nodes)))
        target = nodes[idx]
        sub_tree = self.tree_gen.build_tree(1, 2, "grow", self.rng)
        mutated = replace_node(tree, target, sub_tree)
        if mutated.depth() > self.max_depth:
            return tree.clone()
        return mutated

    def step(self) -> EvolutionDiagnostics:
        """Executes one generation of GP evolution with elitism."""
        self.generation += 1
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        next_gen: list[Individual] = []
        # Elitism: preserve top individual
        elite = self.population[0].clone()
        elite.age += 1
        next_gen.append(elite)

        while len(next_gen) < self.population_size:
            p1 = self._tournament_select()

            if self.rng.random() < self.crossover_rate:
                p2 = self._tournament_select()
                c1_tree, c2_tree = self._subtree_crossover(p1.genome, p2.genome)
            else:
                c1_tree, c2_tree = p1.genome.clone(), p1.genome.clone()

            if self.rng.random() < self.mutation_rate:
                c1_tree = self._subtree_mutation(c1_tree)
            if self.rng.random() < self.mutation_rate:
                c2_tree = self._subtree_mutation(c2_tree)

            for tree in (c1_tree, c2_tree):
                if len(next_gen) < self.population_size:
                    raw_fit = float(self.fitness_function(tree))
                    self.evaluations += 1
                    penalized_fit = raw_fit - self.parsimony * tree.size()
                    child = Individual(
                        genome=tree,
                        fitness=penalized_fit,
                        raw_fitness=raw_fit,
                        individual_id=str(uuid.uuid4())[:8],
                        parent_ids=[p1.individual_id],
                        age=0,
                    )
                    next_gen.append(child)

        self.population = next_gen
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        fitnesses = [ind.fitness for ind in self.population]
        sizes = [ind.genome.size() for ind in self.population]
        best_fit = float(np.max(fitnesses))
        mean_fit = float(np.mean(fitnesses))
        worst_fit = float(np.min(fitnesses))

        diag = EvolutionDiagnostics(
            generation=self.generation,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            worst_fitness=worst_fit,
            evaluations=self.evaluations,
            diversity=DiversityMetrics(fitness_variance=float(np.var(fitnesses))),
            metadata={
                "best_tree": str(self.population[0].genome),
                "mean_size": float(np.mean(sizes)),
            },
        )
        self.history.append(diag)
        return diag

    def evolve(
        self,
        generations: int = 25,
        step_callback: Callable[[EvolutionDiagnostics], None] | None = None,
    ) -> EvolutionDiagnostics:
        """Runs GP evolutionary optimization."""
        last_diag = None
        for _ in range(generations):
            diag = self.step()
            last_diag = diag
            if step_callback is not None:
                step_callback(diag)
        assert last_diag is not None
        return last_diag

    def get_best(self) -> Individual:
        """Returns best GP individual."""
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        return self.population[0].clone()
