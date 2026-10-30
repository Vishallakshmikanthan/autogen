"""Bayesian Network representation and inference algorithms (SYS-09, Module M10)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import numpy as np

from autogen.intelligence.probabilistic.distributions import DiscreteDistribution


@dataclass(frozen=True, slots=True)
class Factor:
    """Multi-dimensional potential table representing joint or conditional distributions over a set of variables."""

    variables: tuple[str, ...]
    values: np.ndarray  # shape matches cardinalities of variables

    def condition(self, var: str, val_idx: int) -> Factor:
        """Sets variable to a specific observed evidence value, reducing the factor dimension."""
        if var not in self.variables:
            return self
        axis = self.variables.index(var)
        sliced_values = np.take(self.values, indices=val_idx, axis=axis)
        new_vars = tuple(v for i, v in enumerate(self.variables) if i != axis)
        return Factor(variables=new_vars, values=sliced_values)

    def multiply(self, other: Factor) -> Factor:
        """Pointwise tensor multiplication of two factors over the union of their variables."""
        if not self.variables:
            return Factor(other.variables, other.values * float(self.values))
        if not other.variables:
            return Factor(self.variables, self.values * float(other.values))

        # Union of variables preserving order
        union_vars: list[str] = list(self.variables)
        for v in other.variables:
            if v not in union_vars:
                union_vars.append(v)

        def align_tensor(factor: Factor) -> np.ndarray:
            arr = factor.values
            curr_vars = list(factor.variables)
            for v in union_vars:
                if v not in curr_vars:
                    arr = np.expand_dims(arr, axis=-1)
                    curr_vars.append(v)
            axes_order = [curr_vars.index(v) for v in union_vars]
            return np.transpose(arr, axes_order)

        product_values = align_tensor(self) * align_tensor(other)
        return Factor(variables=tuple(union_vars), values=product_values)

    def marginalize(self, var: str) -> Factor:
        """Sums out (marginalizes) a variable from the factor."""
        if var not in self.variables:
            return self
        axis = self.variables.index(var)
        summed_values = np.sum(self.values, axis=axis)
        new_vars = tuple(v for i, v in enumerate(self.variables) if i != axis)
        return Factor(variables=new_vars, values=summed_values)

    def normalize(self) -> Factor:
        """Normalizes the factor values to sum to 1.0."""
        total = float(np.sum(self.values))
        if total <= 0.0:
            raise ValueError("Cannot normalize factor with non-positive sum.")
        return Factor(variables=self.variables, values=self.values / total)


@dataclass(slots=True)
class BayesNetNode:
    """A node in a Bayesian network with its parents and Conditional Probability Table (CPT)."""

    name: str
    parents: tuple[str, ...]
    cpt: np.ndarray  # shape: (*parent_cardinalities, domain_size)
    domain_size: int = 2

    def sample_given_parents(
        self, parent_values: Mapping[str, int], rng: np.random.Generator
    ) -> int:
        """Samples a value from the CPT conditioned on parent assignments."""
        cpt_slice = self.cpt
        for p in self.parents:
            cpt_slice = cpt_slice[parent_values[p]]
        probs = np.asarray(cpt_slice, dtype=np.float64)
        probs = probs / np.sum(probs)
        return int(rng.choice(len(probs), p=probs))

    def prob_given_parents(self, val: int, parent_values: Mapping[str, int]) -> float:
        """Evaluates P(X = val | Parents(X))."""
        cpt_slice = self.cpt
        for p in self.parents:
            cpt_slice = cpt_slice[parent_values[p]]
        return float(cpt_slice[val])


class BayesNet:
    """Directed Acyclic Graph (DAG) representing a Bayesian Network."""

    def __init__(self) -> None:
        self.nodes: dict[str, BayesNetNode] = {}
        self.children: dict[str, list[str]] = {}

    def add_node(self, node: BayesNetNode) -> None:
        """Adds a node to the network. Parents must already exist."""
        for p in node.parents:
            if p not in self.nodes:
                raise ValueError(f"Parent '{p}' of node '{node.name}' does not exist in network.")
        self.nodes[node.name] = node
        self.children.setdefault(node.name, [])
        for p in node.parents:
            self.children.setdefault(p, []).append(node.name)

    def topological_sort(self) -> list[str]:
        """Returns variables in topological order."""
        in_degrees = {name: len(node.parents) for name, node in self.nodes.items()}
        queue = deque([name for name, deg in in_degrees.items() if deg == 0])
        order: list[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for child in self.children.get(curr, []):
                in_degrees[child] -= 1
                if in_degrees[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in Bayesian Network DAG.")
        return order


def enumeration_ask(
    query_var: str,
    evidence: Mapping[str, int],
    bn: BayesNet,
) -> DiscreteDistribution[int]:
    """Exact inference in Bayesian Networks via recursive enumeration."""
    topo_order = bn.topological_sort()
    node = bn.nodes[query_var]
    probs: dict[int, float] = {}

    for val in range(node.domain_size):
        extended_evidence = dict(evidence)
        extended_evidence[query_var] = val
        probs[val] = _enumerate_all(topo_order, extended_evidence, bn)

    return DiscreteDistribution(probs)


def _enumerate_all(
    vars_remaining: Sequence[str],
    evidence: Mapping[str, int],
    bn: BayesNet,
) -> float:
    if not vars_remaining:
        return 1.0

    first, rest = vars_remaining[0], vars_remaining[1:]
    node = bn.nodes[first]

    if first in evidence:
        val = evidence[first]
        p_val = node.prob_given_parents(val, evidence)
        return p_val * _enumerate_all(rest, evidence, bn)
    else:
        total = 0.0
        for val in range(node.domain_size):
            ext_ev = dict(evidence)
            ext_ev[first] = val
            p_val = node.prob_given_parents(val, ext_ev)
            total += p_val * _enumerate_all(rest, ext_ev, bn)
        return total


def variable_elimination(
    query_var: str,
    evidence: Mapping[str, int],
    bn: BayesNet,
    elimination_order: Sequence[str] | None = None,
) -> DiscreteDistribution[int]:
    """Exact inference via Variable Elimination (factor product and marginalization)."""
    # 1. Initialize factors from node CPTs
    factors: list[Factor] = []
    for node in bn.nodes.values():
        var_tuple = node.parents + (node.name,)
        factor = Factor(variables=var_tuple, values=node.cpt)
        # Condition factor on evidence
        for ev_var, ev_val in evidence.items():
            if ev_var in factor.variables:
                factor = factor.condition(ev_var, ev_val)
        factors.append(factor)

    # 2. Determine elimination order for non-query, non-evidence variables
    all_vars = bn.topological_sort()
    if elimination_order is None:
        hidden_vars = [v for v in all_vars if v != query_var and v not in evidence]
    else:
        hidden_vars = [v for v in elimination_order if v != query_var and v not in evidence]

    # 3. Eliminate hidden variables
    for var in hidden_vars:
        factors_with_var = [f for f in factors if var in f.variables]
        factors_without_var = [f for f in factors if var not in f.variables]

        if not factors_with_var:
            continue

        # Compute pointwise product of all factors containing var
        prod = factors_with_var[0]
        for f in factors_with_var[1:]:
            prod = prod.multiply(f)

        # Marginalize out var
        summed = prod.marginalize(var)
        factors = factors_without_var + [summed]

    # 4. Multiply remaining factors containing query_var
    final_factor = factors[0]
    for f in factors[1:]:
        final_factor = final_factor.multiply(f)

    # Normalize to obtain posterior distribution
    normalized = final_factor.normalize()
    vals = normalized.values.flatten()
    return DiscreteDistribution({i: float(v) for i, v in enumerate(vals)})


def rejection_sampling(
    query_var: str,
    evidence: Mapping[str, int],
    bn: BayesNet,
    num_samples: int,
    rng: np.random.Generator,
) -> DiscreteDistribution[int]:
    """Approximate inference via Rejection Sampling."""
    topo_order = bn.topological_sort()
    counts: dict[int, int] = {i: 0 for i in range(bn.nodes[query_var].domain_size)}

    accepted = 0
    for _ in range(num_samples):
        sample: dict[str, int] = {}
        is_consistent = True

        for var_name in topo_order:
            node = bn.nodes[var_name]
            val = node.sample_given_parents(sample, rng)
            sample[var_name] = val

            if var_name in evidence and evidence[var_name] != val:
                is_consistent = False
                break

        if is_consistent:
            counts[sample[query_var]] += 1
            accepted += 1

    if accepted == 0:
        # Fallback to uniform distribution if no samples accepted
        return DiscreteDistribution({i: 1.0 for i in counts})

    return DiscreteDistribution({i: float(cnt) for i, cnt in counts.items()})


def likelihood_weighting(
    query_var: str,
    evidence: Mapping[str, int],
    bn: BayesNet,
    num_samples: int,
    rng: np.random.Generator,
) -> DiscreteDistribution[int]:
    """Approximate inference via Likelihood Weighting."""
    topo_order = bn.topological_sort()
    weights: dict[int, float] = {i: 0.0 for i in range(bn.nodes[query_var].domain_size)}

    for _ in range(num_samples):
        sample: dict[str, int] = {}
        w = 1.0

        for var_name in topo_order:
            node = bn.nodes[var_name]
            if var_name in evidence:
                val = evidence[var_name]
                sample[var_name] = val
                w *= node.prob_given_parents(val, sample)
            else:
                val = node.sample_given_parents(sample, rng)
                sample[var_name] = val

        weights[sample[query_var]] += w

    return DiscreteDistribution(weights)


def gibbs_sampling(
    query_var: str,
    evidence: Mapping[str, int],
    bn: BayesNet,
    num_samples: int,
    rng: np.random.Generator,
) -> DiscreteDistribution[int]:
    """Approximate inference via Markov Chain Monte Carlo (MCMC) Gibbs Sampling."""
    topo_order = bn.topological_sort()
    non_evidence = [v for v in topo_order if v not in evidence]
    node = bn.nodes[query_var]

    # Initialize non-evidence variables randomly
    state: dict[str, int] = dict(evidence)
    for v in non_evidence:
        state[v] = int(rng.choice(bn.nodes[v].domain_size))

    counts: dict[int, int] = {i: 0 for i in range(node.domain_size)}

    for _ in range(num_samples):
        for v in non_evidence:
            curr_node = bn.nodes[v]
            # Sample v conditioned on its Markov Blanket
            cond_probs: list[float] = []
            for candidate_val in range(curr_node.domain_size):
                state[v] = candidate_val
                # Likelihood: P(v | parents) * product_{c in children} P(c | parents(c))
                prob = curr_node.prob_given_parents(candidate_val, state)
                for child_name in bn.children.get(v, []):
                    child_node = bn.nodes[child_name]
                    prob *= child_node.prob_given_parents(state[child_name], state)
                cond_probs.append(prob)

            total_p = sum(cond_probs)
            if total_p > 0:
                normalized_p = [p / total_p for p in cond_probs]
                state[v] = int(rng.choice(curr_node.domain_size, p=normalized_p))

        counts[state[query_var]] += 1

    return DiscreteDistribution({i: float(cnt) for i, cnt in counts.items()})
