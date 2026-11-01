# Neuroevolution and NEAT (SYS-15, Module M16)

## 1. Theory & Mathematical Definition
Neuroevolution evolves artificial neural networks using evolutionary algorithms. It encompasses two primary regimes:
1. **Fixed Topology Neuroevolution:** Optimizes the weight parameters $\mathbf{\theta} \in \mathbb{R}^P$ of a fixed architecture $\mathcal{N}_{\mathbf{\theta}}$.
2. **Topological Neuroevolution (NEAT):** NeuroEvolution of Augmenting Topologies simultaneously evolves synaptic weights and network topology (nodes and directed connections) from minimal structures using historical markings.

## 2. Assumptions
- Agents interact with environments to generate non-differentiable cumulative episodic returns $R \in \mathbb{R}$.
- Fitness sharing prevents premature convergence by protecting innovative topological mutations within niches (species).

## 3. Equations & Variable Definitions

### Weight Vector Extraction & Injection
For a neural network with weight tensors $\{W_l, b_l\}_{l=1}^L$:
$$\mathbf{\theta} = \text{vec}(\{W_l, b_l\}_{l=1}^L) \in \mathbb{R}^P$$

### NEAT Compatibility Distance
Distance $\delta$ between two genomes $g_1, g_2$:
$$\delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \bar{W}$$
where:
- $E$: Count of excess connection genes (innovations exceeding the max innovation of the other genome).
- $D$: Count of disjoint connection genes (innovations missing within the shared range).
- $\bar{W}$: Average weight difference of matching connection genes:
$$\bar{W} = \frac{1}{|M|} \sum_{i \in M} |w_i^{(1)} - w_i^{(2)}|$$
- $N$: Normalizing factor (typically set to 1 for smaller genomes).

### Fitness Sharing (Explicit Niches)
Adjusted fitness $f_i'$ of genome $i$ in species $S_k$:
$$f_i' = \frac{f_i}{|S_k|}$$

## 4. Innovation Tracking & Historical Markings
Every new connection $(u, v)$ receives an incremental global innovation number. When two individuals in the same generation undergo identical topological splits, the innovation tracker assigns identical historical markings, enabling linear-time crossover alignment without graph isomorphism checks.

## 5. Pseudocode
```python
def neat_step(population, species, threshold):
    for individual in population:
        assign_to_species(individual, species, threshold)
    compute_adjusted_fitness(population, species)
    next_population = []
    for s in species:
        champ = s.get_champion()
        next_population.append(champ) # Species elitism
        offspring_count = compute_offspring_quota(s)
        for _ in range(offspring_count):
            p1, p2 = s.select_parents()
            child = neat_crossover(p1, p2)
            mutate_weights_and_topology(child)
            next_population.append(child)
    return next_population
```

## 6. Computational Complexity
- **Feedforward Propagation:** $\mathcal{O}(|V| + |E|)$ where $|V|$ is active nodes and $|E|$ is active connections.
- **Compatibility Distance:** $\mathcal{O}(|E_1| + |E_2|)$ linear innovation set scan.
- **Space Complexity:** $\mathcal{O}(N \cdot (|V| + |E|))$ for storing population genomes.

## 7. Implementation Notes
- Pure NumPy implementation in `neuroevolution.py` and `neat_lite.py`.
- Cycle-safe feedforward evaluation via iterative node relaxation.
- Structural mutations: `add_connection` and `add_node` (split connection).

## 8. Verification Strategy & Test Oracles
- **Innovation Tracking Identity:** Identical splits must produce identical innovation IDs.
- **Speciation Distance:** Genomes with identical topologies but differing weights must produce distance proportional to $c_3 \bar{W}$.
- **Crossover Alignment:** Offspring must correctly inherit disjoint/excess genes from the fitter parent.

## 9. Known Limitations
- High mutation rates can induce unstable topologies.
- Extremely large genomes with cycles require explicit recurrence unrolling.

## 10. References
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies*. Evolutionary Computation, 10(2), 99-127.

