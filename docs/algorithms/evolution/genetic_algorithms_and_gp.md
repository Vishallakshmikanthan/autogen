# Genetic Algorithms and Genetic Programming (SYS-15, Module M16)

## 1. Theory & Mathematical Definition
Genetic Algorithms (GA) and Genetic Programming (GP) simulate Darwinian evolutionary mechanisms—selection, recombination, and mutation—over populations of candidate solutions. GA operates over discrete or continuous fixed-length chromosomal strings $\mathbf{x} \in \mathcal{S}^L$, whereas GP evolves variable-size hierarchical Abstract Syntax Tree (AST) programs $\mathcal{T} \in \mathbb{T}$.

## 2. Assumptions
- **Fitness Evaluation:** Every candidate individual $x \in \mathcal{P}$ evaluates to a deterministic or bounded stochastic scalar fitness $f(x) \in \mathbb{R}$.
- **Recombination Viability:** Recombining subtrees or chromosomal substrings preserves syntactically valid executable structures (strongly typed or closure property).
- **Parsimony Pressure:** Tree bloat is mitigated through length/depth penalties or size bounds.

## 3. Equations & Variable Definitions
### Roulette Wheel (Fitness-Proportionate) Selection
For population $\{x_1, \dots, x_N\}$ with fitness $f_i$:
$$P(\text{select } x_i) = \frac{f_i - f_{\min} + \epsilon}{\sum_{j=1}^N (f_j - f_{\min} + \epsilon)}$$

### Linear Rank Selection
Sorted by ascending fitness (rank $i \in \{0, \dots, N-1\}$) with selection pressure parameter $s \in [1.0, 2.0]$:
$$P(i) = \frac{2 - s}{N} + \frac{2 i (s - 1)}{N(N - 1)}$$

### GP Tree Parsimony Penalization
$$f_{\text{penalized}}(\mathcal{T}) = f_{\text{raw}}(\mathcal{T}) - \alpha \cdot \text{size}(\mathcal{T})$$
where $\alpha \ge 0$ is the parsimony coefficient and $\text{size}(\mathcal{T})$ is node count.

### Protected Operations
$$\text{protected\_div}(a, b) = \begin{cases} 1.0 & \text{if } |b| < 10^{-6} \\ \frac{a}{b} & \text{otherwise} \end{cases}$$

## 4. Derivation: Monotonicity Invariant with Elitism
Let $E \ge 1$ individuals be preserved identically into generation $t+1$.
$$\max_{x \in \mathcal{P}_{t+1}} f(x) \ge \max_{x \in \text{Elites}_t} f(x) = \max_{x \in \mathcal{P}_t} f(x)$$
Hence, the maximum fitness across generations is strictly monotonically non-decreasing.

## 5. Pseudocode
```python
def genetic_algorithm(f, N, generations, elitism_count):
    P = initialize_population(N)
    evaluate(P, f)
    for t in 1..generations:
        P_next = []
        sort_by_fitness_descending(P)
        P_next.extend(P[:elitism_count]) # Elitism
        while len(P_next) < N:
            p1, p2 = tournament_select(P), tournament_select(P)
            c1, c2 = crossover(p1, p2)
            c1, c2 = mutate(c1), mutate(c2)
            evaluate([c1, c2], f)
            P_next.extend([c1, c2])
        P = P_next[:N]
    return best(P)
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(G \cdot (N \cdot C_{\text{eval}} + N \cdot L))$ where $G$ is generations, $N$ is population size, $C_{\text{eval}}$ is evaluation cost, and $L$ is genome length (or AST node count).
- **Space Complexity:** $\mathcal{O}(N \cdot L)$ for maintaining the active population in memory.

## 7. Implementation Notes
- Fully implemented from scratch in pure NumPy in `ga.py` and `gp.py`.
- Zero external heuristic libraries; all RNG passes through explicit `np.random.Generator`.

## 8. Verification Strategy & Test Oracles
- **OneMax Oracle:** GA with elitism must achieve 100% ones on a 50-bit string within 50 generations.
- **Elitism Monotonicity:** Assert $\max F_{t+1} \ge \max F_t$ for all generations $t$.
- **GP Symbolic Invariance:** GP expression trees must successfully evaluate arithmetic primitives without raising `ZeroDivisionError` or returning `NaN`.

## 9. Known Limitations
- Standard GA with bit-flip mutation is susceptible to Hamming cliffs without Gray coding.
- GP expression trees may expand rapidly without parsimony pressure (bloat phenomenon).

## 10. References
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. MIT Press.
- Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.

