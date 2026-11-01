# Continuous Population Optimizers: DE, PSO, and Evolution Strategies (SYS-15, Module M16)

## 1. Theory & Mathematical Definition
Differential Evolution (DE), Particle Swarm Optimization (PSO), and Evolution Strategies (ES) represent three major paradigms for continuous numerical optimization over bounded domains $\mathbf{x} \in \mathbb{R}^D$ without requiring gradient information.

## 2. Assumptions
- Objective function $f: \mathbb{R}^D \to \mathbb{R}$ is computable pointwise.
- Search space is bounded: $\mathbf{x} \in [\mathbf{x}_{\min}, \mathbf{x}_{\max}]$.
- Gradients $\nabla f$ are either unavailable, non-smooth, or contaminated with noise.

## 3. Equations & Variable Definitions

### Differential Evolution ($DE/\text{rand}/1/\text{bin}$)
For target vector $\mathbf{x}_i^{(t)}$:
1. **Mutation:** Donor vector $\mathbf{v}_i = \mathbf{x}_{r_1} + F \cdot (\mathbf{x}_{r_2} - \mathbf{x}_{r_3})$ with distinct indices $r_1, r_2, r_3 \neq i$, where $F \in (0, 2]$.
2. **Crossover:** Trial vector $\mathbf{u}_i$:
$$u_{i,j} = \begin{cases} v_{i,j} & \text{if } \text{rand}_j(0, 1) \le CR \text{ or } j = j_{\text{rand}} \\ x_{i,j} & \text{otherwise} \end{cases}$$
3. **Selection:**
$$\mathbf{x}_i^{(t+1)} = \begin{cases} \mathbf{u}_i & \text{if } f(\mathbf{u}_i) \ge f(\mathbf{x}_i) \\ \mathbf{x}_i^{(t)} & \text{otherwise} \end{cases}$$

### Particle Swarm Optimization (PSO)
For particle $i$ with position $\mathbf{x}_i$, velocity $\mathbf{v}_i$, personal best $\mathbf{p}_i$, and swarm global best $\mathbf{g}^*$:
$$\mathbf{v}_i^{(t+1)} = \omega \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{g}^* - \mathbf{x}_i^{(t)})$$
$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$
where $\omega$ is inertia weight, $c_1, c_2$ are cognitive and social acceleration coefficients, and $r_1, r_2 \sim \mathcal{U}(0, 1)$.

### Evolution Strategies (ES with Self-Adaptive Step Sizes)
Offspring mutation with log-normal step adaptation:
$$\sigma_j' = \sigma_j \cdot \exp\left(\tau_0 \xi_0 + \tau \xi_j\right), \quad \xi_0, \xi_j \sim \mathcal{N}(0, 1)$$
$$\mathbf{x}' = \mathbf{x} + \mathbf{\sigma}' \odot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
where $\tau_0 = \frac{1}{\sqrt{2D}}$ and $\tau = \frac{1}{\sqrt{2\sqrt{D}}}$.

## 4. Pseudocode
```python
def differential_evolution(f, bounds, N, F, CR, generations):
    P = uniform_random_population(N, bounds)
    evaluate(P, f)
    for g in 1..generations:
        P_next = []
        for i, target in enumerate(P):
            r1, r2, r3 = sample_distinct_indices(N, exclude=i, count=3)
            donor = clip(P[r1] + F * (P[r2] - P[r3]), bounds)
            trial = binomial_crossover(target, donor, CR)
            if f(trial) >= f(target):
                P_next.append(trial)
            else:
                P_next.append(target)
        P = P_next
    return best(P)
```

## 5. Computational Complexity
- **Time Complexity:** $\mathcal{O}(G \cdot N \cdot (D + C_{\text{eval}}))$ for all three methods.
- **Space Complexity:** $\mathcal{O}(N \cdot D)$ to store candidate vectors and velocities.

## 6. Implementation Notes
- Pure NumPy vectorization in `differential_evolution.py`, `pso.py`, and `evolution_strategies.py`.
- Velocity clamping and boundary handling prevent divergence.

## 7. Verification Strategy & Test Oracles
- **Sphere Benchmark:** All optimizers must converge from random initialization to $f(\mathbf{x}) > -10^{-3}$ on $D=10$ Sphere.
- **Rastrigin Multimodal Benchmark:** Verifies escape from local minima.
- **Monotonicity Check:** Greedy selection in DE ensures best fitness never decreases.

## 8. Known Limitations
- High dimensionality ($D > 1000$) can degrade convergence without coordinate factorization.
- Parameter tuning ($F, CR, \omega, c_1, c_2$) affects search dynamics.

## 9. References
- Storn, R., & Price, K. (1997). *Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces*. Journal of Global Optimization, 11(4), 341-359.
- Kennedy, J., & Eberhart, R. (1995). *Particle Swarm Optimization*. Proc. IEEE ICNN.
- Schwefel, H. P. (1993). *Evolution and Optimum Seeking*. Wiley.

