# Artificial Life and Ecosystem Dynamics (SYS-15, Module M16)

## 1. Theory & Mathematical Definition
Artificial Life (ALife) explores synthetic biology, open-ended evolution, and ecological population dynamics in computational media. Autonomous organisms forage, metabolize resources, reproduce, and undergo Darwinian natural selection in an environment governed by strict demographic and energetic constraints.

## 2. Assumptions
- **Energy Conservation:** Organisms obtain energy solely from environment resource patches and expend energy through basal and locomotory metabolism.
- **Demographic Conservation:** Population change is completely accounted for by births and deaths: no organisms vanish or appear without trace.
- **Genotype-Phenotype Trade-offs:** Morphological or behavioural advantages (e.g., speed, sensory range) incur unavoidable metabolic penalties.

## 3. Equations & Variable Definitions

### Demographic Conservation Invariant
At every discrete simulation step $t$:
$$N_{t+1} \equiv N_t + B_t - D_t$$
where:
- $N_t$: Living population at step $t$
- $B_t$: Total births at step $t$
- $D_t$: Total deaths at step $t$

### Metabolic Burn Invariant
Basal plus locomotory metabolic energy consumption per step:
$$E_{\text{metabolic}} = E_{\text{base}} + c_v \cdot v^2 + c_s \cdot s^{\gamma}$$
where $v$ is movement speed, $s$ is sensory vision range, $c_v = 0.2$, $c_s = 0.1$, and $\gamma = 1.2$.

### Carrying Capacity & Density-Dependent Mortality
Resource regrowth follows constrained logistic replenishment:
$$\Delta R_t = r \cdot R_t \left(1 - \frac{R_t}{K}\right)$$
When population $N_t$ exceeds ecological carrying capacity $K_{\text{pop}}$, density-dependent mortality pressure increases proportionally to $\frac{N_t - K_{\text{pop}}}{K_{\text{pop}}}$.

## 4. Lineage Tracking & Phylogenetic Trees
Every organism maintains an immutable genealogical record tracking parent IDs, generation index, birth step, and death step. The Most Recent Common Ancestor (MRCA) between organisms $A$ and $B$ is found via lowest common ancestor traversal in the directed acyclic graph (DAG) of descent.

## 5. Computational Complexity
- **Time Complexity:** $\mathcal{O}(N_t)$ per simulation step for metabolic update, foraging, reproduction, and death checks.
- **Space Complexity:** $\mathcal{O}(H_{\text{total}})$ where $H_{\text{total}}$ is total historical individuals recorded in the lineage tracker.

## 6. Implementation Notes
- Fully implemented in `packages/autogen/intelligence/evolution/alife.py` and `packages/autogen/agents/evolution/organism_agent.py`.
- Population conservation identity $N_{t+1} \equiv N_t + B_t - D_t$ is asserted at every simulation step.

## 7. Verification Strategy & Test Oracles
- **Conservation Test:** Zero discrepancies across 200 simulation steps: $N_{t+1} - (N_t + B_t - D_t) = 0$.
- **Lineage Consistency:** MRCA of parent and child is guaranteed to be the parent.
- **Trade-off Check:** High-speed organisms strictly deplete energy faster than low-speed organisms under zero-intake conditions.

## 8. Known Limitations
- Continuous open-ended evolution over millions of steps requires persistent pruning of extinct lineages to bound memory.

## 9. References
- Ray, T. S. (1991). *An Approach to the Synthesis of Life*. Artificial Life II, 371-408.
- Langton, C. G. (1989). *Artificial Life*. Addison-Wesley.

