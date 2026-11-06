# Algorithm: Synchronous Multi-Agent Emotional Contagion Coupling

> **Family:** Multiagent  
> **Source Module:** Module M22-H (`autogen.intelligence.village.contagion`)  
> **Standard:** AGENTS.md 10-Section Mathematical Specification

---

## 1. Theory & Mathematical Definition
In human communities, emotional expressions (facial affect, vocal tone, body language) transmit nonverbally among individuals in spatial proximity. The Emotional Contagion Engine couples neighboring agents' continuous PAD emotional states through directed interpersonal affinity and mutual attention, while enforcing strict order-independence and capping per-tick updates to prevent runaway panic cascades.

## 2. Assumptions
1. Emotional coupling operates within a sensory threshold radius ($8.0$ tiles).
2. Weight drops exponentially with spatial distance ($L = 6.0$ tiles).
3. Synchronous updates: all pairwise differentials $\Delta P_i$ are computed from step $t$ states prior to state application, guaranteeing permutation order-independence.
4. Total contagion differential per axis is clamped to $[-0.15, 0.15]$ per tick.

## 3. Equations with Variable Definitions
For agent $i$, the continuous emotional adjustment $\Delta \mathbf{P}_i = (\Delta v_i, \Delta a_i, \Delta d_i)$ is given by:

$$\Delta \mathbf{P}_i = \mathrm{clip}\left( \gamma_i \sum_{j \in \mathcal{N}(i)} w_{ij} (\mathbf{P}_j - \mathbf{P}_i), \, -0.15, \, 0.15 \right)$$

Where:
- $\mathcal{N}(i)$: Set of living peers within sensing distance of agent $i$.
- $\mathbf{P}_j = (v_j, a_j, d_j)$: Continuous PAD vector of agent $j$.
- $\gamma_i = 0.08 \cdot (0.60 + 0.40 E_i + 0.40 A_i)$: Trait-derived susceptibility.
- Directed coupling weight:
  $$w_{ij} = e^{-d_{ij}/6.0} \cdot (0.50 + 0.50 \cdot \text{aff}_{ij}) \cdot \chi_{\text{neg}}$$
  Where $\text{aff}_{ij} \in [0, 1]$ is directed affinity from $i$ toward $j$, and $\chi_{\text{neg}} = 1.30$ if $v_j < -0.30$ (negativity bias), and $1.0$ otherwise.

## 4. Derivation Where Appropriate
Negativity bias reflects extensive psychological evidence that distress, panic, and anger spread faster and more intensely than calm or contented moods. Clamping at $\pm 0.15$ ensures that internal OCC appraisal and decay toward baseline remain the dominant long-horizon attractors.

## 5. Pseudocode
```python
def step_contagion(agents, positions, personalities, relationships):
    deltas = {}
    for aid_i, st_i in agents.items():
        sum_delta = np.zeros(3)
        for aid_j, st_j in agents.items():
            if aid_i == aid_j: continue
            w = compute_weight(pos[i], pos[j], aff[i, j], st_j.valence)
            sum_delta += w * (st_j.to_vector() - st_i.to_vector())
        deltas[aid_i] = np.clip(personalities[aid_i].gamma * sum_delta, -0.15, 0.15)
    
    # Synchronous apply
    for aid, st in agents.items():
        st.vector = clip(st.vector + deltas[aid], -1.0, 1.0)
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(N^2)$ pairwise distance and differential evaluations, requiring $<0.1$ ms for $N = 10$ villagers.
- **Space Complexity:** $\mathcal{O}(N)$ buffer to store synchronous intermediate deltas.

## 7. Implementation Notes
Implemented in `autogen.intelligence.village.contagion`. Executed in Phase 2d of `VillageSimulationController.step()`.

## 8. Verification Strategy & Test Oracles
1. **Order Independence Invariant:** Shuffling the dictionary iteration order of agents produces bitwise-identical PAD vectors.
2. **ODE Recovery at $\gamma=0$:** When $\gamma = 0$, contagion deltas are identically zero.
3. **Decay Domination:** In an isolated village cluster with no ongoing stimuli, all agents' PAD vectors decay exponentially toward their personality baselines.

## 9. Known Limitations
- Stoic villager expressiveness masking (suppressing outward expression) is modeled via uniform transmission in Increment 1.

## 10. References / Citations
- Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1993). *Emotional Contagion*. Current Directions in Psychological Science, 2(3), pp. 96-100.
- Barsade, S. G. (2002). *The Ripple Effect: Emotional Contagion and Its Influence on Group Behavior*. Administrative Science Quarterly, 47(4), pp. 644-675.

