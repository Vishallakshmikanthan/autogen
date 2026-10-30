# Algorithm Specification: Memory Engine & Memory-Augmented Agent

> **Subsystem:** SYS-05 (Memory Engine)  
> **Module:** M04 (Gate 1 Part 4)  
> **Source Ref:** Blueprint §1.2 (Memory), §2.1 (SYS-05); Russell & Norvig §2.4.2

---

## 1. Theory & Mathematical Definition
The Memory Engine provides agents with explicit, retrievable records of past experiences, partitioning cognitive retention into three specialized tiers:
1. **Working Memory ($\mathcal{M}_{wm}$):** A fast, bounded FIFO/LRU buffer of immediate recent observations.
2. **Episodic Memory ($\mathcal{M}_{ep}$):** An experiential transition buffer of discrete interaction tuples:
   $$e_t = \langle t, o_t, a_t, r_t, o_{t+1}, done_t, \text{metadata} \rangle$$
3. **Spatial Memory ($\mathcal{M}_{sp}$):** A persistent spatial hash mapping grid coordinates to remembered environmental attributes:
   $$\mathcal{M}_{sp}(x, y) = \langle \text{terrain}, \text{walkable}, \text{food}_{\text{last}}, \tau_{\text{last}}, c_t(x, y) \rangle$$

---

## 2. Epistemological Distinction: Remembered $\ne$ Believed
A fundamental architectural principle in AUTOGEN is keeping four states distinct:
1. Actual world state $S_t$
2. Agent internal state
3. Agent observation $o_t$
4. Agent belief state $b_t$

### Memory vs. Belief
- **Memory (Retained Evidence):** Memory stores factual evidence of past sensory inputs: *"At tick $\tau = 12$, cell $(4, 5)$ contained 8.0 units of food."* As time elapses, this historical fact remains true about what was sensed, but its epistemic certainty regarding the *current* state degrades.
- **Belief (Probabilistic State Interpretation):** Implemented in M10 (Bayesian state estimation & HMMs). Belief models a probability distribution over the unknown current state given the full evidence history:
  $$b_t(s) = P(S_t = s \mid o_{1:t}, a_{1:t-1})$$
  Memory provides the raw, stale evidence; Belief interprets that evidence through environmental transition models $P(s' \mid s, a)$ to infer whether the food is likely still there.

---

## 3. Mathematical Models & Equations

### 3.1 Staleness & Confidence Decay
Let $t$ be current simulation tick and $\tau_{\text{last}}$ be the tick when coordinate $(x, y)$ was last observed. Staleness is:
$$\Delta t = t - \tau_{\text{last}}$$

- **Exponential Decay:**
  $$c_t(x, y) = c_0 \cdot \exp(-\lambda \Delta t), \quad \lambda > 0$$
  Half-life:
  $$t_{1/2} = \frac{\ln 2}{\lambda}$$
- **Linear Decay:**
  $$c_t(x, y) = \max\left(0, c_0 - \lambda \Delta t\right)$$
- **Monotonicity Invariant:**
  $$\forall \Delta t_1 < \Delta t_2 \implies c(t_1) \ge c(t_2)$$

### 3.2 Retrieval Scoring
- **Proximity Score:**
  $$S_{\text{prox}}(p, p_{\text{target}}) = \frac{1}{1 + \|p - p_{\text{target}}\|_1}$$
- **Vector Cosine Similarity:**
  $$\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **Composite Multi-Criteria Ranking:**
  $$\text{Score}(m) = w_{\text{prox}} S_{\text{prox}} + w_{\text{conf}} c_t + w_{\text{rec}} S_{\text{rec}}$$

---

## 4. Pseudocode

```text
Algorithm: SpatialMemoryUpdate(observation, current_step)
Input: o_t, current step t
Output: updated spatial hash M

1. For each cell in o_t.visible_cells:
     pos <- (cell.position.x, cell.position.y)
     M[pos] <- SpatialMemoryCell(
         position=cell.position,
         terrain=cell.terrain,
         walkable=walkable(cell.terrain),
         last_seen_food=cell.food,
         last_observed_tick=t,
         confidence=1.0
     )

2. For pos in M.keys() not in o_t:
     delta_t <- t - M[pos].last_observed_tick
     M[pos].confidence <- exp(-lambda * delta_t)

3. If len(M) > capacity:
     Evict excess items with lowest confidence (least confident policy)
```

---

## 5. Computational Complexity
- **Working Memory:** $O(1)$ push and access.
- **Episodic Memory:** $O(1)$ push, $O(K)$ retrieval.
- **Spatial Memory:** $O(|V_{\text{cells}}|)$ observation update; $O(1)$ coordinate lookup; $O(M \log M)$ capacity eviction sorting.
- **Space Complexity:** Bounded by strict capacity constants: $O(K_{wm} + K_{ep} + K_{sp})$.

---

## 6. Implementation Notes
- `MemoryAugmentedAgent` decorates any `BaseAgent` (`inner_agent`), preserving existing policy logic while augmenting `DecisionTrace.rationale["memory_used"]`.
- Capacity limits are strictly enforced to prevent memory leaks during extended simulation runs.

---

## 7. Verification Strategy & Test Oracles
1. **FIFO Capacity Oracle:** Inserting $K + N$ elements results in exact size $K$, with the oldest $N$ elements removed.
2. **Monotonicity Verification:** Measured confidence verified against analytical $e^{-\lambda \Delta t}$ calculation.
3. **Cosine Orthogonality:** Confirmed that orthogonal vectors yield exactly $0.0$ and identical vectors yield $1.0$.
4. **Anti-Leakage Security Test:** Proved that unobserved cells never enter spatial memory.

---

## 8. Known Limitations
- When another agent consumes food out of sight, spatial memory retains decaying positive food confidence until re-observed or pruned.
- Cosine similarity retrieval currently operates on dense numerical vectors; symbolic associative retrieval requires M08 (Logic & Knowledge Representation).

---

## 9. References
- Baddeley, A. (1992). "Working memory". *Science*, 255(5044), 556–559.
- Tulving, E. (1983). *Elements of Episodic Memory*. Oxford University Press.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. §2.4.

