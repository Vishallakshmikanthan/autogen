# Grid Belief Filter (2D Spatial Histogram Filter)

## 1. Theory & Mathematical Definition

The **Grid Belief Filter** is a non-parametric histogram filter that tracks the probability distribution $b_t(s)$ of a hidden entity (such as an unobserved predator or hazard) over a discrete 2D spatial grid $\mathcal{S} = \{(x, y) \mid 0 \le x < W, 0 \le y < H\}$.

The belief state $b_t(s) = P(S_t = s \mid o_{1:t})$ evolves via alternating prediction and update phases:

---

## 2. Assumptions

1. **First-Order Markov Property:** Future position $S_{t+1}$ depends only on current position $S_t$, conditional on the environment layout.
2. **Conditional Independence of Observations:** Observation $o_t$ at tick $t$ depends solely on the current true state $S_t$.
3. **Discrete Spatial Grid:** State space $\mathcal{S}$ is partitioned into a finite, discrete 2D lattice.

---

## 3. Equations & Mathematical Model

### Prediction (Motion Model Convolution)
$$
\bar{b}_{t+1}(s') = \sum_{s \in \mathcal{S}} T(s' \mid s) b_t(s)
$$
where $T(s' \mid s)$ models diffusion and autonomous agent locomotion (stay with probability $1 - \epsilon$, or transition to an adjacent neighbor with probability $\epsilon / 4$).

### Correction (Sensor Model Update)
$$
b_{t+1}(s') = \eta \cdot \bar{b}_{t+1}(s') \cdot O(o_t \mid s')
$$
where $\eta = \frac{1}{\sum_{s} \bar{b}_{t+1}(s) O(o_t \mid s)}$ is the normalizing constant, and $O(o_t \mid s')$ evaluates the sensor likelihood given false-positive and false-negative rates.

### Probability & Information Invariants
1. **Probability Invariant**:
   $$\sum_{y=0}^{H-1} \sum_{x=0}^{W-1} b_t(x, y) = 1.0, \quad b_t(x, y) \ge 0$$
2. **Shannon Spatial Entropy**:
   $$H(b) = -\sum_{s \in \mathcal{S}} b(s) \log_2 b(s)$$

---

## 4. Computational Complexity

- **Time Complexity:** $\mathcal{O}(W \cdot H)$ per step when using local 4-connected convolution, or $\mathcal{O}((W \cdot H)^2)$ for dense transition matrices.
- **Space Complexity:** $\mathcal{O}(W \cdot H)$ for storing the probability mass grid.

---

## 5. Implementation Notes

- Normalization handles numerical underflow using `numpy.clip(sum, 1e-12, None)`.
- Transitions near boundary walls enforce reflexive boundary conditions, reflecting probability back into valid grid coordinates.

---

## 6. Verification Strategy & Test Oracles

- **Entropy Reduction Oracle:** True positive observations must monotonically reduce Shannon spatial entropy $H(b_{t+1}) < H(b_t)$.
- **Uniformity under Complete Occlusion:** In absence of observations, belief converges toward the uniform stationary distribution of the motion Markov chain.
- **Total Probability Conservation:** Verified in `tests/unit/test_belief_filter.py` that $|\sum_{s} b(s) - 1.0| < 10^{-7}$ across all update steps.

---

## 7. Known Limitations

- **Scalability with Dimension:** Memory scales linearly with cell count $\mathcal{O}(W \cdot H)$, which becomes expensive for large 3D environments.
- **Discretization Error:** Cannot resolve sub-cell continuous target movements.

---

## 8. References / Citations

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. Chapter 14.
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*, Chapter 2 & 4. MIT Press.
