# Grid Belief Filter (2D Spatial Histogram Filter)

## 1. Theory & Mathematical Definition

The **Grid Belief Filter** is a non-parametric histogram filter that tracks the probability distribution $b_t(s)$ of a hidden entity (such as an unobserved predator or hazard) over a discrete 2D spatial grid $\mathcal{S} = \{(x, y) \mid 0 \le x < W, 0 \le y < H\}$.

The belief state $b_t(s) = P(S_t = s \mid o_{1:t})$ evolves via alternating prediction and update phases:

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

---

## 2. Invariants & Information Metrics

1. **Probability Invariant**:
   $$\sum_{y=0}^{H-1} \sum_{x=0}^{W-1} b_t(x, y) = 1.0, \quad b_t(x, y) \ge 0$$
2. **Shannon Spatial Entropy**:
   $$H(b) = -\sum_{s \in \mathcal{S}} b(s) \log_2 b(s)$$
3. **Sensor Quality Invariant**:
   - A perfect sensor causes belief to collapse to a single point mass ($H(b) \to 0$).
   - A totally uninformative sensor causes the belief to evolve solely according to the motion diffusion model ($H(b) \to \text{max}$).

---

## 3. References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*, Chapter 2 & 4. MIT Press.
