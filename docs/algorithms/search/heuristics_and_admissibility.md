# Grid Heuristics & Admissibility Under Terrain Costs

> **Module:** M06 — Search Algorithms and Problem Solving (SYS-06, Gate 2)

---

## 1. Problem Formulation & Terrain Scaling

On an unweighted grid with unit transition costs $c(s, s') = 1.0$:
- **4-Connected Movement:** Manhattan distance $L_1(s, g) = |x_s - x_g| + |y_s - y_g|$ is admissible and consistent.
- **8-Connected Movement:** Octile distance or Chebyshev distance $L_\infty(s, g) = \max(|x_s - x_g|, |y_s - y_g|)$ is admissible.

### Critical Requirement: Heterogeneous Terrain Step Costs
When cells have variable traversal costs $c(s, a, s') \in [c_{\min}, c_{\max}]$ with $c_{\min} > 0$:
$$h(s) = c_{\min} \cdot \left(|x_s - x_g| + |y_s - y_g|\right)$$

**Proof of Admissibility:**  
Let $\pi^* = (e_1, e_2, \dots, e_k)$ be the optimal path from $s$ to $g$ with length $k$ steps.
The topological shortest distance is $d_{\text{Manhattan}}(s, g) \le k$.
Because every edge $e_i$ costs at least $c_{\min}$:
$$h^*(s) = \sum_{i=1}^k c(e_i) \ge \sum_{i=1}^k c_{\min} = k \cdot c_{\min} \ge c_{\min} \cdot d_{\text{Manhattan}}(s, g) = h(s)$$
Therefore, $h(s) \le h^*(s)$, proving admissibility.

If $c_{\min}$ is omitted (e.g. assumed $1.0$ while grass cost is $0.5$), then $h(s) > h^*(s)$, breaking admissibility and resulting in suboptimal paths.

---

## 2. Distance Formulations

Let $\Delta x = |x_1 - x_2|$ and $\Delta y = |y_1 - y_2|$.

1. **Scaled Manhattan (L1):**
   $$h_{\text{Manhattan}}(p_1, p_2) = c_{\min} \cdot (\Delta x + \Delta y)$$
2. **Scaled Euclidean (L2):**
   $$h_{\text{Euclidean}}(p_1, p_2) = c_{\min} \cdot \sqrt{\Delta x^2 + \Delta y^2}$$
3. **Scaled Chebyshev ($L_\infty$):**
   $$h_{\text{Chebyshev}}(p_1, p_2) = c_{\min} \cdot \max(\Delta x, \Delta y)$$
4. **Scaled Octile:**
   $$h_{\text{Octile}}(p_1, p_2) = c_{\min} \cdot \left(\max(\Delta x, \Delta y) + (\sqrt{2} - 1) \min(\Delta x, \Delta y)\right)$$
