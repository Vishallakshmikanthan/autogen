# Social Capital PageRank & Rational Risk-Reward Ethics Engine

> **Algorithms:** `trust_pagerank`, `social_utility_ethics`  
> **Families:** `MULTIAGENT`, `DECISION`  
> **Implementation:** `packages/autogen/intelligence/village/social_graph.py`, `packages/autogen/intelligence/village/utility_ethics.py`

---

## 1. Theory & Mathematical Definition

In **AUTOGEN Village**, autonomous villagers operate within a finite-resource physical economy where social standing $\mathcal{S}_i(t)$ is an emergent mathematical property of directed interpersonal trust $\mathbf{T}(t) \in [0, 1]^{N \times N}$, and ethical vs. criminal behavior emerges from optimizing a multi-objective expected utility function under sanction risk.

---

## 2. Assumptions

1. **Concave Physiological & Economic Preferences:** Marginal utility of survival vitals ($H_i, T_i, E_i$) and liquid wealth $W_i$ is strictly concave, with a discontinuous survival crisis penalty $\kappa$ below threshold $\tau$.
2. **Asymmetric Trust Dynamics:** Trust is lost faster than it is gained ($\beta_{\text{defect}} > \alpha_{\text{coop}} > 0$).
3. **Ergodic Trust Walk:** With teleportation damping factor $d \in (0, 1)$ and uniform redistribution of dangling nodes, the Markov chain over $\mathbf{\tilde{T}}$ is irreducible and aperiodic, guaranteeing a unique stationary distribution $\mathbf{s} \in \Delta^{N-1}$.
4. **Conditional Independence of Witness Sight:** Given positions $\mathbf{x}_i, \mathbf{x}_j$ and ambient light $L(t)$, individual witness detection events are conditionally independent.

---

## 3. Equations & Variable Definitions

### 3.1 Multi-Objective Utility Function
$$U_i(t) = w_{\text{phys}} \cdot U_{\text{phys}}(H_i, T_i, E_i) + w_{\text{econ}} \cdot U_{\text{econ}}(W_i) + w_{\text{soc}} \cdot \mathcal{S}_i(t) - \mathcal{R}_i(a)$$

* **Physiological Survival Utility:**
  $$U_{\text{phys}}(H_i, T_i, E_i) = \ln(H_i + \epsilon) + \ln(T_i + \epsilon) + \ln(E_i + \epsilon) - \kappa \cdot \mathbb{I}_{[\min(H_i, T_i, E_i) < \tau]}$$
* **Economic Wealth Utility:**
  $$U_{\text{econ}}(W_i) = \sqrt{C_i + \sum_{k} p_k \cdot I_{i, k}}$$
* **Expected Sanction & Status Penalty for Unethical Action $a \in \mathcal{A}_{\text{unethical}}$:**
  $$\mathcal{R}_i(a) = P_{\text{detect}}(a, \mathbf{x}_i, \text{env}) \cdot (1 - 0.65\,\theta_{\text{risk}}) \cdot \left[ C_{\text{penalty}} + \Delta \mathcal{S}_{\text{loss}} \cdot w_{\text{soc}} \right]$$

### 3.2 Directed Trust Update & PageRank Societal Status
$$T_{j \to i}(t+1) = \text{clip}\left((1 - \lambda_{\text{decay}}) T_{j \to i}(t) + \Delta T_{j \to i},\; 0,\; 1\right)$$

$$\Delta T_{j \to i} = \begin{cases}
+\alpha_{\text{coop}} \cdot (1 - T_{j \to i}(t)) & \text{if cooperation / granary tithe} \\
-\beta_{\text{defect}} \cdot T_{j \to i}(t) & \text{if defection / hoarding} \\
-\gamma_{\text{crime}} & \text{if caught stealing}
\end{cases}$$

Global Societal Status $\mathbf{s} = (\mathcal{S}_1, \dots, \mathcal{S}_N)^T$:
$$\mathbf{s} = \frac{1 - d}{N} \mathbf{1} + d \, \mathbf{\tilde{T}}^T \mathbf{s}$$

### 3.3 Witness Line-of-Sight Detection Probability
$$P_{\text{detect}} = 1 - \prod_{j \in \text{Witnesses}} \left(1 - P_{j, \text{sight}}(\mathbf{x}_i, \mathbf{x}_j)\right)$$

---

## 4. Derivation

### 4.1 Starvation-Driven Rational Crime Threshold
Let $a_{\text{steal}}$ restore hunger from $H_i < \tau$ to $H_i + \Delta H \ge \tau$. The survival utility gain is:
$$\Delta U_{\text{gain}} = w_{\text{phys}} \left[\ln\left(\frac{H_i + \Delta H + \epsilon}{H_i + \epsilon}\right) + \kappa\right] + w_{\text{econ}}\Delta U_{\text{econ}}$$
An agent rationally chooses $a_{\text{steal}}$ over legal alternatives when:
$$\Delta U_{\text{gain}} > P_{\text{detect}} \cdot (1 - 0.65\,\theta_{\text{risk}}) \cdot \left(C_{\text{penalty}} + \Delta \mathcal{S}_{\text{loss}} w_{\text{soc}}\right)$$
As $H_i \to 0^+$, $\ln\left(\frac{H_i + \Delta H + \epsilon}{H_i + \epsilon}\right) + \kappa$ grows large enough to exceed any finite expected sanction, proving the emergence of rational survival crime.

---

## 5. Pseudocode

```text
ALGORITHM TrustPageRank(T, d = 0.85, tol = 1e-8, max_iter = 200):
  N <- rows(T)
  T_tilde <- RowNormalizeWithDanglingUniform(T)
  s <- (1 / N) * ones(N)
  FOR k = 1 TO max_iter:
    s_next <- ((1 - d) / N) * ones(N) + d * (T_tilde^T @ s)
    s_next <- s_next / sum(s_next)
    IF ||s_next - s||_1 <= tol:
      RETURN s_next
    s <- s_next
  RETURN s
```

---

## 6. Computational Complexity

* **Trust PageRank (`trust_pagerank`):**
  * **Time Complexity:** $\mathcal{O}(K \cdot N^2)$ where $N$ is the number of villagers and $K \le 200$ is the number of power iterations to reach $\|\mathbf{s}^{(k+1)} - \mathbf{s}^{(k)}\|_1 \le 10^{-8}$.
  * **Space Complexity:** $\mathcal{O}(N^2)$ to store $\mathbf{T}(t)$ and $\mathbf{\tilde{T}}$.
* **Social Utility & Ethics Deliberation (`social_utility_ethics`):**
  * **Time Complexity:** $\mathcal{O}(|\mathcal{A}| + N \cdot R_{\text{vision}})$ per villager per tick (Bresenham raycast across visible witnesses + constant-time evaluation per candidate action).
  * **Space Complexity:** $\mathcal{O}(|\mathcal{A}| + N)$.

---

## 7. Implementation Notes

* Dangling nodes (villagers who distrust everyone, $\sum_i T_{j \to i} = 0$) are replaced with a uniform row $\frac{1}{N}\mathbf{1}^T$ so $\mathbf{\tilde{T}}$ remains strictly row-stochastic.
* Diagonal entries $T_{i \to i}$ are fixed to $0$ so agents cannot inflate their own PageRank status via self-trust.
* The optional `LLMPersonaBridge` operates strictly downstream of `VillagerAgent.decide()`, guaranteeing zero impact on physical simulation coordinates, market clearing, or utility calculations.

---

## 7b. Affective PAD Emotional Dynamics & Gram Sabha Deliberation (§2 & §5)

### Mathematical Model: Continuous 3D PAD Emotional State Vector

Each villager $i$ maintains a continuous affective state vector:
$$\mathbf{E}_i(t) = \begin{bmatrix} v_i(t) \\ a_i(t) \\ d_i(t) \end{bmatrix} \in [-1, 1]^3$$
where $v_i$ is Valence (pleasure/displeasure), $a_i$ is Arousal (alertness/energy), and $d_i$ is Dominance (sense of agency/control).

The discrete-time dynamic update equation is:
$$\mathbf{E}_i(t+1) = \lambda_{\text{decay}} \mathbf{E}_i(t) + (1 - \lambda_{\text{decay}}) \mathbf{E}_{\text{baseline}} + \sum_{k} \mathbf{w}_k \cdot \text{Stimulus}_k(t)$$
constrained by Euclidean/hypercube projection $\|\mathbf{E}_i(t+1)\|_\infty \le 1$.

Discrete emotions emerge from octant clustering in PAD space:
* **Joy / Gratitude:** $v > 0.3, a > 0.0, d > 0.0$
* **Grief / Despair:** $v < -0.3, a < 0.0, d < -0.2$
* **Malice / Hostility:** $v < -0.2, a > 0.2, d > 0.1$
* **Jealousy / Envy:** $v < -0.2, a > 0.1, d < 0.0$
* **Guilt / Remorse:** $v < -0.2, a < 0.2, d < -0.2$
* **Righteous Anger / Authority:** $v < -0.1, a > 0.3, d > 0.3$
* **Calm / Composed:** otherwise

### Gram Sabha Trust-Weighted Consensus

In the Town Assembly, policy adoption or trial verdicts are weighted by each citizen's societal status PageRank $\mathcal{S}_i(t)$:
$$V_{\text{consensus}} = \frac{\sum_{i \in \text{Quorum}} \mathcal{S}_i(t) \cdot y_i}{\sum_{i \in \text{Quorum}} \mathcal{S}_i(t)}$$
where $y_i \in \{+1, -1\}$ is voter $i$'s vote based on ethical and communal utility. If $V_{\text{consensus}} > 0.50$, the policy or verdict is enacted.

---

## 8. Verification Strategy & Test Oracles

1. **Trust Boundedness Oracle:** Verify $\forall i, j: T_{j \to i}(t) \in [0, 1]$ after 1,000 random cooperation, defection, crime, and gossip events.
2. **Stationary Distribution Oracle:** Verify $\sum_{i=1}^N \mathcal{S}_i = 1$ within $10^{-6}$ and compare power-iteration output against exact linear-system solution $(\mathbf{I} - d\mathbf{\tilde{T}}^T)\mathbf{s} = \frac{1-d}{N}\mathbf{1}$ via `numpy.linalg.solve`.
3. **Detection Monotonicity Oracle:** Verify $\frac{\partial P_{\text{detect}}}{\partial \|\mathbf{x}_i - \mathbf{x}_{\text{witness}}\|} \le 0$ along unoccluded rays.
4. **Determinism & LLM Isolation Oracles:** Verify identical SHA-256 `state_hash` across repeated runs with the same seed and under mocked/failing LLM persona decorators.

---

## 9. Known Limitations

* Single-step expected utility lookahead evaluates immediate zone transitions rather than multi-day dynamic programming over seasonal weather cycles.
* Double-auction market clearing processes one commodity at a time rather than combinatorial bundle auctions.

---

## 10. References / Citations

1. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab Technical Report.
2. Becker, G. S. (1968). *Crime and Punishment: An Economic Approach*. Journal of Political Economy, 76(2), 169–217.
3. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.), Chapter 16: Making Simple Decisions.

