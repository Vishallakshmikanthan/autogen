# Hybrid Cognitive Architectures and Component Ablations (SYS-15, Module M17)

## 1. Theory & Mathematical Definition
The Hybrid Cognitive Architecture unifies symbolic, probabilistic, search-based, and learning-based algorithms into a cohesive cognitive system. By explicitly decoupling spatial memory (M04), navigation (M06), hazard belief estimation (M10), high-level action selection (M13/M17), and symbolic runtime verification (M08/M17), the architecture provides complete glass-box explainability and principled empirical ablation.

---

## 2. Assumptions

1. **Dual-Process Deliberation:** Reactive safety checks (System 1) take precedence over long-horizon heuristic path planning (System 2).
2. **Deterministic Seed Tracking:** All random tie-breaking operations derive from independent RNG streams.
3. **Decoupled Faculty Independence:** Cognitive components (memory, belief filter, search, options, safety shield) can be toggled via independent boolean ablation flags without code crashes.

---

## 3. Equations & Mathematical Model

The cognitive decision pipeline resolves chosen action $a^*$ through multi-tier arbitration:
$$a_{\text{proposed}} = \begin{cases}
a_{\text{flee}}, & \text{if } \max_{c \in \text{Sensors}} P(\text{Hazard}(c)) > \tau_{\text{danger}} \\
a_{\text{A*}}, & \text{if } \text{GoalInPlan}(s) \\
a_{\text{explore}}, & \text{otherwise}
\end{cases}$$

The symbolic safety shield functions as a runtime constraint filter:
$$a^* = \begin{cases}
a_{\text{proposed}}, & \text{if } \text{Shield}(s, a_{\text{proposed}}) = \text{SAFE} \\
\arg\max_{a \in \mathcal{A}_{\text{safe}}} U(s, a), & \text{otherwise (Shield Veto)}
\end{cases}$$

---

## 4. Architectural Subsystems

| Subsystem | Underlying Module | Cognitive Role | Ablation Flag |
|---|---|---|---|
| **Spatial Memory** | M04 (Memory) | Tracks visited coordinates & known resource locations | `enable_memory` |
| **A\* Navigation** | M06 (Search) | Manhattan path planning toward memorized food targets | `enable_astar` |
| **Bayesian Belief Filter**| M10 (Probability) | Estimates hazard probability field from partial evidence | `enable_bayes` |
| **Hierarchical Options** | M17 (Hybrid) | Semi-Markov macro-action selection (Forage, Flee, Wander)| `enable_options` |
| **Safety Shield** | M08/M17 (Shield) | Runtime verification vetoing lethal hazard moves | `enable_safety_shield`|

---

## 5. Ablation Arena & Statistical Effect Sizes
To rigorously determine the contribution of each cognitive faculty on held-out worlds, the Ablation Arena computes Cohen's $d$ effect size:
$$d = \frac{\bar{X}_{\text{Full}} - \bar{X}_{\text{Ablated}}}{s_{\text{pooled}}}$$
where:
$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

### Interpretation Scale:
- $|d| < 0.2$: Negligible effect
- $0.2 \le |d| < 0.5$: Small effect
- $0.5 \le |d| < 0.8$: Medium effect
- $|d| \ge 0.8$: Large effect

---

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(V + |\mathcal{A}|)$ per step where $V$ is visible cells.
- **Space Complexity:** $\mathcal{O}(K)$ where $K$ is number of remembered cells.

---

## 7. Implementation Notes
- Pure NumPy implementation in `packages/autogen/agents/hybrid/hybrid_agent.py` and `packages/autogen/research/ablation_arena.py`.
- No external LLM or deep library dependencies.

---

## 8. Verification Strategy & Test Oracles
- **Ablation Significance:** Removing the safety shield must strictly increase hazard mortality rate on hazard-dense environments ($p < 0.05$).
- **Determinism:** Two identical seeds must produce bit-for-bit identical decision traces.

---

## 9. Known Limitations

- **Discrete Action Focus:** Decision arbitration assumes finite discrete action primitives (Movement, Harvest, Rest).
- **Static Shield Rules:** Safety constraints are predefined propositional rules; dynamic synthesis of new invariants during lifetime is not supported.

---

## 10. References / Citations

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed., Chapter 2.4). Pearson.
- Laird, J. E. (2012). *The Soar Cognitive Architecture*. MIT Press.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Lawrence Erlbaum.
