# Hybrid Cognitive Architectures and Component Ablations (SYS-15, Module M17)

## 1. Theory & Mathematical Definition
The Hybrid Cognitive Architecture unifies symbolic, probabilistic, search-based, and learning-based algorithms into a cohesive cognitive system. By explicitly decoupling spatial memory (M04), navigation (M06), hazard belief estimation (M10), high-level action selection (M13/M17), and symbolic runtime verification (M08/M17), the architecture provides complete glass-box explainability and principled empirical ablation.

## 2. Architectural Subsystems

| Subsystem | Underlying Module | Cognitive Role | Ablation Flag |
|---|---|---|---|
| **Spatial Memory** | M04 (Memory) | Tracks visited coordinates & known resource locations | `enable_memory` |
| **A\* Navigation** | M06 (Search) | Manhattan path planning toward memorized food targets | `enable_astar` |
| **Bayesian Belief Filter**| M10 (Probability) | Estimates hazard probability field from partial evidence | `enable_bayes` |
| **Hierarchical Options** | M17 (Hybrid) | Semi-Markov macro-action selection (Forage, Flee, Wander)| `enable_options` |
| **Safety Shield** | M08/M17 (Shield) | Runtime verification vetoing lethal hazard moves | `enable_safety_shield`|

## 3. Glass-Box Decision Tracing
At every decision step $t$, `HybridCognitiveAgent` constructs an authentic `DecisionTrace`:
- `proposed_action`: Action nominated by high-level option or A* navigation
- `final_action`: Action approved after safety shield verification
- `shield_vetoed`: Boolean indicating whether safety shield intervened
- `active_option`: Identifier of active hierarchical macro-action
- `hazard_belief_count`: Number of cells with active non-zero hazard probability

## 4. Ablation Arena & Statistical Effect Sizes
To rigorously determine the contribution of each cognitive faculty on held-out worlds, the Ablation Arena computes Cohen's $d$ effect size:
$$d = \frac{\bar{X}_{\text{Full}} - \bar{X}_{\text{Ablated}}}{s_{\text{pooled}}}$$
where:
$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

### Interpretation Scale:
- $|d| < 0.2$: Negligible effect
- $0.2 \le |d| < 0.5$: Small effect
- $0.5 \le |d| < 0.8$: Medium effect
- $|d| \ge 0.8$: Large effect

## 5. Computational Complexity
- **Time Complexity:** $\mathcal{O}(V + |\mathcal{A}|)$ per step where $V$ is visible cells.
- **Space Complexity:** $\mathcal{O}(K)$ where $K$ is number of remembered cells.

## 6. Implementation Notes
- Pure NumPy implementation in `packages/autogen/agents/hybrid/hybrid_agent.py` and `packages/autogen/research/ablation_arena.py`.
- No external LLM or deep library dependencies.

## 7. Verification Strategy & Test Oracles
- **Ablation Significance:** Removing the safety shield must strictly increase hazard mortality rate on hazard-dense environments ($p < 0.05$).
- **Determinism:** Two identical seeds must produce bit-for-bit identical decision traces.

## 8. References
- Laird, J. E. (2012). *The Soar Cognitive Architecture*. MIT Press.
- Anderson, J. R. (1996). *ACT: A simple theory of complex cognition*. American Psychologist.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Lawrence Erlbaum.

