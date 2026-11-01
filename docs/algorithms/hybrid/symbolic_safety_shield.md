# Symbolic Safety Shielding (SYS-15, Module M17)

## 1. Theory & Mathematical Definition
Safety Shielding provides runtime verification and formal safety guarantees for learning and planning agents. The shield sits between the agent's policy $\pi(s)$ and the environment transition engine, intercepting proposed actions and substituting safe fallback actions whenever a formal safety specification $\phi_{\text{safe}}$ would be violated.

## 2. Assumptions
- Safe state predicate $\phi_{\text{safe}}(s)$ is decidable in $\mathcal{O}(1)$ time.
- The environment supports at least one safe fallback action (e.g., `STAY` or stepping away from hazards).
- Hazard belief maps quantify Bayesian posterior probability $P(\text{hazard} \mid \text{evidence})$.

## 3. Equations & Variable Definitions

### Shield Action Mapping
For proposed action $a_{\text{prop}} \sim \pi(s)$ and target predicted position $s' = \hat{T}(s, a_{\text{prop}})$:
$$a_{\text{final}} = \begin{cases} a_{\text{prop}} & \text{if } \phi_{\text{safe}}(s') = \text{True} \land P(\text{Hazard}(s')) < \theta_{\text{hazard}} \\ a^*_{\text{fallback}} & \text{otherwise} \end{cases}$$
where:
$$a^*_{\text{fallback}} = \arg\min_{a \in \mathcal{A}_{\text{legal}}(s)} P(\text{Hazard}(\hat{T}(s, a)))$$

### Safety Violation Trace
Every intervention generates an immutable `ShieldViolationRecord`:
$$\mathcal{V} = \langle t, a_{\text{prop}}, a_{\text{final}}, \text{rule}, P_{\text{hazard}}(s') \rangle$$
guaranteeing full explainability in the agent's `DecisionTrace`.

## 4. Property Verification & Safety Invariant
For any trajectory $\{s_0, a_0, s_1, a_1, \dots\}$ executed under active shielding:
$$\forall t \ge 0, \quad s_t \in \mathcal{S}_{\text{safe}}$$
Zero permitted transitions into known lethal hazard cells across all states.

## 5. Computational Complexity
- **Time Complexity:** $\mathcal{O}(|\mathcal{A}|)$ candidate verification steps per decision tick.
- **Space Complexity:** $\mathcal{O}(1)$ working memory per tick, $\mathcal{O}(H)$ for stored violation history.

## 6. Implementation Notes
- Fully implemented in `packages/autogen/intelligence/hybrid/safety_shield.py`.
- Evaluates grid boundary collisions, visible hazard cells, and Bayesian hazard probability maps.

## 7. Verification Strategy & Test Oracles
- **Property Test:** Over 50,000 synthetic random and adversarial states, the shield must satisfy:
$$\text{Safety Violations} \equiv 0$$
- **Minimal Interference:** Safe actions must pass through completely unmodified.

## 8. Known Limitations
- When all directional actions lead to lethal hazards, the shield falls back to minimum-risk action or `STAY`.

## 9. References
- Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., & Topcu, U. (2018). *Safe Reinforcement Learning via Shielding*. AAAI.

