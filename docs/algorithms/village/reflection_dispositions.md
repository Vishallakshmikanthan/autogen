# Algorithm: Nightly Episodic Reflection, Grudge Consolidation & Forgiveness

> **Family:** Decision  
> **Source Module:** Module M22-H (`autogen.intelligence.village.reflection`)  
> **Standard:** AGENTS.md 10-Section Mathematical Specification

---

## 1. Theory & Mathematical Definition
Raw episodic memories fade rapidly without consolidation. Inspired by sleep-time memory consolidation and Park et al.'s Generative Agent architecture, the Reflection Engine consolidates daily episodic experiences into durable, explainable Dispositions toward specific villagers. Dispositions capture net appraisal valence, durable grudges from past harms, fondness from past assistance, statistical confidence, and explicit citing evidence IDs.

## 2. Assumptions
1. Reflection occurs during rest/sleep cycles (e.g., `REST_HUT` or nightfall $t \ge 22.0$ or $t \le 5.0$).
2. Negative harm events impact grudges more severely than positive events impact fondness (asymmetry of moral judgment).
3. Grudges decay exponentially over ticks, modulated by Neuroticism and Agreeableness.
4. Memory remains strictly bounded by capping citing evidence IDs to the most recent 16 items.

## 3. Equations with Variable Definitions
Given unreflected memory episodes for peer $j$:

### 3.1 Valence Integration
$$v_{ij} \leftarrow 0.85 \cdot v_{ij} + 0.15 \cdot \bar{v}_{\text{evidence}}$$

### 3.2 Grudge and Fondness Dynamics
$$g_{ij} \leftarrow \mathrm{clip}\left( g_{ij} + 0.60 \cdot \text{harm} - 0.40 \cdot \text{help} \cdot \mathbb{1}[g_{ij} > 0], \, 0.0, \, 1.0 \right)$$
$$f_{ij} \leftarrow \mathrm{clip}\left( f_{ij} + 0.60 \cdot \text{help} - 0.20 \cdot \text{harm}, \, 0.0, \, 1.0 \right)$$

Where:
- $\text{harm} = \sum_{e \in \text{evidence}} e.\text{importance} \cdot \mathbb{1}[e.\text{is\_harm}]$
- $\text{help} = \sum_{e \in \text{evidence}} e.\text{importance} \cdot \mathbb{1}[e.\text{is\_help}]$

### 3.3 Confidence Saturation
$$\text{conf}_{ij} = 1.0 - \exp(-0.40 \cdot n_{\text{total}})$$

### 3.4 Temporal Grudge Decay & Forgiveness
Each simulation tick:
$$g_{ij} \leftarrow g_{ij} \cdot \lambda_{\text{decay}}$$
$$\lambda_{\text{decay}} = \min(0.999, \, \max(0.950, \, 0.995 + 0.004 N - 0.003 A))$$

## 4. Derivation Where Appropriate
The saturating confidence function $1 - e^{-0.4 n}$ approaches $0.90$ certainty when $n \ge 6$ interactions have occurred. The asymmetric coefficients ($0.60$ harm vs. $0.40$ help) reflect the negativity bias in interpersonal moral attribution.

## 5. Pseudocode
```python
def reflect_episodes(unreflected, current_dispositions, traits):
    grouped = group_by_target(unreflected)
    for other_id, episodes in grouped.items():
        evidence = top_k(episodes, k=8, key=importance)
        harm = sum(e.importance for e in evidence if e.is_harm)
        help_ = sum(e.importance for e in evidence if e.is_help)
        
        d = current_dispositions.get(other_id, Disposition(other_id))
        d.valence = 0.85 * d.valence + 0.15 * weighted_mean(evidence)
        d.grudge = clip(d.grudge + 0.6*harm - 0.4*help_*(d.grudge > 0), 0, 1)
        d.fondness = clip(d.fondness + 0.6*help_ - 0.2*harm, 0, 1)
        d.confidence = 1.0 - exp(-0.40 * (d.evidence_count + len(evidence)))
        d.evidence_ids = (d.evidence_ids + tuple(e.id for e in evidence))[-16:]
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(M \log M)$ per agent where $M$ is the number of unreflected memory episodes in the day.
- **Space Complexity:** $\mathcal{O}(D)$ where $D \le 16$ is the number of distinct peers known to the agent.

## 7. Implementation Notes
Implemented in `autogen.intelligence.village.reflection`. Telemetry uses `ReflectionEngine.render_thought` to generate natural-language explanations citing specific evidence IDs for inspection panels.

## 8. Verification Strategy & Test Oracles
1. **Theft Grudge Oracle:** A villager robbed 3 consecutive times by the same culprit ends with $g > 0.60$ and exactly 3 citing evidence IDs.
2. **Reconciliation Invariant:** A culprit who subsequently provides repeated help reduces the victim's grudge strictly monotonically.
3. **Memory Boundedness:** An agent experiencing 1,000 interactions retains at most 16 evidence IDs per disposition.

## 9. Known Limitations
- Third-party rumor evidence is integrated directly into firsthand dispositions in Increment 1; provenance discount chains land in Increment 4.

## 10. References / Citations
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23), pp. 1-22.
- Baumeister, R. F., Bratslavsky, E., Finkenauer, C., & Vohs, K. D. (2001). *Bad is Stronger than Good*. Review of General Psychology, 5(4), pp. 323-370.

