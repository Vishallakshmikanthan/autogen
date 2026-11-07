# Utility-Grounded Dialogue Act Selection with Eavesdropping Risk

## 1. Theory & Mathematical Definition
In human communication, speakers do not produce random utterances or uniform responses. Utterances are **illocutionary speech acts** chosen to achieve specific communicative goals (Austin, 1962; Searle, 1969): reinforcing social bonds, acquiring vital information, venting emotional distress, or influencing group reputation.

However, communicative choices carry social risks. Disseminating defamatory gossip or lodging public criminal accusations in the presence of guards, rivals, or magistrates risks severe social retribution.

The **Dialogue Act Utility Engine** evaluates candidate conversational acts (`GREET`, `SMALL_TALK`, `SHARE_NEWS`, `GOSSIP`, `ASK_FAVOR`, `COMFORT`, `COMPLAIN`, `ACCUSE`, `APOLOGIZE`, `FAREWELL`) via multi-attribute utility theory (Russell & Norvig, 2020), balancing social status gain, interpersonal warmth, information transfer, emotional mood congruence, and **eavesdropping risk** derived from spatial sound propagation.

---

## 2. Assumptions
1. Candidate dialogue acts are generated according to active conversational contexts and intention states.
2. Speakers evaluate the risk of sensitive acts based on the likelihood of third-party eavesdropping ($P_{\text{overheard}}$).
3. Agents possess personality-specific risk aversion parameters ($\kappa_i$) that modulate sensitivity to being overheard.
4. Calculations are entirely deterministic, reproducible, and executed without external language models.

---

## 3. Equations with Variable Definitions

### 3.1 Multi-Attribute Act Utility ($U_{\text{act}}(d)$)
For candidate speech act $d$ evaluated by speaker $i$ communicating with listener $j$:
$$U_{\text{act}}(d) = w_{\text{soc}} \cdot \Delta \text{status}(d) + w_{\text{rel}} \cdot \Delta \text{affinity}(d) + w_{\text{info}} \cdot \text{info\_value}(d) + w_{\text{mood}} \cdot \text{mood\_congruence}(d) - \text{cost}(d) - R_{\text{overhear}}(d)$$

### 3.2 Weighting Coefficients & Personality Modulation
- $w_{\text{soc}}$: Base societal esteem weight
- $w_{\text{rel}}$: Interpersonal relationship weight
- $w_{\text{info}} = 1.0 + 0.50 \cdot O_i$ (elevated by Openness)
- $w_{\text{mood}} = 0.80 + 0.50 \cdot N_i$ (elevated by Neuroticism)

### 3.3 Mood Congruence Component
Let $v_{\text{act}} \in [-1, 1]$ be the emotional valence of act $d$ and $v_{\text{speaker}}$ be speaker $i$'s current PAD pleasure coordinate:
$$\text{mood\_congruence}(d) = 1.0 - \frac{|v_{\text{speaker}} - v_{\text{act}}|}{2.0}$$

### 3.4 Eavesdropping Risk Penalty ($R_{\text{overhear}}$)
For socially sensitive or defamatory acts (`GOSSIP`, `ACCUSE`, `COMPLAIN`):
$$R_{\text{overhear}}(d) = \mathbb{1}_{\text{risky}}(d) \cdot \theta_{\text{risk}} \cdot P_{\text{overheard}} \cdot (\kappa_i \cdot 0.05)$$
where:
- $\theta_{\text{risk}} = 0.35$ is the canonical risk penalty coefficient
- $P_{\text{overheard}} = \max_{k \notin \{i, j\}} \text{audibility}(i, k)$ using spatial sound attenuation
- $\kappa_i = 20.0(0.50 + 0.50 C_i)(1.0 - 0.50 O_i)$ is speaker $i$'s trait-derived risk aversion

---

## 4. Derivation & Properties
- **Risk Suppression under Surveillance:**
  When a high-ranking guard is within earshot ($P_{\text{overheard}} \approx 1.0$) and speaker Conscientiousness is high ($C = 0.9, O = 0.2 \implies \kappa \approx 17.1$), the overhear penalty is:
  $$R_{\text{overhear}} = 0.35 \cdot 1.0 \cdot (17.1 \cdot 0.05) \approx 0.30$$
  This penalty outweighs the modest social benefit of casual gossip ($\Delta U_{\text{gossip}} \approx 0.15$), compelling prudent villagers to avoid illicit gossip in public squares.

---

## 5. Pseudocode

```text
function EvaluateDialogueActUtility(speaker, listener, act, p_overheard) -> float:
    w_info = 1.0 + 0.50 * speaker.traits.openness
    w_mood = 0.80 + 0.50 * speaker.traits.neuroticism
    kappa = 20.0 * (0.50 + 0.50 * speaker.traits.conscientiousness) * (1.0 - 0.50 * speaker.traits.openness)

    d_status, d_affinity, info_val, cost, act_val = GetActAttributes(act.type)
    mood_cong = 1.0 - abs(speaker.pad.pleasure - act_val) / 2.0

    u_base = (speaker.weights.w_soc * 0.10 * d_status
              + speaker.weights.w_rel * 1.50 * d_affinity
              + w_info * info_val
              + w_mood * mood_cong
              - cost)

    if act.is_risky or act.type in [GOSSIP, ACCUSE]:
        risk_penalty = 0.35 * p_overheard * (kappa * 0.05)
        u_base -= risk_penalty

    return u_base
```

---

## 6. Computational Complexity
- **Time Complexity:** $O(A)$ evaluations where $A \le 10$ candidate acts; execution time $< 15 \ \mu\text{s}$.
- **Space Complexity:** $O(1)$ stack allocation.

---

## 7. Implementation Notes
- Implemented in `packages/autogen/intelligence/village/conversation.py`.
- Integrates with `VillageDialogueBus.compute_audibility` to query empirical eavesdropping risk from surrounding coordinates.

---

## 8. Verification Strategy & Test Oracles
1. **Hand-Computed Utility Oracle:** Test known parameter set for `GREET` ($w_{\text{soc}}=10, w_{\text{rel}}=12, \text{affinity}=0.5, v=0.5$). Verify calculated utility matches hand calculation to within $10^{-4}$.
2. **Eavesdropping Risk Invariant:** For `GOSSIP` with $P_{\text{overheard}} = 0.0$ vs $P_{\text{overheard}} = 1.0$, verify utility drops strictly by $\theta_{\text{risk}} \cdot \kappa \cdot 0.05$.
3. **Conscientiousness Moderation:** Verify high-$C$ agents penalize risky gossip more severely than low-$C$ impulsive agents.

---

## 9. Known Limitations
- Candidate speech acts are generated categorically rather than generated through free-form grammatical search.

---

## 10. References / Citations
- Austin, J. L. (1962). *How to Do Things with Words*. Oxford University Press, Lectures 1-4.
- Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge University Press.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson, Chapter 16 (Making Complex Decisions).
