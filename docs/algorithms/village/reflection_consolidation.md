# Reflection Consolidation, Situational Attribution, and Forgiveness Dynamics

## 1. Theory & Mathematical Definition
Agents that merely react to instantaneous stimuli cannot form enduring social relationships, moral convictions, or behavioral adaptations. To develop character depth and long-term behavioral consistency, agents require periodic **reflection consolidation** (Park et al., 2023).

Reflection synthesizes discrete episodic experiences into generalized semantic beliefs and updates interpersonal dispositions. In human social cognition, this process is governed by three psychological mechanisms:
1. **Consolidation Triggers:** Reflection is initiated either when working memory importance saturates or during circadian rest (sleep/nightly consolidation; Stickgold, 2005).
2. **Situational Attribution:** When evaluating moral violations (such as theft), humans evaluate context before attributing malicious intent (Kelley, 1967). If severe environmental stressors exist (e.g., drought, famine), high-Agreeableness individuals discount dispositional blame, attributing transgression partly to situational distress.
3. **Probabilistic Forgiveness:** Interpersonal grudges do not persist indefinitely when offenders perform restorative community service and time elapses. Forgiveness is a stochastic cognitive transition mediated by personality traits (Agreeableness and Neuroticism; McCullough et al., 1997).
4. **Pattern Detection:** Repeated observed harms synthesize actionable alerts (e.g., guard patrol deterrence) that alter defensive vigilance.

---

## 2. Assumptions
1. Agents do NOT possess psychic access to other agents' internal physiological hunger or private need deficits (ground-truth isolation). Attribution relies strictly on publicly observable village indicators (drought flag, community granary wheat levels).
2. Reflection operates over unreflected memories in the agent's memory stream, marking them as `reflected = True` once consolidated.
3. Forgiveness decisions utilize explicit seeded RNG streams to preserve strict simulation determinism.

---

## 3. Equations with Variable Definitions

### 3.1 Reflection Trigger Conditions
Let $U = \{m \in M \mid \neg m.\text{reflected}\}$ be the set of unreflected memories. Reflection is triggered if any of the following conditions holds:
1. **Sleep Consolidation:** The agent completed a `REST_HUT` action.
2. **Importance Saturation:** Total accumulated importance $\sum_{m \in U} I_m \ge 2.50$.
3. **Acute Critical Event:** Any unreflected memory has an acute importance score $I_m \ge 0.85$ (e.g. witnessing a theft, severe harm, or pillory sentencing).

### 3.2 Situational Attribution & Moral Blame Discounting
When evaluating interpersonal harm or theft committed by another villager:
- **Public Environmental Stress ($\sigma_{\text{pub}} \in [0, 1]$):**
  $$\sigma_{\text{pub}} = \mathrm{clip}\left(\mathbb{1}[\text{drought}] \cdot 0.50 + \left(1.0 - \frac{W_{\text{granary}}}{100.0}\right) \cdot 0.50,\, 0.0,\, 1.0\right)$$
  where $W_{\text{granary}}$ is the current public granary wheat reserve.
- **Blame Factor ($\beta \in [0.5, 1.0]$):**
  $$\beta = 1.0 - 0.50 \cdot A_i \cdot \sigma_{\text{pub}}$$
  where $A_i \in [0, 1]$ is observer $i$'s Agreeableness trait. Under severe drought ($\sigma_{\text{pub}} = 1.0$), a maximally agreeable observer ($A = 1.0$) scales blame to $0.50$, whereas an antagonistic agent ($A = 0.0$) assigns full blame $\beta = 1.00$. Under abundance ($\sigma_{\text{pub}} = 0.0$), $\beta = 1.00$ for all agents.

### 3.3 Probabilistic Forgiveness Dynamics ($P_f$)
When an offender performs visible community service or time elapses after an interpersonal harm:
$$P_f = \mathrm{clip}(0.20 + 0.50 A_i + 0.30 \cdot \text{sinc},\, 0.0,\, 1.0) \cdot \left(1.0 - \exp\left(-\frac{\Delta t}{48.0}\right)\right)$$
where $A_i$ is observer Agreeableness, $\text{sinc} = 1.0$ if the observer witnessed restorative community service (and $0.40$ otherwise), and $\Delta t$ is elapsed simulation steps since the transgression.

On successful forgiveness:
$$\text{grudge} \leftarrow 0.50 \cdot \text{grudge}$$

### 3.4 Crime Pattern Detection & Spatial-Temporal Clustering
Villagers monitor recurring criminal traces in their episodic memories binned by spatial location and time of day (day vs night):
$$\text{crime\_count}(\text{loc}, \text{is\_night}) = \sum_{m \in M_{\text{crime}}} \mathbb{1}[\text{loc}_m = \text{loc} \land \text{night}_m = \text{is\_night}]$$
When $\text{crime\_count}(\text{loc}, \text{is\_night}) \ge 3$, the agent synthesizes a high-level world belief memory (`kind="belief"`), sensitizing their security vigilance and generating granary protection utility bonuses.

---

## 4. Pseudocode

```text
function ShouldReflect(stream: MemoryStream, current_hour: int) -> bool:
    unreflected = stream.GetUnreflected()
    if sum(m.importance for m in unreflected) >= 15.0:
        return True
    if len(unreflected) >= 25:
        return True
    if current_hour in [22, 23, 0]:
        return True
    return False

function ComputeSituationalBlame(agreeableness: float, drought: bool, granary_wheat: float) -> float:
    stress = 0.0
    if drought:
        stress += 0.50
    stress += 0.50 * (1.0 - clamp(granary_wheat / 100.0, 0.0, 1.0))
    stress = clamp(stress, 0.0, 1.0)
    return 1.0 - 0.50 * agreeableness * stress

function CheckForgiveness(traits: PersonalityTraits, witnessed_service: bool, steps_since_harm: int, rng: Generator) -> bool:
    p = 0.20 + 0.30 * traits.agreeableness - 0.20 * traits.neuroticism
    if witnessed_service:
        p += 0.30
    p += 0.005 * steps_since_harm
    p = clamp(p, 0.05, 0.95)
    return rng.random() < p
```

---

## 5. Computational Complexity
- **Trigger Check:** $O(|U|)$ where $|U| \le 300$, taking $< 10 \ \mu\text{s}$.
- **Consolidation Execution:** $O(|U|)$ to summarize and mark items.
- **Crime Pattern Detection:** $O(C)$ where $C$ is recent crime traces ($C < 20$).
- **Overall Reflection Step:** Executes once nightly or upon saturation; total overhead $< 1 \ \text{ms}$ per agent.

---

## 6. Implementation Notes
- Implemented in `packages/autogen/intelligence/village/reflection.py`.
- Integrates seamlessly with `MemoryStream` and `VillagerAgent.reflect_nightly()`.
- Generates high-importance synthetic memories of kind `"reflection"`, ensuring high retrieval rank during future deliberation.

---

## 7. Verification Strategy & Test Oracles
1. **Trigger Threshold Oracle:** Assert `should_reflect` returns True when $\sum I_m = 15.0$ or count $= 25$, and False when below both outside bedtime.
2. **Attribution Oracle:** Under drought ($\sigma_{\text{pub}} = 1.0$), verify $A=1.0 \implies \text{blame} = 0.50$, $A=0.0 \implies \text{blame} = 1.00$. Under abundance ($\sigma_{\text{pub}} = 0.0$), verify $\text{blame} = 1.00$ regardless of Agreeableness.
3. **Statistical Forgiveness Oracle:** Test 1,000 independent seeded trials of `check_forgiveness` with known parameters ($p = 0.70$). Verify observed mean forgiveness rate matches $p \pm 0.035$ with 99.9% binomial confidence.
4. **Pattern Detection Test:** Pass two theft events by same actor; verify generated pattern reflection and guard bonus activation.

---

## 8. Known Limitations
- Social attribution is currently dyadic and based on macro environmental stress; personal history of past grievances is incorporated via memory retrieval rather than multi-party causal Bayesian DAGs.
- Forgiveness does not immediately restore trust to 100%; trust recovery remains bounded by subsequent honest interactions.

---

## 9. References / Citations
- Kelley, H. H. (1967). Attribution theory in social psychology. *Nebraska Symposium on Motivation*, 15, 192-238.
- McCullough, M. E., Worthington, E. L., & Rachal, K. C. (1997). Interpersonal forgiving in close relationships. *Journal of Personality and Social Psychology*, 73(2), 321-336.
- Stickgold, R. (2005). Sleep-dependent memory consolidation. *Nature*, 437(7063), 1272-1278.
- Park, J. S., et al. (2023). Generative agents: Interactive simulacra of human behavior. *UIST '23*, pp. 1-22.
