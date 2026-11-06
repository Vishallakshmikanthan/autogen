# Maslow-Style Soft Prepotency Needs Hierarchy & Dynamic Utility Weights

## 1. Theory & Mathematical Definition
Human decision-making is fundamentally non-stationary: preferences and priorities shift dynamically with physiological deprivation and psychological states. While baseline personality and economic models often assume static linear preference weights $w_{phys}, w_{soc}, w_{econ}$, biological agents experience drives that fluctuate based on homeostatic deficits and satiation.

The **Needs Prepotency Model** introduces a 5-level hierarchical needs structure based on Abraham Maslow's theory of human motivation, formalizing soft prepotency gating across 5 distinct deficiency levels:
1. $L_1$ Physiological: Homeostatic survival requirements (hunger, hydration, energy, health)
2. $L_2$ Safety: Shelter, stored reserves, absence of active threats, legal standing
3. $L_3$ Belonging: Companionship, conversation quality, and social inclusion (loneliness drive)
4. $L_4$ Esteem: Relative societal status ($S_i$), recognition, and reputation
5. $L_5$ Growth / Mastery: Role-specific craft productivity and exploration

Lower-level physiological deficits prioritize survival by softly throttling the activation of higher-level drives via a prepotency product.

---

## 2. Assumptions
1. Vital physiological variables are observable within homeostatic bounds $[0, 100]$ where higher values represent satiety.
2. Deficit levels $d_k \in [0, 1]$ grow monotonically as biological resources deplete.
3. Prepotency is soft: moderate hunger does not completely extinguish social impulses, but extreme starvation drives higher-level weights toward zero.
4. Drive pressure rises quadratically ($P_k = d_k^2$), reflecting steep escalation during acute crises.

---

## 3. Equations with Variable Definitions

### 3.1 Deficit Formulation ($d_k \in [0, 1]$)
- **$L_1$ Physiological Bottleneck:**
  $$d_1 = 1.0 - \frac{\min(h, t, e, \text{health})}{100.0}$$
  where $h$ is hunger, $t$ is hydration, $e$ is energy, and $\text{health}$ is physical vitality.
- **$L_2$ Safety Deficit:**
  $$d_2 = \mathrm{clip}\left(0.50 \cdot \text{exposed\_in\_storm} + 0.30 \cdot \left(1.0 - \min\left(1.0, \frac{\text{food\_stock}}{6.0}\right)\right) + 0.20 \cdot \text{threat}_{\max},\, 0.0,\, 1.0\right)$$
- **$L_3$ Belonging / Loneliness Deficit:**
  With contact score $c \leftarrow 0.90 \cdot c + q_t$ ($q_t$ being contact quality in tick $t$) and target $c^* = 2.0(0.5 + E)$ ($E$ being Extraversion):
  $$d_3 = \mathrm{clip}\left(1.0 - \frac{c}{c^*},\, 0.0,\, 1.0\right)$$
- **$L_4$ Esteem Deficit:**
  $$d_4 = \mathrm{clip}\left(1.0 - \frac{S_i}{\max(0.01, \max_j S_j)},\, 0.0,\, 1.0\right)$$
- **$L_5$ Growth / Mastery Deficit:**
  With mastery score $m \leftarrow 0.98 \cdot m + \mathbb{1}[\text{role-productive action}]$ and target $m^* = 3.0(0.5 + O)$ ($O$ being Openness):
  $$d_5 = \mathrm{clip}\left(1.0 - \frac{m}{m^*},\, 0.0,\, 1.0\right)$$

### 3.2 Drive Pressure ($P_k$)
$$P_k = d_k^p \quad (p = 2.0)$$

### 3.3 Soft Prepotency Gating ($g_k \in [0, 1]$)
With satisfaction level $\text{sat}_j = 1.0 - d_j$:
$$g_1 = 1.0$$
$$g_k = \prod_{j < k} \text{sat}_j^\beta \quad (\beta = 0.70)$$

### 3.4 Dynamic Weight Modulation
$$w_k(t) = \text{base}_k \cdot (1.0 + \lambda P_k) \cdot g_k \quad (\lambda = 2.0)$$
$$w_{phys}(t) = \text{base}_{phys} \cdot (1.0 + \lambda P_1)$$
$$w_{soc}(t) = \text{base}_{soc} \cdot \frac{1}{2}\left[(1.0 + \lambda P_3)g_3 + (1.0 + \lambda P_4)g_4\right]$$

### 3.5 Action Satiation Tracker ($s_a \in [0, 1]$)
On use of satisfier action $a$:
$$s_a \leftarrow s_a + 0.25(1.0 - s_a)$$
On inactive ticks:
$$s_a \leftarrow 0.85 \cdot s_a$$

---

## 4. Derivation
Traditional Maslow hierarchies use strict discrete tiers: $L_{k+1}$ cannot activate until $L_k$ is completely satisfied. In empirical human behavior, this leads to artificial bang-bang thrashing. By raising satisfaction $\text{sat}_j$ to power $\beta = 0.70$ in a continuous product $\prod_{j < k} \text{sat}_j^\beta$, gating is smooth:
- At $d_1 = 0.90$ (acute starvation), $\text{sat}_1 = 0.10 \implies g_3 = 0.10^{0.70} \approx 0.1995$, cutting social drive by $\approx 80\%$.
- At $d_1 = 0.20$ (mild hunger), $\text{sat}_1 = 0.80 \implies g_3 = 0.80^{0.70} \approx 0.855$, allowing social and craft drives to function normally.

---

## 5. Pseudocode
```python
def update_needs(obs, contact_quality, executed_action, traits):
    c = 0.90 * prev_c + contact_quality
    m = 0.98 * prev_m + (1.0 if is_productive(executed_action) else 0.0)
    
    d1 = 1.0 - min(obs.hunger, obs.hydration, obs.energy, obs.health) / 100.0
    d2 = clip(0.5 * exposed + 0.3 * low_reserves + 0.2 * threat, 0.0, 1.0)
    d3 = clip(1.0 - c / (2.0 * (0.5 + traits.extraversion)), 0.0, 1.0)
    d4 = clip(1.0 - obs.status / max_status, 0.0, 1.0)
    d5 = clip(1.0 - m / (3.0 * (0.5 + traits.openness)), 0.0, 1.0)
    
    P = [d**2.0 for d in (d1, d2, d3, d4, d5)]
    
    gates = [1.0]
    prod = 1.0
    for d in (d1, d2, d3, d4):
        sat = max(0.0, min(1.0, 1.0 - d))
        prod *= sat ** 0.70
        gates.append(prod)
        
    w_phys = base_w_phys * (1.0 + 2.0 * P[0])
    w_soc = base_w_soc * 0.5 * ((1.0 + 2.0 * P[2]) * gates[2] + (1.0 + 2.0 * P[3]) * gates[3])
    return NeedState(deficits=(d1..d5), gates=gates, pressures=P), weights(w_phys, w_soc)
```

---

## 6. Computational Complexity
- **Time Complexity:** $O(1)$ constant-time algebraic vector evaluation.
- **Space Complexity:** $O(1)$ memory footprint storing 5 floating-point deficits, gates, and pressures.

---

## 7. Implementation Notes
- All need states are immutable `NeedState` models adhering to typed contracts.
- Satiation tracking prevents repetitive action loops (e.g. non-stop eating or greeting).
- Preserves backwards compatibility: when `enable_needs=False`, agents use static baseline utility weights.

---

## 8. Verification Strategy & Test Oracles
- **Hand-computed Prepotency Oracle:** With $d = (0.9, 0, 0, 0, 0)$, verify $g_3 = 0.10^{0.70} \approx 0.1995$.
- **Monotonicity Oracle:** Verify $\partial w_{phys} / \partial d_1 > 0$ across $d_1 \in [0, 1]$.
- **Loneliness Growth Oracle:** An isolated agent with $q_t = 0$ experiences monotonically increasing $d_3 \to 1.0$.
- **Satiation Decay Oracle:** Consecutive satisfier executions strictly increase $s_a$ while decreasing marginal utility gains.

---

## 9. Known Limitations
- Does not model physiological illness vectors or disease contagion.
- Uses scalar proxy for status deficit rather than pairwise comparative status graph calculations.

---

## 10. References / Citations
- Maslow, A. H. (1943). *A Theory of Human Motivation*. Psychological Review, 50(4), pp. 370–396.
- Hull, C. L. (1943). *Principles of Behavior: An Introduction to Behavior Theory*. Appleton-Century.
