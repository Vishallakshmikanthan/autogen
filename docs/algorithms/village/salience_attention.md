# Algorithm: Multi-Modal Salience Attention Filtering & Habituation Budget

> **Family:** Decision  
> **Source Module:** Module M22-H (`autogen.intelligence.village.attention`)  
> **Standard:** AGENTS.md 10-Section Mathematical Specification

---

## 1. Theory & Mathematical Definition
Biological agents operate under strictly bounded cognitive and sensory attention. Rather than processing all ambient environmental events equally, sensory percepts are filtered through a multi-modal bottom-up salience map attenuated by physical distance and repeated-stimulus habituation, amplified by subjective relevance and internal arousal, and capped by an attention budget.

## 2. Assumptions
1. Acoustic signals attenuate exponentially over Manhattan/Euclidean spatial distance with characteristic length $L = 6.0$ tiles.
2. Global environmental phenomena (thunderstorms) propagate with zero attenuation ($\mathrm{prox} = 1.0$).
3. Habituation rises upon repeated stimulus occurrences and decays geometrically toward zero in their absence.
4. An agent processes at most $\text{ATTN\_BUDGET} = 2$ high-priority alerts per tick.

## 3. Equations with Variable Definitions
For a perceived event $p$ sensed by agent $i$:

$$S_i(p) = g_{\text{mod}}(p) \cdot I(p) \cdot \text{prox}(p) \cdot (1 - h_{\text{kind}}) \cdot (1 + R_i(p)) \cdot (1 + A_i^+) \cdot G_{\text{trait}}$$

Where:
- $g_{\text{mod}}$: Modality gain (Auditory: 1.0, Visual: 0.8, Environmental: 1.2, Social: 1.0).
- $I(p) \in [0, 1]$: Raw event physical intensity.
- $\text{prox}(p)$: Spatial proximity factor, $1.0$ for global events, $\exp(-d / 6.0)$ otherwise.
- $h_{\text{kind}} \in [0, 1]$: Current habituation level for event kind.
- $R_i(p) \ge 0$: Relational and goal relevance (+1.0 for kin/trusted/enemy source, +0.5 for goal zone).
- $A_i^+ = \max(0, \min(1, a_i))$: Current positive PAD arousal.
- $G_{\text{trait}}$: Personality salience gain ($1 + 0.5 N$ for threats, $1 + 0.5 A$ for cries, $1 + 0.5 O$ for novel non-threats).

### Habituation Dynamics
On tick $t$:
$$h(t+1) = \begin{cases} h(t) + 0.30 \cdot (1 - h(t)) & \text{if kind observed at step } t \\ 0.90 \cdot h(t) & \text{otherwise} \end{cases}$$

### Attention Thresholds & Latency
- Notice Threshold: $\theta_{\text{notice}} = 0.25$ (encoded into passive peripheral memory).
- Alert Threshold: $\theta_{\text{alert}} = 0.55$ (eligible to interrupt, latency 1 tick).
- Startle Threshold: $\theta_{\text{startle}} = 1.10$ (immediate preemption reflex, latency 0 ticks).

## 4. Derivation Where Appropriate
The exponential decay $\exp(-d/L)$ models atmospheric and architectural acoustic attenuation in a 2D grid settlement. The multiplicative formulation ensures that completely habituated ($h=1$) or zero-intensity stimuli yield identically zero salience.

## 5. Pseudocode
```python
def process_percepts(percepts, traits, arousal, step):
    scored = []
    for p in percepts:
        s = compute_salience(p, traits, arousal, habituation[p.kind])
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    
    alerts, notices = [], []
    for s, p in scored[:ATTN_BUDGET]:
        if s >= THETA_ALERT:
            lat = 0 if s >= THETA_STARTLE else 1
            alerts.append(Alert(p, salience=s, eligible_step=step+lat))
        elif s >= THETA_NOTICE:
            notices.append(p)
    update_habituation(percepts)
    return alerts, notices
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(P \log P)$ where $P$ is the number of raw percepts sensed in the tick, dominated by sorting for the top-$K$ budget.
- **Space Complexity:** $\mathcal{O}(K)$ where $K = \text{ATTN\_BUDGET} = 2$.

## 7. Implementation Notes
Implemented in `autogen.intelligence.village.attention`. Gaze tracking is immediately oriented toward `alerts[0]` or the top perceived event.

## 8. Verification Strategy & Test Oracles
1. **Zero Novelty Invariant:** If $h_{\text{kind}} = 1.0$, $S_i(p) \equiv 0.0$.
2. **Budget Guarantee:** $\mathrm{len}(\text{alerts}) \le 2$ regardless of input size.
3. **Thunder Walkthrough Oracle:** For global thunder ($I=0.9, g_{\text{mod}}=1.2, d=0, h=0, R=0, A=0.1, N=0.7$), hand-calculated $S \approx 1.2 \cdot 0.9 \cdot 1.0 \cdot 1.0 \cdot 1.0 \cdot 1.1 \cdot 1.35 = 1.6038 > \theta_{\text{startle}}$.

## 9. Known Limitations
- Ray-traced acoustic occlusion through thick stone walls is approximated by indoor/outdoor attenuation rather than full wavefront simulation.

## 10. References / Citations
- Broadbent, D. E. (1958). *Perception and Communication*. Pergamon Press.
- Itti, L., Koch, C., & Niebur, E. (1998). *A Model of Saliency-Based Visual Attention for Rapid Scene Analysis*. IEEE TPAMI, 20(11), pp. 1254-1259.

