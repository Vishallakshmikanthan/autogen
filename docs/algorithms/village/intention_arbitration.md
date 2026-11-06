# Algorithm: Utility-Guided Intention Commitment & Switching Cost Arbitration

> **Family:** Planning  
> **Source Module:** Module M22-H (`autogen.intelligence.village.intention`)  
> **Standard:** AGENTS.md 10-Section Mathematical Specification

---

## 1. Theory & Mathematical Definition
Pure frame-by-frame utility maximization produces erratic jitter and action flip-flopping when ambient opportunities fluctuate. The Intention Layer introduces temporal commitment: an active goal retains behavioral inertia unless an external Alert possesses sufficient urgency to overcome the intention's stickiness plus an explicit hysteresis margin. When deliberating primitive actions, actions serving the active intention are privileged, while off-intention actions incur an explicit switching cost $C_{\text{switch}}$.

## 2. Assumptions
1. An agent holds exactly one primary active Intention at any discrete step.
2. Uncompleted intentions suffer slow commitment decay over time ($c \leftarrow 0.98 c$).
3. Preemption by an Alert requires urgency to exceed stickiness plus a hysteresis margin of $0.15$.

## 3. Equations with Variable Definitions
### 3.1 Interruption Condition
$$\text{urgency} = S \cdot (1 + \text{threat})$$
$$\text{stickiness} = c \cdot (1 + 0.40 C) + 0.30 \cdot \mathbb{1}[\text{progress} > 0.80]$$
$$\text{Interrupt iff: } \text{urgency} > \text{stickiness} + 0.15$$

Where:
- $S$: Alert salience score.
- $\text{threat} \in [0, 1]$: Threat severity of the alert ($\max(0, -\text{valence\_hint})$).
- $c \in [0, 1]$: Current commitment fraction.
- $C$: Agent conscientiousness trait.
- $\text{progress} \in [0, 1]$: Fractional progress toward completing the intention.

### 3.2 Deliberation Utility Modification
For candidate action $a$:
$$U'(a) = U(a) + \lambda_{\text{att}} \cdot u_{\text{react}}(a \mid \text{alerts}) - C_{\text{switch}}(a)$$

Where:
- $U(a)$: Multi-objective utility from the core utility engine.
- $\lambda_{\text{att}} = 4.0$: Reactive attention scaling parameter.
- $u_{\text{react}}(a)$: Reaction utility bonus addressing active alerts (e.g. `SHELTER` during storms).
- $C_{\text{switch}}(a) = \kappa_{\text{sw}} \cdot c \cdot s_{\text{switch}} \cdot \mathbb{1}[a \notin I]$, with $\kappa_{\text{sw}} = 3.0$ and $s_{\text{switch}} = 1 + 0.4 C$.

## 4. Derivation Where Appropriate
The hysteresis margin $+0.15$ guarantees that if an alert urgency marginally matches current commitment, the agent does not oscillate between competing goals on consecutive ticks.

## 5. Pseudocode
```python
def should_interrupt(alert, intention, conscientiousness):
    if not intention.interruptible and alert.salience < THETA_STARTLE:
        return False
    urgency = alert.salience * (1.0 + alert.threat)
    stickiness = intention.commitment * (1.0 + 0.40 * conscientiousness)
    if intention.progress > 0.80:
        stickiness += 0.30
    return urgency > (stickiness + 0.15)
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(A)$ where $A$ is the number of evaluated candidate actions (typically $\le 15$).
- **Space Complexity:** $\mathcal{O}(1)$ for active intention dataclass.

## 7. Implementation Notes
Implemented in `autogen.intelligence.village.intention`. Interrupted intentions record the preempting alert in their metadata for explainable telemetry.

## 8. Verification Strategy & Test Oracles
1. **No-Oscillation Bound:** Under alternating stimuli, intention switches over 20 consecutive ticks $\le 4$.
2. **Personality Latency Oracle:** In identical paired random seeds, a high-Neuroticism agent reacts to a storm alert no later than a high-Conscientiousness agent.
3. **Progress Protection:** An intention at progress $0.90$ is strictly harder to interrupt than the same intention at progress $0.10$.

## 9. Known Limitations
- Hierarchical sub-goal decomposition (full behavior trees / HTNs) is simplified into atomic intention phases in Increment 1.

## 10. References / Citations
- Bratman, M. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.
- Mark, D. (2009). *Behavioral Mathematics for Game AI*. Charles River Media.

