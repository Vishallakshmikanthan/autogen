# Utility-Guided Hierarchical Intention Behavior Tree with Hysteresis

## 1. Theory & Mathematical Definition
Pure per-tick utility evaluation (argmax over all available actions) lacks temporal commitment and task continuity: agents can easily thrash between distant objectives across consecutive simulation ticks. Conversely, standard rigid Behavior Trees (BTs) or finite-state machines can produce robotic, brittle patterns that fail to adapt when acute emergencies arise.

The **Utility-Guided Hierarchical Intention Behavior Tree** bridges these approaches:
1. High-level goals are structured as multi-step sequential intention pipelines (`SURVIVE_FOOD`, `SURVIVE_WATER`, `WORK`, `MARKET`, `SOCIALIZE`, `RESPOND`, `REST`).
2. Candidate intentions are dynamically evaluated and scored using current need pressures, expected action utilities, circadian time-of-day priors, and switching penalties.
3. Once an intention is adopted, the agent maintains it with temporal commitment until it completes, fails, or an alternative intention surpasses it by a strict hysteresis barrier $H$.

---

## 2. Assumptions
1. High-level human motivations decompose into identifiable sequential phases (e.g. traveling to a resource, executing work/trade, returning to shelter).
2. Switching intentions incurs psychological and physical switching costs $C_{switch}$.
3. Desperation (extreme hunger) and emergency safety hazards (`SHELTER`) must always remain available to preempt non-urgent routines.

---

## 3. Equations with Variable Definitions

### 3.1 Intention Scoring Equation
$$\text{Score}(I) = \sum_{k=1}^5 w_k(t) \cdot \Delta d_k(I) + U_{\text{first}}(I) + \pi_I(\tau) - C_{\text{switch}} \cdot \mathbb{1}[I \neq I_{\text{cur}}]$$
- $w_k(t)$: Dynamic need weights from the prepotency engine
- $\Delta d_k(I)$: Expected reduction of deficit $k$ achieved by completing intention $I$
- $U_{\text{first}}(I)$: Expected utility of the best candidate action advancing $I$
- $\pi_I(\tau)$: Circadian prior boost at clock time $\tau$
- $C_{\text{switch}}$: Action switching penalty $C_{\text{switch}} = 1.50 \cdot \text{switching\_cost\_scale} \cdot \mathbb{1}[I \neq I_{\text{cur}}]$

### 3.2 Hysteresis Switching Condition
A rival intention candidate $I^*$ can interrupt and replace active intention $I_{\text{cur}}$ if and only if:
$$\text{Score}(I^*) > \text{Score}(I_{\text{cur}}) + H$$
where the hysteresis barrier $H$ is defined as:
$$H = 0.10 \cdot |\text{Score}(I_{\text{cur}})| + 0.05$$

### 3.3 Intention Lifecycle Management
An intention transitions to terminal state (`DONE` or `FAILED`):
- When progress $p \ge 1.0 \implies \text{status} = \text{DONE}$
- When simulation steps elapsed $(t - t_{\text{start}}) \ge \text{max\_duration} \implies \text{status} = \text{DONE}$ (timeout expiration)
- When environmental condition clears (e.g. `SHELTER` completes when storm subsides)

---

## 4. Derivation
Without hysteresis, when two intentions have nearly identical utilities (e.g. hunger drive $\approx 5.01$ and work drive $\approx 5.02$), micro-fluctuations in distance or price cause high-frequency oscillations every tick (the "ping-pong" effect). Setting the switching threshold to $H = 0.10 \cdot |\text{Score}(I_{\text{cur}})| + 0.05$ creates an energy barrier requiring a significant priority advantage before an agent aborts an in-progress journey.

---

## 5. Pseudocode
```python
def arbitrate_intentions(current_intention, step, need_state, weights, role, time_of_day, candidates):
    if active_emergency:
        return Intention(kind="RESPOND", commitment=0.95), {"RESPOND": 99.0}
        
    scores = {}
    for kind, spec in INTENTION_SPECS.items():
        need_gain = sum(w * delta for w, delta in zip(weights, spec.deficit_deltas))
        u_first = max((c.expected_utility for c in candidates if c.action in spec.allowed), default=0.0)
        c_prior = circadian_priors[kind]
        c_switch = 0.0 if kind == current_intention.kind else 1.5 * scale
        scores[kind] = spec.base_priority + need_gain + 0.35 * u_first + c_prior - c_switch

    best_kind = argmax(scores)
    best_score = scores[best_kind]

    if current_intention is active and not current_intention.is_expired(step):
        h = 0.10 * abs(scores[current_intention.kind]) + 0.05
        if best_score <= scores[current_intention.kind] + h:
            return current_intention.with_decayed_commitment(0.99), scores

    return Intention(kind=best_kind, started_step=step, max_duration=spec.max_duration), scores
```

---

## 6. Computational Complexity
- **Time Complexity:** $O(I \cdot A)$ where $I \le 7$ intentions and $A \le 29$ candidate actions $\implies O(1)$ in practice.
- **Space Complexity:** $O(I)$ storing intention specifications and score dictionaries.

---

## 7. Implementation Notes
- Restricts action evaluation in `decide()` to intention-serving actions $\cup$ safety actions (`CONSUME_RATION`, `SHELTER`).
- Unethical desperation pathways (`STEAL_GRANARY`) can bypass filtering when physiological deficit $d_1 > 0.70$.
- Intention state is fully serialized in `DecisionTrace.rationale` and API state payloads.

---

## 8. Verification Strategy & Test Oracles
- **Hysteresis Non-Oscillation Oracle:** In an ambiguous environment over 200 ticks, intention switches are strictly bounded without 2-cycle flip-flops.
- **Intention Expiration Oracle:** An intention cannot exceed its `max_duration` without completing or resetting.
- **Crisis Preemption Oracle:** A sudden violent thunderstorm immediately interrupts background work into `RESPOND` / `SHELTER`.

---

## 9. Known Limitations
- Behavior tree sub-steps are sequential linear pipelines rather than arbitrary recursive parallel trees.
- Does not yet support multi-agent shared cooperative plans (planned for Increment 4).

---

## 10. References / Citations
- Bratman, M. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.
- Colledanchise, M., & Ögren, P. (2018). *Behavior Trees in Robotics and AI: An Introduction*. CRC Press.
