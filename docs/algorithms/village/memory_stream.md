# Episodic Memory Stream, Importance Scoring, and Bounded Forgetting

## 1. Theory & Mathematical Definition
Human episodic memory is neither an unbounded log of all past events nor a lossy single-state summary. Instead, human cognition maintains a structured, time-ordered memory stream of experiences, observations, internal thoughts, and reflections (Park et al., 2023; Tulving, 1983).

Because perceptual and cognitive capacity is fundamentally resource-constrained, two essential biological mechanisms govern the memory stream:
1. **Salience / Importance Encoding:** Experiences are encoded with an intrinsic importance score $I_m \in [0, 1]$ proportional to emotional valence extremity and evolutionary/survival significance (Kahneman, 2011; Easterbrook, 1959).
2. **Ebbinghaus Retention & Forgetting:** Memories decay in accessibility over time following an exponential decay curve moderated logarithmically by rehearsal and retrieval access frequency (Ebbinghaus, 1885; Wixted & Ebbesen, 1991).

To maintain bounded computational complexity in open-ended village simulations, the agent's memory stream operates under an invariant capacity cap ($K_{\max} = 300$). When memory capacity is exceeded, retention-strength-based pruning evicts the weakest items while guaranteeing **unreflected memory protection**, ensuring raw experiences are consolidated into higher-order reflections before discard.

---

## 2. Assumptions
1. Discrete simulation time steps $t \in \mathbb{N}$ serve as the chronological axis.
2. Memory items carry structured categorical and numeric metadata: timestamp of creation $t^{crt}$, last access step $t^{acc}$, access count $n^{acc}$, importance $I_m \in [0, 1]$, affective coordinates $(P, A, D)$, and reflection flag $\text{reflected} \in \{\text{True}, \text{False}\}$.
3. Emotional valence extremity $|v|$ and event type saliency $s$ reliably correlate with cognitive memorability.
4. Access events reinforce memory strength nonlinearly, reflecting cognitive consolidation via rehearsal.

---

## 3. Equations with Variable Definitions

### 3.1 Intrinsic Memory Importance ($I_m \in [0, 1]$)
For an experience with emotional valence $v \in [-1, 1]$, arousal $a \in [-1, 1]$, self-involvement $s \in [0, 1]$, rarity $r \in [0, 1]$, and consequence $c \in [0, 1]$:
$$I_m = \mathrm{clip}(0.35 |v| + 0.25 a^{+} + 0.20 s + 0.10 r + 0.10 c,\, 0.0,\, 1.0)$$

where:
- $a^{+} = \max(0.0, a)$ captures heightened arousal as a memorability multiplier
- $s \in [0.0, 1.0]$ is self-involvement ($1.0$ for direct self actions/victimhood, $0.5$ for direct dyadic interaction, $0.2$ for third-party observation)
- $r = \frac{1}{1 + n_{48}} \in (0, 1]$ represents novelty/rarity based on identical event tag occurrences in the past 48 ticks
- $c \in [0.0, 1.0]$ scales consequence from self-observed economic (hunger, currency) and social (trust, hostility) shifts

### 3.2 Retention Strength with Access Reinforcement ($S(m, t)$)
Let $t$ be the current simulation step, $t^{acc}$ be the step when memory item $m$ was last accessed (or $t^{crt}$ upon creation), and $n^{acc} \ge 0$ be the cumulative retrieval access count:
$$S(m, t) = I_m \cdot \exp\left(-\frac{t - t^{acc}}{S_0}\right) \cdot \left(1.0 + 0.30 \ln(1 + n^{acc})\right)$$
where $S_0 = 50.0$ represents the characteristic half-life scaling factor in steps.

### 3.3 Bounded Pruning with Unreflected Protection
When stream cardinality $|M| > K_{\max}$ (default $K_{\max} = 300$), items are selected for eviction by sorting ascending by retention strength $S(m, t)$:
1. Eviction candidates are drawn from items with $\text{reflected} = \text{True}$.
2. If and only if the count of reflected items is insufficient to reach capacity $K_{\max}$, the weakest unreflected items are pruned as a hard fallback.

---

## 4. Derivation & Properties
- **Monotonicity in Valence and Saliency:**
  $$\frac{\partial I_m}{\partial |v|} = 0.35 > 0, \quad \frac{\partial I_m}{\partial s} = 0.45 > 0$$
- **Decay and Rehearsal Asymptotics:**
  As $t - t^{acc} \to \infty$, $S(m, t) \to 0$. Conversely, repeated retrievals increase $n^{acc}$, producing a sub-linear logarithmic boost that models cognitive spaced retrieval without causing strength divergence.

---

## 5. Pseudocode

```text
function ComputeImportance(valence: float, saliency: float) -> float:
    raw = 0.20 + 0.35 * abs(valence) + 0.45 * saliency
    return clamp(raw, 0.0, 1.0)

function ComputeRetentionStrength(item: MemoryItem, current_step: int, S_0: float = 50.0) -> float:
    dt = max(0, current_step - item.last_accessed_step)
    decay = exp(-dt / S_0)
    rehearsal = 1.0 + 0.30 * ln(1.0 + item.access_count)
    return item.importance * decay * rehearsal

function PruneMemoryStream(stream: List[MemoryItem], current_step: int, capacity: int = 300):
    if len(stream) <= capacity:
        return
    excess = len(stream) - capacity
    
    // Partition into reflected and unreflected
    reflected = [m for m in stream if m.reflected]
    unreflected = [m for m in stream if not m.reflected]
    
    sort reflected ascending by ComputeRetentionStrength(m, current_step)
    
    if len(reflected) >= excess:
        evicted = set(reflected[:excess])
    else:
        sort unreflected ascending by ComputeRetentionStrength(m, current_step)
        needed = excess - len(reflected)
        evicted = set(reflected + unreflected[:needed])
        
    stream.filter_out(evicted)
```

---

## 6. Computational Complexity
- **Insertion:** $O(1)$ dictionary append.
- **Strength Evaluation:** $O(1)$ elementary math.
- **Pruning:** $O(N \log N)$ sorting over stream size $N \le 300$. Because $N$ is capped at 300, pruning executes in $< 100 \ \mu\text{s}$.
- **Storage Space:** $O(K_{\max})$ bounded memory footprint per agent ($300 \times \approx 300 \text{ bytes} \approx 90\text{ KB}$ per villager).

---

## 7. Implementation Notes
- Implemented in `packages/autogen/intelligence/village/memory_stream.py`.
- Generates a deterministic MD5 hash digest (`digest()`) incorporating item IDs, timestamps, access counts, and importance values for state verification and drift detection.
- `MemoryItem` uses an immutable dataclass or Pydantic model with hashability by `memory_id`.

---

## 8. Verification Strategy & Test Oracles
1. **Importance Bounds & Monotonicity:** Assert $I_m \in [0, 1]$ across all domain extremes. Verify $I(v=1, s=1) = 1.0$ and $I(v=0, s=0) = 0.20$.
2. **Retention Decay Oracle:** For $I=1.0, n^{acc}=0, \Delta t=50$, assert $S = 1.0 \cdot e^{-1.0} \approx 0.367879$.
3. **Rehearsal Boost Oracle:** For $I=1.0, \Delta t=0, n^{acc}=5$, assert $S = 1.0 \cdot (1 + 0.3 \ln(6)) \approx 1.5375$.
4. **Capacity & Protection Invariant:** Insert 350 items (50 reflected, 300 unreflected). Prune to 300. Verify all 50 reflected items are evicted first, leaving all 300 unreflected items intact.

---

## 9. Known Limitations
- Textual summaries are generated by rule-based sentence templates rather than neural encoders.
- Saliency values are assigned categorically; complex cross-modal subtlety (e.g. nuanced nonverbal betrayal) relies on upstream event classifiers.

---

## 10. References / Citations
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)*, pp. 1-22.
- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*. Teachers College, Columbia University.
- Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science*, 2(6), 409-415.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Tulving, E. (1983). *Elements of Episodic Memory*. Oxford University Press.
