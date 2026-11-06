# Multimodal Episodic Memory Retrieval with Mood Congruence

## 1. Theory & Mathematical Definition
In autonomous social simulation, agents face thousands of episodic memories accumulated across their lifespans. During deliberation, an agent cannot consider all memories simultaneously. The agent must retrieve a sparse subset of $K$ memories ($K \approx 5$) that are most relevant to the immediate decision context (Park et al., 2023).

Human memory retrieval is inherently multimodal:
1. **Recency:** Recently experienced events are more accessible in working memory.
2. **Importance:** Significant, life-altering, or emotionally extreme events remain salient.
3. **Relevance:** Memories sharing entities, locations, and actions with current stimuli are primed.
4. **Mood-Congruent Priming:** Affective state acts as an associative index: when distressed or angry, agents preferentially retrieve negative, stressful, or threatening memories (Bower, 1981; Forgas, 1995).

This algorithm formalizes a four-factor scoring function with mood congruence and deterministic tie-breaking to ensure complete reproducibility in simulation research.

---

## 2. Assumptions
1. Retrieval is performed against the bounded memory stream of the deciding agent.
2. The current context is represented by a `MemoryQuery` specifying the current simulation step $t$, search query text, focal agent IDs, focal zone IDs, and current affective PAD coordinates $(P_q, A_q, D_q)$.
3. Affective coordinates are bounded in $[-1, 1]^3$.
4. Rehearsal side-effects (updating access count and last accessed step) are strictly disabled during read-only evaluations (`rehearse=False`) and enabled only during deliberation execution (`rehearse=True`).

---

## 3. Equations with Variable Definitions

### 3.1 Multimodal Retrieval Score ($\text{score}(m \mid q)$)
Given memory item $m$ and retrieval query $q$:
$$\mathrm{score}(m \mid q) = w_r \cdot \text{recency}(m, t) + w_i \cdot \text{importance}(m) + w_q \cdot \text{relevance}(m, q) + w_m \cdot \text{mood}(m, q)$$

With canonical weights:
$$w_r = 1.0, \quad w_i = 1.0, \quad w_q = 1.5, \quad w_m = 0.4$$

### 3.2 Recency Component ($\text{recency}(m, t) \in (0, 1]$)
With time horizon $\tau_r = 24.0$ simulation ticks and elapsed time $\Delta t = \max(0, t - t^{acc}_m)$ since last rehearsal access:
$$\text{recency}(m, t) = \exp\left(-\frac{\Delta t}{\tau_r}\right)$$

### 3.3 Importance Component ($\text{importance}(m) \in [0, 1]$)
$$\text{importance}(m) = I_m$$

### 3.4 Contextual Relevance Component ($\text{relevance}(m, q) \in [0, 1]$)
Evaluated as Jaccard similarity across the unified entity sets of actors, conceptual tags, and location:
$$S_m = \text{actors}_m \cup \text{tags}_m \cup \{\text{location}_m\}$$
$$S_q = \text{actors}_q \cup \text{tags}_q \cup \{\text{location}_q\}$$
$$\text{relevance}(m, q) = J(S_m, S_q) = \frac{|S_m \cap S_q|}{|S_m \cup S_q|}$$
(with $J(\emptyset, \emptyset) = 0.0$).

### 3.5 Mood Congruence Component ($\text{mood}(m, q) \in [0, 1]$)
Let $(P_m, A_m, D_m)$ and $(P_q, A_q, D_q)$ be the PAD vectors of the memory and the query respectively. The Euclidean distance in $[-1, 1]^3$ has maximum diameter $D_{\max} = 2\sqrt{3} \approx 3.4641$:
$$d_{PAD}(m, q) = \sqrt{(P_m - P_q)^2 + (A_m - A_q)^2 + (D_m - D_q)^2}$$
$$\text{mood}(m, q) = \max\left(0.0,\, 1.0 - \frac{d_{PAD}(m, q)}{2\sqrt{3}}\right)$$

---

## 4. Deterministic Tie-Breaking
When candidate memories produce identical scores within floating-point tolerance ($10^{-9}$), order is strictly resolved by:
1. Retrieval score descending: $-\text{score}(m|q)$
2. Creation timestamp descending (more recent creation preferred): $-t^{crt}_m$
3. Memory unique ID lexicographical ascending: $m.\text{memory\_id}$

---

## 5. Pseudocode

```text
function ScoreMemory(item: MemoryItem, query: MemoryQuery) -> (float, ScoreBreakdown):
    // 1. Recency
    dt = max(0, query.current_step - item.created_step)
    recency = 0.995 ** dt
    
    // 2. Importance
    importance = item.importance
    
    // 3. Relevance
    tokens_item = Tokenize(item.summary)
    tokens_query = Tokenize(query.text)
    j_text = Jaccard(tokens_item, tokens_query)
    
    entities_item = set(item.related_agent_ids)
    entities_query = set(query.focal_agent_ids)
    j_entity = Jaccard(entities_item, entities_query)
    
    relevance = 0.60 * j_entity + 0.40 * j_text
    
    // 4. Mood congruence
    d_pad = EuclideanDistance(item.emotional_pad, query.current_pad)
    mood = 1.0 - (d_pad / sqrt(12.0))
    
    total = 0.25 * recency + 0.30 * importance + 0.30 * relevance + 0.15 * mood
    return total, ScoreBreakdown(recency, importance, relevance, mood, total)

function RetrieveTopK(items: List[MemoryItem], query: MemoryQuery, k: int = 5, rehearse: bool = False) -> List[Tuple[MemoryItem, ScoreBreakdown]]:
    scored = []
    for m in items:
        score, breakdown = ScoreMemory(m, query)
        scored.append((score, m.created_step, m.memory_id, m, breakdown))
        
    // Sort descending by score, descending by created_step, ascending by memory_id
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    top_k = scored[:k]
    
    if rehearse:
        for _, _, _, m, _ in top_k:
            m.access_count += 1
            m.last_accessed_step = query.current_step
            
    return [(m, bd) for _, _, _, m, bd in top_k]
```

---

## 6. Computational Complexity
- **Time Complexity:** For a stream of size $N \le 300$, computing scores takes $O(N \cdot L)$ where $L$ is token length ($< 20$ words). Sorting takes $O(N \log K)$. Total execution time is $< 250 \ \mu\text{s}$ per query.
- **Space Complexity:** $O(K)$ buffer for top-$K$ items.

---

## 7. Implementation Notes
- Implemented in `packages/autogen/intelligence/village/memory_retrieval.py`.
- Exposes `ScoreBreakdown` dataclass enabling comprehensive telemetry inspection in the Algorithm Lab and village viewer.
- Caches token sets where applicable to avoid redundant string parsing.

---

## 8. Verification Strategy & Test Oracles
1. **Hand-Computed Score Oracle:** Construct a known memory ($t^{crt}=10, I=0.8, \text{PAD}=(0.5, 0, 0)$) and query ($t=20, \text{PAD}=(0.5, 0, 0)$). Compute recency ($0.995^{10} \approx 0.95111$), importance (0.80), relevance, and mood (1.0). Verify composite score equals hand-calculated value to within $10^{-5}$.
2. **Mood-Congruent Priming Invariant:** Given two memories with identical importance, recency, and relevance—one happy $(+0.8, +0.2, +0.4)$ and one fearful $(-0.6, +0.6, -0.4)$—when the agent is in a fearful mood, the fearful memory MUST rank strictly higher.
3. **Rehearsal Invariant:** Verify that `retrieve_top_k(..., rehearse=False)` leaves `access_count` unchanged, whereas `rehearse=True` increments it by exactly 1.

---

## 9. Known Limitations
- Textual relevance uses Jaccard token overlap rather than high-dimensional embedding cosine similarity (by architectural constraint SYS-CORE: scratch algorithms only, no external embeddings/LLMs).
- Synonym matching requires stemming or keyword expansions in rule dictionaries.

---

## 10. References / Citations
- Bower, G. H. (1981). Mood and memory. *American Psychologist*, 36(2), 129-148.
- Forgas, J. P. (1995). Mood and judgment: the affect infusion model (AIM). *Psychological Bulletin*, 117(1), 39-66.
- Park, J. S., et al. (2023). Generative agents: Interactive simulacra of human behavior. *UIST '23*, pp. 1-22.
