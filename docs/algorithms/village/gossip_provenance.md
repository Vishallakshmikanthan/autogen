# Epidemic Lossy Gossip Transmission with Provenance & Noisy-OR Corroboration

## 1. Theory & Mathematical Definition
In decentralized agrarian societies without print journalism or telecommunications, social information propagates through informal peer-to-peer transmission: **gossip** (Dunbar, 1996; Foster, 2004). Gossip serves two vital social functions:
1. **Reputation Monitoring:** Informing community members of unobserved moral infractions, generosity, or economic shortages.
2. **Social Bonding:** Creating dyadic alignment against common threats or rivals.

However, gossip is fundamentally **lossy and subjective**:
- **Speaker Distortion:** Speakers exaggerate claims proportionally to their emotional arousal and personal hostility toward the subject (Allport & Postman, 1947).
- **Hop Decay:** Indirect hearsay loses epistemic certainty across social hops ($A \to B \to C$).
- **Listener Skepticism:** Listeners evaluate incoming claims through source credibility and prior consistency.
- **Corroboration vs. Rumor Echoes:** Multiple independent witnesses corroborate a claim via Noisy-OR probabilistic fusion (Pearl, 1988), while circular hearsay (rumor loops) must be suppressed to prevent confirmation cascades.

---

## 2. Assumptions
1. Beliefs concern an identifiable subject and a categorical claim (e.g. `("bheem", "stole_grain")`).
2. Transmissions track an explicit chronological provenance chain `source_chain = (origin, hop_1, ..., immediate_source)`.
3. If the listener's ID is present in the `source_chain`, the transmission is recognized as a rumor echo and rejected ($P_{\text{accept}} = 0$).
4. Acceptance of incoming beliefs is probabilistic, mediated by listener trust, speaker social status, and listener Openness.

---

## 3. Equations with Variable Definitions

### 3.1 Speaker Spin & Valence Distortion
For speaker $s$ conveying belief $B$ about subject $j$:
$$\text{spin} = 1.0 + 0.25 \cdot \max(0, a_s) + 0.20 \cdot \text{dislike}_{s \to j}$$
$$\text{dislike}_{s \to j} = \max\left(0.0,\, (0.50 - \text{affinity}_{s \to j}) \times 2.0\right) \in [0, 1]$$
$$\text{sent\_valence} = \mathrm{clip}(B.\text{valence} \cdot \text{spin},\, -1.0,\, 1.0)$$

### 3.2 Epistemic Hop Decay
First-hand observations retain full confidence ($1.0$), while indirect hearsay decays by 10% per hop:
$$\text{sent\_confidence} = \mathrm{clip}(B.\text{confidence} \cdot (1.0 \text{ if } B.\text{first\_hand} \text{ else } 0.90),\, 0.0,\, 1.0)$$

### 3.3 Listener Source Credibility ($\text{cred}$)
For listener $l$ evaluating speaker $s$:
$$\text{cred} = 0.50 \cdot \text{trust}_{l \to s} + 0.30 \cdot \text{status\_norm}(s) + 0.20 \cdot \text{affinity}_{l \to s}$$

### 3.4 Prior Consistency ($\text{consistency}$)
If listener $l$ already holds prior belief $B_{\text{prior}}$ regarding the same subject and claim:
$$\text{consistency} = 1.0 - \frac{|B_{\text{prior}}.\text{valence} - \text{sent\_valence}|}{2.0}$$
(or $1.0$ if no prior belief exists).

### 3.5 Sigmoidal Acceptance Probability ($P_{\text{accept}}$)
With listener Openness trait $O_l \in [0, 1]$:
$$\text{skepticism} = 0.50 + 0.50 \cdot (1.0 - O_l)$$
$$x = 4.0 \cdot \left(\frac{\text{cred} \cdot \text{consistency}}{\text{skepticism}} - 0.50\right)$$
$$P_{\text{accept}} = \frac{1}{1 + e^{-x}}$$

### 3.6 Noisy-OR Independent Corroboration
If listener $l$ accepts an incoming belief where $B_{\text{prior}}.\text{source\_chain} \cap B.\text{source\_chain} = \emptyset$:
$$c_{\text{merged}} = 1.0 - (1.0 - c_{\text{prior}})\left(1.0 - \text{sent\_confidence} \cdot \text{cred}\right)$$
$$\text{times\_heard} \leftarrow B_{\text{prior}}.\text{times\_heard} + 1$$

---

## 4. Derivation & Properties
- **Echo Loop Invariant:** If $l \in B.\text{source\_chain}$, $P_{\text{accept}} = 0.0$. This formally guarantees that an agent cannot corroborate their own rumor after it circulates through intermediaries.
- **Corroboration Monotonicity:** Let $c_1, c_2 \in (0, 1)$. By Noisy-OR:
  $$c_{\text{merged}} = c_1 + c_2 - c_1 c_2 > \max(c_1, c_2)$$
  Two independent hearsay reports strictly yield higher epistemic confidence than either report in isolation.

---

## 5. Pseudocode

```text
function TransmitGossip(speaker, listener, belief, rng, step) -> (Belief, bool):
    // 1. Loop suppression
    if listener.id in belief.source_chain:
        return None, False

    // 2. Speaker spin & confidence decay
    spin = 1.0 + 0.25 * max(0, speaker.pad.arousal) + 0.20 * max(0, (0.50 - speaker.affinity[belief.subject]) * 2)
    sent_val = clip(belief.valence * spin, -1.0, 1.0)
    decay = 1.0 if belief.first_hand else 0.90
    sent_conf = clip(belief.confidence * decay, 0.0, 1.0)

    // 3. Listener credibility and skepticism
    cred = 0.50 * listener.trust[speaker.id] + 0.30 * speaker.status + 0.20 * listener.affinity[speaker.id]
    prior = listener.beliefs.get(belief.subject, belief.claim)
    consistency = 1.0 if prior is None else (1.0 - abs(prior.valence - sent_val) / 2.0)
    skepticism = 0.50 + 0.50 * (1.0 - listener.traits.openness)

    x = 4.0 * (cred * consistency / skepticism - 0.50)
    p_accept = 1.0 / (1.0 + exp(-x))

    if rng.random() >= p_accept:
        return None, False

    // 4. Corroboration vs first reception
    new_chain = belief.source_chain + (speaker.id,)
    if prior and Disjoint(prior.source_chain, new_chain):
        merged_conf = 1.0 - (1.0 - prior.confidence) * (1.0 - sent_conf * cred)
        times = prior.times_heard + 1
    else:
        merged_conf = sent_conf * cred
        times = 1

    return Belief(subject=belief.subject, claim=belief.claim, valence=sent_val, confidence=merged_conf, source_chain=new_chain, times_heard=times), True
```

---

## 6. Computational Complexity
- **Time Complexity:** $O(H)$ ancestry intersection check where $H \le 10$ hops; executes in $< 10 \ \mu\text{s}$.
- **Space Complexity:** $O(B)$ bounded local beliefs per villager ($B \le 50$).

---

## 7. Implementation Notes
- Implemented in `packages/autogen/intelligence/village/gossip.py`.
- `BeliefStore` provides state hashing (`digest()`) for simulation reproducibility assertions.

---

## 8. Verification Strategy & Test Oracles
1. **Speaker Spin Oracle:** Test hand-computed spin with $a=0.8, \text{affinity}=0.1 \implies \text{dislike}=0.8 \implies \text{spin} = 1 + 0.2 + 0.16 = 1.36$.
2. **Loop Suppression Invariant:** Assert transmission is rejected with 100% certainty if listener ID is present in `source_chain`.
3. **Noisy-OR Oracle:** Verify two independent reports with $c_1=0.60, c_2=0.50, \text{cred}=1.0$ yield $c_{\text{merged}} = 1 - 0.40 \times 0.50 = 0.80$.

---

## 9. Known Limitations
- The claim taxonomy is discrete/symbolic; semantic nuance is preserved in categorical keys rather than continuous embeddings.

---

## 10. References / Citations
- Dunbar, R. I. (1996). *Grooming, Gossip, and the Evolution of Language*. Harvard University Press, Chapter 4.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann, Chapter 4 (Noisy-OR Models).
- Allport, G. W., & Postman, L. (1947). *The Psychology of Rumor*. Henry Holt and Company.
