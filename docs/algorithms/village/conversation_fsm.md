# Multi-Tick Conversation State Machine & Engagement Dynamics

## 1. Theory & Mathematical Definition
Human conversation is not an instantaneous single-shot broadcast; it is a collaborative, multi-tick, embodied ritual (Goffman, 1967; Clark, 1996). Conversants approach each other, establish mutual gaze, exchange initial greetings, negotiate common ground over a series of communicative turns, and either close gracefully or terminate abruptly due to external hazards or boredom.

The **Conversation Finite State Machine (FSM)** models multi-tick social encounters across seven distinct operational states:
1. `IDLE`: Not currently engaged in social interaction.
2. `APPROACHING`: Navigating into conversational proximity ($d \le 2$ grid cells).
3. `GREETING`: Exchanging initial communicative contact.
4. `TALKING`: Alternating communicative turns governed by dialogue act utility and active topics.
5. `CLOSING`: Performing mutual farewell rituals.
6. `INTERRUPTED`: Suspended mid-exchange by an urgent environmental or auditory alert.
7. `ABANDONED`: Prematurely terminated by one party walking away due to depleted engagement (social snub).

A continuous **engagement dynamic** governs each conversant's willingness to sustain the exchange, decaying with turn fatigue and boosted by topic relevance and interpersonal affinity.

---

## 2. Assumptions
1. Conversations are dyadic (between two agents $i, j$) within conversational proximity ($d \le 2$ tiles).
2. Turns alternate synchronously between participants.
3. Conversational abandonment mid-turn without a clean `FAREWELL` is perceived as a social slight ("snub"), inflicting affinity and trust penalties unless excused by a salient shared emergency.
4. All simulation state transitions are discrete, deterministic, and verifiable.

---

## 3. Equations with Variable Definitions

### 3.1 Conversational Engagement Dynamics ($E_i(t) \in [0, 1]$)
For conversant $i$ interacting with partner $j$ at turn $k$:
$$E_i(k+1) = \mathrm{clip}\left(E_i(k) + 0.15 \cdot r_{\text{topic}} + 0.10 \cdot \text{affinity}_{i \to j} - 0.08 - 0.10 \cdot b_i,\, 0.0,\, 1.0\right)$$
where:
- $r_{\text{topic}} \in [0, 1]$ represents relevance of the current topic to $i$'s needs and archetype
- $\text{affinity}_{i \to j} \in [0, 1]$ is the directed relationship warmth toward $j$
- $b_i = \min(1.0, k \times 0.15)$ is the boredom/fatigue accumulator proportional to completed turns $k$

### 3.2 Abandonment / Snub Threshold ($\theta_{\text{leave}}$)
If either participant's engagement drops below the critical threshold:
$$E_i < \theta_{\text{leave}} \quad (\theta_{\text{leave}} = 0.20)$$
The state transitions to `ABANDONED`. The slighted partner $j$ updates directed relationships:
$$\Delta \text{affinity}_{j \to i} = -0.10, \quad \Delta \text{trust}_{j \to i} = -0.05$$

### 3.3 Emergency Interruption Bypass
If an active perceived alert satisfies:
$$\text{threat} > 0.40 \quad \lor \quad \text{event.kind} \in \{\text{"THUNDER"}, \text{"RAIN\_ONSET"}\}$$
The state immediately transitions to `INTERRUPTED` with `interrupted_by_alert = True`. The relational snub penalty is completely bypassed ($\Delta \text{affinity} = 0, \Delta \text{trust} = 0$).

### 3.4 Post-Conversation Settlement
Upon clean closure (`CLOSING` / `IDLE`), exchange quality $Q$ is computed from history:
$$Q = 0.60 \cdot \bar{v}_{\text{acts}} + 0.40 \cdot (\bar{E} - 0.50)$$
where $\bar{v}_{\text{acts}}$ is the mean valence of uttered dialogue acts, and $\bar{E}$ is the average terminal engagement.
Both directed edges receive a saturating boost:
$$\text{affinity} \leftarrow \text{affinity} + 0.15 \cdot Q \cdot (1 - \text{affinity})$$
$$\text{familiarity} \leftarrow \min(1.0, \text{familiarity} + 0.05)$$

---

## 4. Derivation & Properties
- **Asymptotic Convergence:** Under neutral stimuli ($r = 0.5, \text{affinity} = 0.5, b = 0$), engagement change per turn is:
  $$\Delta E = 0.15(0.5) + 0.10(0.5) - 0.08 = 0.075 + 0.050 - 0.080 = +0.045$$
  As turns accumulate, boredom rises by $0.10 \times 0.15 = 0.015$ per turn, naturally forcing $\Delta E < 0$ after 4–5 turns. This guarantees conversations naturally terminate within 5–7 turns without infinite loops.

---

## 5. Pseudocode

```text
function StepConversation(conv: Conversation, alerts: List[Alert], relationships: RelationshipStore) -> (ConvState, DialogueAct, str):
    // 1. Check emergency interruption
    for alert in alerts:
        if alert.threat > 0.40 or alert.kind in ["THUNDER", "RAIN_ONSET"]:
            conv.state = ConvState.INTERRUPTED
            conv.interrupted_by_alert = True
            return conv.state, None, "interrupted_by_hazard"

    // 2. FSM Progression
    if conv.state == ConvState.APPROACHING:
        conv.state = ConvState.GREETING
        act = DialogueAct(type=GREET, speaker=conv.turn_owner, target=conv.other(conv.turn_owner))
        conv.turns += 1
        conv.turn_owner = conv.other(conv.turn_owner)
        return conv.state, act, "approached_greeting"

    if conv.state == ConvState.GREETING:
        conv.state = ConvState.TALKING
        act = DialogueAct(type=SMALL_TALK, speaker=conv.turn_owner, target=conv.other(conv.turn_owner))
        conv.turns += 1
        conv.turn_owner = conv.other(conv.turn_owner)
        return conv.state, act, "greeting_acknowledged"

    if conv.state == ConvState.TALKING:
        spk = conv.turn_owner
        lis = conv.other(spk)
        UpdateEngagement(conv, spk, relationships)
        UpdateEngagement(conv, lis, relationships)

        if conv.engagement[spk] < 0.20 or conv.engagement[lis] < 0.20:
            conv.state = ConvState.ABANDONED
            conv.abandoned_by = spk if conv.engagement[spk] < 0.20 else lis
            return conv.state, None, "abandoned_snub"

        act = SelectDialogueAct(spk, lis)
        conv.turns += 1
        if conv.turns >= max_turns or act.type == FAREWELL:
            conv.state = ConvState.CLOSING
            return conv.state, act, "closing"
        conv.turn_owner = lis
        return conv.state, act, "turn_stepped"
```

---

## 6. Computational Complexity
- **Time Complexity:** $O(1)$ constant time FSM transitions and relationship lookups per conversation per tick.
- **Space Complexity:** $O(C)$ bounded memory for active conversations ($C \le \frac{N}{2}$ where $N$ is population).

---

## 7. Implementation Notes
- Implemented in `packages/autogen/intelligence/village/conversation.py`.
- Emits structured `DialogueAct` records consumed downstream by `VillageDialogueBus` and `SymbolicDialogueEngine`.
- Synthesizes an episodic memory item (`kind="conversation"`) upon conversation closure.

---

## 8. Verification Strategy & Test Oracles
1. **Transition Progression Oracle:** Verify sequential progression `APPROACHING` $\to$ `GREETING` $\to$ `TALKING` $\to$ `CLOSING` $\to$ `IDLE`.
2. **Engagement Decay Oracle:** Set boredom high ($k=5$), verify engagement drops below 0.20, triggering `ABANDONED` with exact snub penalty $\Delta \text{affinity} = -0.10$.
3. **Emergency Interruption Invariant:** Inject thunderstorm alert ($\text{threat} = 0.90$); verify immediate transition to `INTERRUPTED` and assert victim affinity is NOT penalized.

---

## 9. Known Limitations
- Conversations are currently strictly dyadic (pairwise); multi-party group banter is modeled as public assembly speeches rather than shared FSMs.

---

## 10. References / Citations
- Goffman, E. (1967). *Interaction Ritual: Essays on Face-to-Face Behavior*. Doubleday Anchor, pp. 1-45.
- Clark, H. H. (1996). *Using Language*. Cambridge University Press, Chapters 1-3.
- Park, J. S., et al. (2023). Generative agents: Interactive simulacra of human behavior. *UIST '23*, pp. 1-22.
