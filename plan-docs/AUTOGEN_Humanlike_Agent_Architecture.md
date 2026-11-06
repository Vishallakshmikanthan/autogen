# AUTOGEN: Making Villagers Attentive, Social, Reflective and Nuanced

A concrete design for upgrading the 10-villager society without throwing away your utility engine.

---

## 0. Where you are today (from the uploaded repo)

I read `agents/village/villager_agent.py`, `controller.py`, `intelligence/village/*` and `memory/*`. Relevant facts:

| Capability | What exists | Gap |
|---|---|---|
| Decision | `UtilityEngine.evaluate_action` → argmax over candidates | Memoryless per tick: no commitment, no interruption concept |
| Perception | `VillageObservation` (vision radius, `visible_villagers`, `witnessed_outcomes`) | No sounds, weather events, or "what changed since last tick" |
| Emotion | `AffectivePADEngine.appraise_observation` + `step_pad_state`, `compute_utility_modifiers` | Only appraises own needs and witnessed outcomes; no contagion between agents |
| Social | `VillageTrustGraph` with `record_public_event`, `propagate_gossip`, PageRank status | Gossip is a graph op, not an in-world act with a speaker, listener, content, and distortion |
| Dialogue | `VillageDialogueBus` + `SymbolicDialogueEngine` + `LLMPersonaBridge` | Dialogue is decoration after the decision (Phase 6); it does not feed back into goals or relationships |
| Memory | `VillagerBeliefState.episodic_memories: tuple[str, ...]` (last 15 strings) | Raw strings; `memory/episodic_memory.py` and `retrieval.py` exist but villagers do not use them |

Two small bugs worth fixing before you build on top:

1. In `VillagerAgent.__init__`, `self.affective_state` is assigned twice. The second line (`AffectiveStateVector()`) overwrites the one carrying `agent_id`.
2. In `update_belief`, the dedupe check `out.description not in memories` compares the raw description against entries stored as `"[Step n] description"`, so it never matches and duplicates accumulate.

`VillageObservation` and `VillagerBeliefState` are frozen Pydantic models with `extra="forbid"`. Every new field below must be added to the schema explicitly.

---

## 1. Target architecture

Keep your utility engine as the arbiter. Add layers around it that (a) change what the agent perceives, (b) change the weights and candidate set, and (c) give behavior temporal continuity.

```
                 ┌──────────────────────────────────────────────┐
 World events →  │ 1. PERCEPTION + SALIENCE FILTER              │
 (sights,sounds, │    Percept[] → salience score → attention     │
  weather,gossip)│    budget → "Alerts" that can interrupt       │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ 2. APPRAISAL                                  │
                 │    OCC appraisal → PAD ODE (+ contagion term) │
                 │    Needs/drives (Maslow-style pressure)       │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ 3. PERSONALITY MODULATION (Big Five)          │
                 │    traits → utility weights, salience gains,  │
                 │    PAD baseline, talkativeness, gossip bias   │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ 4. INTENTION LAYER (utility-guided BT/HTN)    │
                 │    current Intention + commitment + switching │
                 │    cost; Alerts can preempt                   │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ 5. UTILITY ARGMAX (your existing engine)      │
                 │    + u_react + u_rel − switching_cost         │
                 └───────────────┬──────────────────────────────┘
                                 ▼
          Action  ──►  World  ──►  Outcomes ──►  6. MEMORY STREAM ──► REFLECTION
                                                  (affect-tagged)      (at rest/sleep)
                                   ▲                                       │
                                   └──── dispositions, grudges, beliefs ───┘
                                      feed back into steps 1-5
```

Design rules:

- **Deterministic by seed.** Use a per-agent `random.Random(seed ^ hash(agent_id))`. Your controller already computes a canonical hash; keep it valid.
- **Symbolic core, LLM optional.** All decisions, memory updates, and reflections are rule-based. The LLM only renders surface text for utterances and thought bubbles.
- **Belief/ground-truth separation.** Agents only see percepts the world grants them (AGENTS.md rule 6 in your repo).

---

## 2. Attentiveness and salience

### 2.1 Problem

Today an agent picks an action each tick from the same candidate set and utility terms regardless of what just happened. There is no notion of "something startling occurred, drop what I'm doing."

### 2.2 World-side: emit events, not just states

Add an event channel to `VillageWorld`. Each tick the world appends `WorldEvent`s, and each agent receives the subset it can sense.

```python
class Modality(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    ENVIRONMENTAL = "environmental"   # weather, fire, drought onset
    SOCIAL = "social"                 # someone addressed me

class WorldEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_id: str
    step: int
    kind: str                # "CRY_OUT", "THUNDER", "CHASE", "ARRIVAL", "THEFT_SEEN", "GREETING", ...
    modality: Modality
    source_id: str | None
    target_id: str | None
    position: Position
    intensity: float         # 0..1 (loudness, brightness, severity)
    valence_hint: float      # -1 (threat/harm) .. +1 (pleasant)
    payload: dict[str, Any] = {}
```

Sources of events:

- `step_village` emits `CRY_OUT` when a villager's health or hunger crosses a crisis threshold, or when they are stolen from or caught.
- A guard in pursuit emits `CHASE` (visual + shouted `HALT` auditory).
- `ArrivalEvent` when a villager enters another's vision radius, or passes within 2 tiles.
- Weather: a stochastic weather process emits `THUNDER` (global, high intensity) and `RAIN_ONSET`.
- `GREETING` and conversation turns (Section 3) are also events.

Add to `VillageObservation`:

```python
percepts: tuple[PerceivedEvent, ...] = ()   # filtered by sensing range/occlusion
```

Audibility can reuse the formula in `VillageDialogueBus.compute_audibility`. Do not duplicate it; call it.

### 2.3 Agent-side: salience scoring

For each percept `p` and agent `i`:

```
S_i(p) = g_mod(p) * I(p) * prox(p) * N_i(p) * (1 + R_i(p)) * (1 + A_i)
```

| Term | Meaning | Suggested form |
|---|---|---|
| `g_mod` | Modality gain | auditory 1.0, visual 0.8, environmental 1.2 |
| `I(p)` | Intensity | event intensity 0..1 |
| `prox` | Proximity | `exp(-d / L)` with `L`≈6 tiles; thunder uses `prox=1` |
| `N_i` | Novelty | `1 - habituation[kind]`; habituation rises with repeats, decays with time |
| `R_i` | Relevance | +1.0 if source is a kin/friend/enemy (from relationship store), +0.5 if it touches my current goal zone, +1.0 if it threatens something I own |
| `A_i` | Arousal boost | current PAD arousal clipped to [0,1]; a tense agent startles more |

Personality (Section 5) scales it: `S_i *= (1 + 0.5*Neuroticism_i)` for threats, `(1 + 0.5*Openness_i)` for novel non-threat events, `(1 + 0.5*Agreeableness_i)` for distress calls.

### 2.4 Attention budget and thresholds

Humans do not react to everything. Model limited attention:

```python
ATTN_BUDGET = 2          # max percepts processed to "Alert" per tick
THETA_NOTICE = 0.25      # above: remembered as a passing observation
THETA_ALERT  = 0.55      # above: may interrupt the current intention
THETA_STARTLE = 1.1      # above: involuntary reflex (freeze/flinch) next tick

def attend(agent, percepts):
    scored = sorted(((salience(agent, p), p) for p in percepts), key=lambda t: -t[0])
    alerts, notices = [], []
    for s, p in scored[:ATTN_BUDGET]:
        if s >= THETA_ALERT:  alerts.append(Alert(p, s))
        elif s >= THETA_NOTICE: notices.append(p)
    for p in notices: agent.memory.add_observation(p, salience=s)
    return alerts
```

Add a reaction latency so reactions are not frame-perfect. Alerts become eligible after `lat = base_latency + noise` ticks. With 0.5 h/tick, use 0 ticks for startle events and 1 tick for ordinary alerts (a "double-take"). The frontend can use this window to play a head-turn animation.

### 2.5 Interruption logic

Interruption is a decision, not a hard override. Use commitment with switching cost, so agents are not jittery.

```python
@dataclass
class Intention:
    kind: str              # "WORK_FIELD", "FETCH_WATER", "CONVERSE", "FLEE", ...
    goal_zone: str
    started_step: int
    commitment: float      # 0..1, decays as time passes or progress stalls
    interruptible: bool    # False for e.g. mid-pillory, mid-vote
    progress: float

def should_interrupt(agent, alert, intention):
    if not intention.interruptible and alert.salience < THETA_STARTLE:
        return False
    # Urgency of the alert vs. stickiness of what I'm doing
    urgency = alert.salience * (1 + alert.threat)          # threat: 0..1
    stickiness = intention.commitment * (1 + 0.4*agent.traits.conscientiousness) \
                 + (0.3 if intention.progress > 0.8 else 0.0)   # almost done: finish it
    # Hysteresis margin prevents flip-flopping
    return urgency > stickiness + HYSTERESIS   # e.g., 0.15
```

Fold this into the utility argmax rather than bypassing it. Two additions to `evaluate_action`:

```
U'(a) = U(a)                                   # your existing multi-objective utility
      + λ_att * u_react(a | alerts)            # reaction value of action a
      - C_switch * 1[a does not serve current intention]
```

with

- `u_react(a)` is high for actions that address an alert: `INVESTIGATE_NOISE` for a cry, `SHELTER` for thunder, `FLEE` when chased and guilty, `ALERT_GUARD` when you saw a theft, `GREET` for a neighbor passing by. It is scaled by salience, empathy (Agreeableness), and relationship to the source.
- `C_switch = κ_sw * commitment`, typically 2-6 utility points on your scale, where `w_soc=14` and `w_phys=2.2` imply units of roughly 1-30 for typical actions. Calibrate empirically (Section 8).

New candidate actions to add to `VillageActionType` (and to `evaluate_all_pathways`):

| Action | Trigger | Effect |
|---|---|---|
| `INVESTIGATE_NOISE` | auditory alert from unknown/known source | move toward source; on arrival, gain observation |
| `SHELTER` | thunder / heavy rain | go to nearest hut/chapel; drop outdoor intention |
| `FLEE` | threat alert and guilty or low `Dominance` | move away, +arousal |
| `RAISE_ALARM` | witnessed crime / chase | emits `CRY_OUT`; sets `notified` on guards |
| `ASSIST_VILLAGER` | distress call from friend/kin | travel; transfer water/food (links to `COMMUNITY_SERVICE`) |
| `GREET` / `ACKNOWLEDGE` | someone passed within 2 tiles | opens conversation FSM (Section 3) |
| `GAZE_AT` (non-blocking) | any alert | purely cosmetic head/eye turn, costs no tick |

`GAZE_AT` is the cheap trick that makes the biggest visual difference: it is a presentation-layer orientation toward the highest-salience percept, emitted as telemetry even when the agent does not change behavior.

### 2.6 Walkthrough: thunderstorm

1. Tick t: world emits `THUNDER(intensity=0.9, global)`.
2. Farmer (Neuroticism 0.7, Arousal 0.1): `S = 1.2 * 0.9 * 1 * 1 * (1+0) * 1.1 * (1+0.35)` ≈ 1.6 > `THETA_STARTLE`: flinch (PAD arousal spike), gaze to sky.
3. t+1: `should_interrupt` passes (urgency > stickiness). `SHELTER` gets `u_react` bonus; `HARVEST_FIELD` is penalized by `C_switch` only if commitment is high, and a lightning risk term in `u_phys` makes outdoor work unattractive anyway.
4. Merchant (Conscientiousness 0.9, mid-trade): the trade intention is `interruptible=False` until the haggle resolves, unless S exceeds startle. Result: a visible hesitation, then packing up. Different agents, different timing.

---

## 3. Social and conversational responsiveness

### 3.1 Conversation as a multi-tick, interruptible state machine

Make `CONVERSE` an intention with its own FSM. Both participants hold a shared `Conversation` object.

```
IDLE ──(notice + approach decision)──► APPROACHING
APPROACHING ──(within 2 tiles, eye contact)──► GREETING
GREETING ──(acknowledged)──► TALKING  ──(turn loop)──► CLOSING ──► IDLE
         └─(ignored / snub)──► IDLE   (relationship penalty)
TALKING ──(Alert with urgency > engagement)──► INTERRUPTED ──► (resume | abandon)
```

```python
@dataclass
class Conversation:
    conv_id: str
    participants: tuple[str, str]       # extend to n-party later
    state: ConvState
    topic_stack: list[Topic]            # what's being discussed
    turn_owner: str
    turns: int
    engagement: dict[str, float]        # per-participant 0..1
    started_step: int
```

**Engagement** decays each turn and is boosted by relevance and liking:

```
engagement_i ← clip( engagement_i + 0.15*topic_relevance + 0.10*affinity_ij
                     − 0.08 − 0.10*boredom_i, 0, 1 )
```

A participant leaves when `engagement_i < θ_leave` or when an Alert beats `engagement_i * stickiness_conv`. Leaving mid-sentence is a *social event*: the other agent perceives `CONVERSATION_ABANDONED` and may feel snubbed (negative PAD pleasure, small trust drop) unless the cause was a visible emergency (a shared, observed alert negates the snub; both then react together).

### 3.2 Dialogue acts chosen by utility

Instead of picking text, pick a *dialogue act*. Text is rendered afterward by `SymbolicDialogueEngine` or the LLM bridge.

| Act | When | Effects |
|---|---|---|
| `GREET` | passing a known villager | familiarity +, mood contagion small |
| `SMALL_TALK` | high Extraversion, low urgency | affinity + small, boredom − |
| `SHARE_NEWS` | I hold fresh high-salience memory | transfers a `Belief` (3.3) |
| `GOSSIP` | news concerns a third party, I like/trust listener | transfers belief with spin |
| `ASK_FAVOR` | I have a need and listener has resource | triggers `TRADE`/`ASSIST` negotiation |
| `COMFORT` | listener has strong negative pleasure | their PAD += empathy term; affinity + |
| `COMPLAIN` / `ACCUSE` | grudge toward third party/listener | may lower listener's trust in target |
| `APOLOGIZE` | I wronged listener and Agreeableness high | grudge decay accelerated |
| `FAREWELL` | closing | ends cleanly |

Act selection reuses your utility structure:

```
U_act(d) = w_soc * Δstatus(d) + w_rel * Δaffinity(d) + w_info * info_value(d)
         + w_mood * mood_congruence(d) − cost(d) − I(risky) * θ_risk * P_overheard * κ
```

`P_overheard` matters: gossiping about the guard captain within earshot of the guard is risky. `VillageDialogueBus.compute_audibility` already gives you the overhearing probability, so use it.

### 3.3 Gossip as a first-class, lossy, provenance-tracked process

Today `trust_graph.propagate_gossip` is a matrix operation. Replace the trigger with in-world acts and keep the graph update as the *consequence*.

```python
@dataclass
class Belief:
    belief_id: str
    subject: str                # who it's about
    claim: str                  # "stole_grain", "generous", "ill", "price_high:WHEAT"
    valence: float              # -1..1
    confidence: float           # 0..1
    source_chain: list[str]     # origin → ... → me   (provenance)
    first_hand: bool
    heard_at_step: int
    times_heard: int            # corroboration count
```

**Transmission** from speaker `s` to listener `l`:

```python
def transmit(s, l, belief):
    # Speaker's distortion: exaggeration grows with arousal and dislike of subject
    spin = 1.0 + 0.25*s.pad.arousal + 0.20*max(0, -s.affinity[belief.subject])
    sent_conf = clip(belief.confidence * (0.9 if not belief.first_hand else 1.0), 0, 1)
    sent_val  = clip(belief.valence * spin, -1, 1)

    # Listener's credibility of the SOURCE (not the claim)
    cred = (0.5*l.trust[s.id] + 0.3*status_norm(s) + 0.2*l.affinity[s.id])
    # Prior consistency: contradicts what I know firsthand?
    prior = l.beliefs.get(belief.subject, belief.claim)
    consistency = 1.0 if prior is None else (1.0 - abs(prior.valence - sent_val)/2)
    skepticism = 0.5 + 0.5*l.traits.openness_inverse  # low Openness → more skeptical of novelty

    accept_p = sigmoid(4*(cred*consistency/skepticism - 0.5))
    if rng.random() < accept_p:
        l.beliefs.merge(belief.subject, belief.claim, sent_val,
                        confidence=sent_conf*cred,
                        source_chain=belief.source_chain + [s.id])
        l.trust_update_via_gossip(belief.subject, sent_val, weight=cred*sent_conf)
```

Corroboration: hearing the same claim from two independent chains (disjoint `source_chain`) raises `confidence` via noisy-OR: `c = 1 - (1-c1)(1-c2)`. Chains that loop back to the listener are discounted (rumor echo).

Properties that make gossip feel real:

- Exaggeration and decay: confidence drops per hop; valence can inflate.
- Reputation loop: a villager who is the subject of negative gossip receives lower trust from people who never met them, and this feeds PageRank status. This gives an emergent mechanism for social hierarchy that does not require every witness to see every event.
- Gossip can be **wrong**. A slandered villager's pillory risk and trade terms change. That creates narrative, and a natural lever for the Gram Sabha to investigate (`INVESTIGATE_CRIME` can reduce false-belief confidence).

### 3.4 Emotional contagion

Your PAD update is an ODE decaying toward baseline. Add a coupling term from nearby agents:

```
dP_i/dt = −α (P_i − B_i)                      # decay to baseline (existing)
        + Σ_k stimuli_k                       # appraisal impulses (existing)
        + γ_i Σ_{j∈N(i)} w_ij (P_j − P_i)     # NEW: contagion
```

where

- `N(i)` is villagers within hearing/vision range that are visibly expressing (use emote telemetry: `expressed_P_j`, which is `P_j` filtered through the *expressiveness* of j, so stoic villagers leak less).
- `w_ij = prox(d_ij) * (0.5 + 0.5*affinity_ij) * attention_ij` — contagion needs attention; you do not catch moods from people you did not notice.
- `γ_i = γ_0 * (0.6 + 0.8*Extraversion_i*0.5 + 0.8*Agreeableness_i*0.5)`; start with `γ_0≈0.08` per tick.
- Asymmetric weighting: negative arousal spreads faster (`w *= 1.3` when the source has pleasure < −0.3), matching the empirical negativity bias, and it produces panic cascades you can tune.

Implement as a second pass after all agents' own appraisal so the result is order-independent: compute all `expressed_P` first, then apply contagion with a synchronous update.

### 3.5 Conversations change goals and relationships

After each conversation, run a **post-conversation update** (also triggered on interruption):

```python
def settle_conversation(conv):
    for me, other in ((a, b), (b, a)):
        quality = mean_valence_of_acts(conv) + 0.5*(engagement_end[me] - 0.5)
        me.rel[other].affinity    += η * quality * (1 - me.rel[other].affinity)  # saturating
        me.rel[other].familiarity += 0.05
        me.rel[other].trust       += trust_delta_from_acts(conv, me)
        me.memory.add_episode(Episode(kind="conversation", with_=other,
                                      valence=quality, pad_snapshot=me.pad, ...))
        # Goal side effects
        if conv.has_act("ASK_FAVOR") and conv.accepted:  me.intentions.push(Obligation(to=other))
        if conv.has_act("SHARE_NEWS"): me.goals.add_from(belief)   # e.g., "avoid the north field"
```

Relationships need more than one number. Use four per directed pair:

| Dimension | Meaning | Moves with |
|---|---|---|
| `trust` | reliability / honesty (your existing graph) | cooperation, defection, gossip |
| `affinity` | liking / warmth | conversation quality, shared hardship, contagion |
| `respect` | competence / status | status rank, observed skill |
| `familiarity` | contact frequency | every interaction; decays slowly |

`affinity` feeds `u_rel` in act selection, `R_i` in salience, and `w_ij` in contagion. `trust` stays as the PageRank input so your institutions code does not change.

---

## 4. Memory and reflection

### 4.1 Replace strings with a structured memory stream

Wire `memory/episodic_memory.py` and `retrieval.py` into villagers (or write a village-specific variant). Memory item:

```python
@dataclass
class MemoryItem:
    mid: str
    step: int
    kind: Literal["observation", "action", "conversation", "feeling", "reflection", "belief"]
    actors: tuple[str, ...]         # who was involved
    location: str                   # zone id
    summary: str                    # short symbolic description
    tags: frozenset[str]            # {"theft","water","storm","kindness",...}
    valence: float                  # -1..1
    pad: tuple[float, float, float] # emotion at encoding (mood-tagged)
    importance: float               # 0..1 (computed once at write time)
    access_count: int
    last_access_step: int
    evidence_of: list[str] = []     # for reflections: ids of source memories
```

**Importance at write time** (no LLM needed):

```
importance = clip( 0.35*|valence| + 0.25*arousal + 0.20*self_involvement
                 + 0.10*rarity + 0.10*consequence, 0, 1 )
```

where `self_involvement` is 1 if I was actor/target, 0.5 if I witnessed, and `consequence` is derived from stat/coin/trust deltas.

### 4.2 Retrieval: recency + importance + relevance (+ mood)

Following the retrieval scoring popularized by Park et al.'s "Generative Agents" (2023; I can't verify citations from here, so please double-check the reference) and extending it:

```
score(m | query q) =  w_r * exp(−(t − m.last_access)/τ_r)
                    + w_i * m.importance
                    + w_q * relevance(m, q)          # tag/actor/location overlap (Jaccard); or embedding sim
                    + w_m * mood_congruence(m.pad, agent.pad)   # sad agents recall sad things
```

Typical weights: `(1.0, 1.0, 1.5, 0.4)`. Query `q` is built from the current context: nearby villagers, zone, active alerts, current intention. This is what makes a grudge surface exactly when you meet the person you resent.

Use it in three places: (1) Utility terms (`u_rel` pulls relevant memories about the other party), (2) conversation topic selection, (3) reflection input.

### 4.3 Forgetting and rehearsal

```
strength(m, t) = importance * exp(−(t − m.last_access) / S) * (1 + 0.3*log1p(m.access_count))
```

Items under `strength < ε` are pruned or compressed into a reflection. Rehearsal (retrieval) resets recency, so repeatedly recalled grudges persist, which models rumination. Cap the stream at ~300 items per agent.

### 4.4 Reflection: turning episodes into dispositions

Trigger reflection:

- when `REST_HUT` completes (a "sleep consolidation" step, narratively clean), or
- when the sum of `importance` of unreflected memories exceeds a threshold (≈ 2.5, from the Park et al. idea), or
- after a very high-importance event (betrayal, rescue, being pilloried).

Rule-based reflection pipeline:

```python
def reflect(agent):
    recent = agent.memory.unreflected()
    groups = group_by_actor(recent)                       # who was involved
    for other, mems in groups.items():
        evidence = top_k(mems, key=importance, k=8)
        net_valence = weighted_mean(m.valence, w=m.importance for m in evidence)
        harm   = sum(m.importance for m in evidence if "harm_to_me" in m.tags)
        help_  = sum(m.importance for m in evidence if "help_to_me" in m.tags)
        n      = len(evidence)

        d = agent.dispositions.get(other) or Disposition(other)
        # Slow integration: dispositions move gradually, anchored to history
        d.valence = 0.85*d.valence + 0.15*net_valence
        d.grudge  = clip(d.grudge  + 0.6*harm  - 0.4*help_*(d.grudge>0), 0, 1)
        d.fondness= clip(d.fondness+ 0.6*help_ - 0.2*harm, 0, 1)
        d.confidence = 1 - exp(-0.4 * (d.evidence_count + n))
        d.evidence_ids = merge(d.evidence_ids, ids(evidence))
        agent.dispositions[other] = d

        # Emit a reflection memory (this is what the UI shows as an inner thought)
        agent.memory.add(MemoryItem(kind="reflection",
            summary=render_reflection(agent, other, d),     # template, or LLM-rendered
            importance=0.6 + 0.3*d.confidence, evidence_of=ids(evidence), ...))

    # Higher-order reflections: patterns across people/places
    for pat in detect_patterns(recent):    # e.g., "thefts cluster near granary at night"
        agent.beliefs.add_world_belief(pat); agent.memory.add(reflection_for(pat))
```

Grudges and friendships, concretely:

- **Disposition** is a durable per-person summary (`valence`, `grudge`, `fondness`, `confidence`, `evidence_ids`) that *reads from* the memory stream and *writes to* `affinity`/`trust` baselines. Evidence ids give explainability for the UI ("Why does Arun distrust Mani? Because of memories #231, #240").
- **Asymmetry of weight**: harm is weighted higher than help (negativity bias) but grudges *decay* (`d.grudge *= 0.995` per tick, slower for high Neuroticism) and an `APOLOGIZE` or `COMMUNITY_SERVICE` event by the offender triggers a *forgiveness check* (`P_forgive = f(Agreeableness, offender's sincerity, time elapsed)`), producing reconciliation arcs.
- **Attribution**: reflect on *why*. If the harm happened while the other was in crisis (starving), reduce blame via `situational_attribution = need_pressure_of_other_at_time`, so Agreeable agents forgive thieves who were starving. This echoes the actor-observer asymmetry and creates sharper moral dynamics with the Gram Sabha.
- **Optional LLM upgrade**: pass `evidence` summaries to your `LLMPersonaBridge` to *phrase* reflections and ask it to propose candidate higher-order insights, but gate the result through the rule-based validator (must cite evidence ids; valence bounded). The simulation state never depends on free text.

### 4.5 How memory feeds back into behavior

| Memory-derived signal | Used in |
|---|---|
| `grudge`, `fondness`, `valence` | `u_rel`, act selection, trade price concessions, who to help, who to avoid (path planning adds a repulsion cost near grudge targets) |
| World beliefs ("granary thefts at night") | `believed_detection_risk`, guard patrol targets, locking huts |
| Reflected preferences ("market is cheapest at dawn") | candidate scoring bonus; routine shaping |
| Mood-tagged retrieval | affect-congruent conversation topics and rumination |

---

## 5. Behavioral nuance: complementary psychological architectures

### 5.1 Recommendation

Do not replace the utility engine. Use three complementary mechanisms, each doing a different job:

| Mechanism | Job | Plugs into |
|---|---|---|
| **Big Five** | Stable individual differences | Utility weights, PAD baseline, salience gains, social propensities |
| **Drive/needs hierarchy** (Maslow-flavored, with drive reduction and satiation) | Time-varying motivation | Dynamic `w_phys`, `w_soc`, new need terms |
| **Utility-guided behavior tree / HTN for intentions** | Temporal continuity, routines, commitment | Intention layer; utility arbitrates at selector nodes |

### 5.2 Big Five → engine parameters

Each trait in `[0,1]`, 0.5 = average.

| Trait | Utility / engine mapping | Affect mapping | Social mapping |
|---|---|---|---|
| **Openness (O)** | exploration bonus on novel actions (`FORAGE`, new trade partners) `+0.8*O` | baseline Arousal `+0.1*(O−.5)` | gossip acceptance (less skeptical) |
| **Conscientiousness (C)** | `C_switch` scale `(1+0.4C)`, routine adherence, lower `θ_risk`, longer-horizon discount | lower arousal volatility | keeps promises (`Obligation` weight) |
| **Extraversion (E)** | `w_soc ×(0.7+0.6E)`, initiates `GREET`/`SMALL_TALK` | baseline Pleasure `+0.15*(E−.5)`, Dominance `+` | talkativeness, contagion γ, longer conversations |
| **Agreeableness (A)** | `u_react` for distress calls; cooperation bias in `TITHE`/`ASSIST`; lowers `STEAL` utility via guilt term | Pleasure sensitivity to others' pain | forgiveness rate, less gossip spin |
| **Neuroticism (N)** | raises `θ`-aversion (risk aversion) and threat-salience gain `(1+0.5N)`; strengthens crisis penalty κ | PAD decay rate slower for negative states; baseline Pleasure `−0.15*(N−.5)` | grudge persistence, ruminative retrieval |

Mehrabian's mapping from Big Five to PAD baselines provides a literature-grounded starting point; treat the coefficients above as tunable priors, not facts.

A single function turns a trait vector into your existing `VillagerUtilityWeights` plus a new `PersonalityParams`:

```python
def weights_from_traits(t: Traits, role: VillageRole) -> tuple[VillagerUtilityWeights, PersonalityParams]:
    w = VillagerUtilityWeights(
        w_phys = 2.2,
        w_econ = 1.0 * (0.8 + 0.4*t.conscientiousness) * role_econ_mult[role],
        w_soc  = 14.0 * (0.7 + 0.6*t.extraversion),
        theta_risk = clip(0.25 + 0.35*(t.openness-0.5) - 0.30*(t.conscientiousness-0.5)
                          - 0.25*(t.agreeableness-0.5) - 0.20*(t.neuroticism-0.5), 0.02, 0.9),
        crisis_penalty_kappa = 18.0 * (0.8 + 0.5*t.neuroticism),
    )
    p = PersonalityParams(
        salience_threat_gain = 1 + 0.5*t.neuroticism,
        salience_novel_gain  = 1 + 0.5*t.openness,
        empathy_gain         = 1 + 0.5*t.agreeableness,
        switching_cost_scale = 1 + 0.4*t.conscientiousness,
        contagion_gamma      = 0.08*(0.6 + 0.4*t.extraversion + 0.4*t.agreeableness),
        grudge_decay         = 0.995 + 0.004*t.neuroticism - 0.003*t.agreeableness,
        talkativeness        = 0.2 + 0.7*t.extraversion,
        pad_baseline         = pad_baseline_from(t),
    )
    return w, p
```

Starting archetypes for your 10 villagers (roles you already have; trait vectors are suggestions to spread behavior, tune freely):

| Archetype | O | C | E | A | N | What it should produce |
|---|---|---|---|---|---|---|
| Diligent Farmer | .3 | .9 | .4 | .6 | .3 | rarely interrupted, finishes rows, quiet loyalty |
| Anxious Farmer | .4 | .6 | .3 | .5 | .8 | first to shelter, panic contagion source, holds grudges |
| Gossipy Water-Carrier | .5 | .5 | .9 | .5 | .5 | gossip hub; passes the well daily; high PageRank for info flow |
| Stoic Water-Carrier | .3 | .8 | .2 | .6 | .2 | stable, low contagion, trusted neutral |
| Shrewd Merchant | .7 | .7 | .7 | .3 | .4 | haggles, uses gossip strategically, higher `θ_risk` |
| Generous Merchant | .6 | .5 | .8 | .9 | .4 | gives credit, forgives, comforts |
| Stern Guard | .2 | .9 | .4 | .3 | .4 | strong threat salience for crime, low gossip spin |
| Lenient Guard | .5 | .6 | .6 | .8 | .3 | de-escalates, uses warnings, vulnerable to pleading |
| Curious Wanderer | .95 | .3 | .7 | .6 | .3 | investigates every noise, carries news between zones |
| Drifter Wanderer | .6 | .2 | .3 | .3 | .7 | high `θ_risk`, grudge-prone, most likely to `STEAL` |

### 5.3 Needs and drives: dynamic weights instead of static ones

Your `w_phys` is a constant. Real motivation shifts with deprivation. Use a prepotency (Maslow-style) scheme with soft gating, not hard levels:

```
Levels:  L1 physiological  (hunger, hydration, energy, health)
         L2 safety         (shelter, stored food, low perceived threat, being in good standing with law)
         L3 belonging      (recent positive contact, family/friends nearby, loneliness drive)
         L4 esteem         (status S_i, respect, competence feedback)
         L5 growth         (exploration, craft mastery, curiosity)

Deficit_k ∈ [0,1],  Pressure_k = Deficit_k^p   (p≈2: pressure rises sharply when deficit is large)

Prepotency gate:  g_k = Π_{j<k} (1 − Satisfaction_j)^β   ... NO: use  g_k = Π_{j<k} sat_j^β   (β≈0.7)
w_k(t) = base_w_k * (1 + λ * Pressure_k) * g_k
```

So a starving villager's belonging and esteem weights shrink (`g_k → 0` because `sat_1` is low) and physiology dominates, but a moderately hungry villager still cares about talking to a friend. Map onto your engine:

- `w_phys` ← `w_L1`; `w_soc` ← blend of `w_L3` and `w_L4`; add `w_safety` and `w_growth` terms to `u(a)`.
- **Loneliness drive**: `deficit_L3 = clip(1 − recent_contact_score/target, 0, 1)` where `recent_contact_score` is an exponentially-decayed sum of conversation quality. This single drive makes agents seek company on their own and gives social behavior intrinsic motivation instead of only utilitarian status gain.
- **Drive reduction with satiation**: an action reduces its target deficit (`Δdeficit = −gain * (1 − habituation)`), and repeated use of the same satisfier within a short window reduces its marginal value. This stops agents from, for example, chatting nonstop or trading in circles.

### 5.4 Utility-guided behavior tree for intentions

Pure per-tick argmax has no memory of *what I was in the middle of*. Add a thin hierarchical layer, but let utility score the branches, so you keep one decision principle:

```
Root (Utility Selector over Intentions)
├── SURVIVE            (score from L1/L2 pressure)
│   ├── Sequence: GetWater → [goto well] → DRAW_WATER → CONSUME
│   └── Sequence: GetFood  → [goto granary|field|market] → ...
├── WORK (role routine)
│   └── Farmer:  goto field → HARVEST_FIELD ×N → carry to granary → TITHE?/TRADE
├── SOCIALIZE          (score from L3 loneliness + opportunity)
│   └── Sequence: ChoosePartner → Approach → CONVERSE → Close
├── RESPOND            (score only when Alerts present; can outscore all)
│   └── Selector: Flee | Shelter | Investigate | Assist | RaiseAlarm
├── REST               (score from energy + time of day)
└── ILLICIT / CIVIC    (STEAL, ATTEND_SABHA, ...) gated by existing risk model
```

Mechanics:

1. Each Intention node has a score `Score(I) = Σ_k w_k(t)·Δdeficit_k(I) + expected U from your engine for its first action − C_switch·1[I≠current]`.
2. The agent *keeps* the current Intention until it completes, fails, or a rival's score exceeds `current + hysteresis` (the `should_interrupt` logic above).
3. Within an Intention, the next primitive is chosen with your existing `rank_candidate_actions`, restricted to the actions that serve that Intention plus always-available safety actions (consume, flee). This means crime or desperation can still break routines by outscoring them, consistent with your existing ethics model.
4. Time-of-day priors (`time_of_day` is already in the observation) bias routine scores: work in the morning, market at midday, socialize at dusk near the well or chapel, rest at night. This produces visible daily rhythms in the 3D twin at near-zero cost.

### 5.5 Habit and routine strength

Add a habit table per agent: `habit[(context_key, intention)] += α` on completion, decaying slowly. A habit adds a small constant to that Intention's score in that context (time-of-day + zone). Habits produce believable repeated paths (the Gossipy Water-Carrier at the well every dusk), make behavior predictable enough for other agents to learn expectations, and give a clean hook for "breaking routine" as a salient novelty (someone absent from their usual place is itself an event worth noticing and gossiping about).

---

## 6. Revised tick pipeline

Keep your phase structure in `VillageSimulationController.step()`; insert the new work as marked.

```
Phase 0  (NEW) World event generation: weather, crisis cries, arrivals → events[t]
Phase 1  Sensory observation   (+ percepts filtered by sensing range)
Phase 2a Belief update          (existing; fix the two bugs from §0)
Phase 2b (NEW) Attention: salience scoring → alerts/notices; habituation update
Phase 2c Appraisal: own-needs + witnessed outcomes → PAD stimuli (existing)
Phase 2d (NEW) Emotional contagion pass (synchronous, after all agents' expressed PAD computed)
Phase 2e (NEW) Needs/drive update → dynamic weights; personality-modulated parameters
Phase 3  Deliberation:
           - intention arbitration (continue / interrupt / switch)
           - candidate generation (+ new reactive & social actions)
           - utility argmax with u_react, u_rel, C_switch
Phase 4  Environment physics & resolution (existing). Now also resolves conversation turns.
Phase 5  Trust graph + PageRank (existing) + relationship dimension updates
Phase 5b Justice/Sabha/lifecycle (existing)
Phase 5c (NEW) Memory write (affect-tagged), reflection triggers, grudge decay, forgiveness checks
Phase 6  Dialogue rendering (existing bus), now rendering chosen dialogue acts, not inventing them
Phase 7  (NEW) Telemetry: gaze target, emote, thought bubble, conversation links, alert flags
```

Performance: with 10 agents this is trivial (<1 ms per agent for salience, contagion `O(n²)`=100 pairs). The first cost that matters if you scale to 100+ agents is retrieval; add a per-actor index (`dict[actor_id, list[MemoryItem]]`) and cap at ~300 items. Use a spatial hash for percept delivery.

---

## 7. Telemetry for the Three.js frontend

Humanlike behavior is mostly *perceived* through animation. Extend the telemetry payload (`get_full_state_payload`) with a per-agent block:

```json
{
  "agent_id": "V03",
  "gaze": {"target_pos": [x, y], "target_id": "V07", "reason": "CRY_OUT"},
  "alert_level": 0.0,
  "emote": "startled|worried|laugh|angry|neutral",
  "intention": {"kind": "FETCH_WATER", "progress": 0.6, "interrupted_by": null},
  "conversation": {"with": "V05", "state": "TALKING", "act": "GOSSIP", "line": "…"},
  "thought": "Mani looked guilty near the granary again.",
  "relationships": {"V05": {"trust": 0.7, "affinity": 0.6, "grudge": 0.0}}
}
```

Frontend ideas that cost little:

- Head/eye turn toward `gaze.target_pos` with a ~200 ms ease. This is the single biggest "alive" cue.
- Emote billboards over heads, mapped from `dominant_emotion` (you already sync `pad_sync`).
- Lines between conversing agents; fade thickness by engagement.
- A click-to-inspect panel: show top memories, dispositions with evidence ids, and the utility breakdown (you already have `ActionUtilityEvaluation` rationale notes). Explainability will also sell the project as research.
- A "gossip trace" overlay: highlight the `source_chain` of a rumor as it travels.

---

## 8. Implementation roadmap

Phased so each step is demonstrable and testable on its own. File paths follow your package layout.

### Phase A: Foundations (1-2 days)
- Fix the two bugs in §0.
- Add `Traits` + `PersonalityParams` (`intelligence/village/personality.py`) and `weights_from_traits`; plumb into `VillageSimulationController._init_agents`.
- Add `WorldEvent` / `PerceivedEvent` schemas and the world event bus (`simulation/environment/village/events.py`); extend `VillageObservation.percepts`.
- Tests: trait → weights mapping is monotonic; determinism hash unchanged for identical seeds.

### Phase B: Attention and interruption (2-3 days)
- `intelligence/village/attention.py`: salience, habituation, attention budget, latency.
- `intelligence/village/intention.py`: `Intention`, commitment, `should_interrupt`.
- Add actions `SHELTER`, `INVESTIGATE_NOISE`, `FLEE`, `RAISE_ALARM`, `GAZE_AT`; extend `evaluate_all_pathways` with `u_react` and `C_switch`.
- Add a weather generator (thunderstorm) and `CRY_OUT` triggers.
- Acceptance tests: (1) thunder causes ≥70% of outdoor agents to change intention within 2 ticks; (2) a low-N, high-C agent mid-task reacts later than a high-N agent; (3) no oscillation: intention switches per agent per 20 ticks stays below a bound.

### Phase C: Structured memory and reflection (3-4 days)
- `intelligence/village/memory_stream.py` (or adapt `memory/episodic_memory.py` and `retrieval.py`); replace `episodic_memories: tuple[str, ...]` in `VillagerBeliefState` with a handle to the stream.
- Importance scoring, retrieval, forgetting.
- `intelligence/village/reflection.py`: dispositions, grudge/fondness, forgiveness; trigger at `REST_HUT` and by importance threshold.
- Feed `Disposition` into `affinity` and into trust baselines.
- Tests: a villager robbed 3× by the same culprit ends with grudge > 0.6 and avoids that path; a culprit who performs `COMMUNITY_SERVICE` reaches reconciliation in expectation; memory size stays bounded.

### Phase D: Conversation, gossip, contagion (4-5 days)
- `intelligence/village/conversation.py`: FSM, engagement, dialogue acts, interruption handling.
- `intelligence/village/gossip.py`: `Belief`, transmission, corroboration, distortion; call `trust_graph.record_interaction`/`propagate_gossip` as the *consequence*, not the cause.
- Contagion pass in the controller; use `AffectivePADEngine.step_pad_state` with the extra coupling term (extend `AffectivePADConfig` with `contagion_gamma`).
- Hook `SymbolicDialogueEngine` / `LLMPersonaBridge` to render chosen acts.
- Tests: a rumor seeded at one villager reaches ≥6/10 within N ticks on average; a panic emote seeded in a cluster spreads and then decays; conversation interrupted by thunder produces `CONVERSATION_ABANDONED` without a snub penalty.

### Phase E: Drives and intentions (3-4 days)
- `intelligence/village/needs.py`: deficits, prepotency gating, loneliness drive, satiation.
- Utility-guided behavior tree for Intentions (`intelligence/village/bt.py`); role routines, time-of-day priors, habits.
- Tests: daily rhythm autocorrelation (agents are at the market at similar hours day to day); loneliness drive makes isolated agents initiate conversation.

### Phase F: Telemetry and polish (2-3 days)
- Payload extensions (§7), frontend gaze/emotes/conversation links, inspect panel, gossip overlay.

### Evaluation (use your existing `research/ablation_arena.py`)

Make each layer an ablatable flag and report differences on the same seeds:

| Metric | What it shows |
|---|---|
| Reaction latency (ticks from event to intention change) | attentiveness |
| Interruption rate and flip-flop rate | stability vs jitter |
| Gossip reach, mean hops, and belief accuracy | information dynamics, error propagation |
| Mood variance across agents / contagion cascades | emotional coupling |
| Relationship graph: reciprocity, clustering, Gini of PageRank | social structure realism |
| Grudge/forgiveness cycle count, reconciliation time | memory-driven narrative |
| Crime rate and Sabha convocations vs baseline | do new layers destabilize institutions? |
| Behavioral diversity (entropy of action histograms by agent) | personality actually matters |

Report ablations for: no-salience, no-contagion, no-reflection, no-personality, no-BT. If an ablation changes nothing, the layer is decorative; retune or cut it.

### Pitfalls to watch

1. **Jitter.** Without hysteresis and `C_switch`, high-salience percepts cause flip-flopping. Tune hysteresis first.
2. **Runaway emotional cascades.** Cap total contagion input per tick and keep `γ` small; PAD decay must dominate at long horizons.
3. **Gossip death spirals.** Without corroboration thresholds and confidence decay, one false rumor can permanently lower a villager's status. Add forgetting and Sabha corrections.
4. **Over-modeling.** 10 agents will not show statistically smooth behavior. Use trait spread and seeded replays to make personalities readable instead of adding more parameters.
5. **Utility scale drift.** New terms (`u_react`, `u_rel`, needs) must be normalized against your existing magnitudes (`w_soc=14`). Log per-term contribution histograms and rescale so no single term dominates >60% of decisions.
6. **LLM leakage.** Keep the LLM downstream of decisions. If it ever feeds back state (e.g., generates a belief), validate against a schema and cite evidence ids.
7. **Determinism.** Draw randomness from per-agent seeded streams in sorted agent order (you already iterate `sorted(self.agents.keys())`), so replays and your canonical hash stay stable.

---

## 9. Minimal vertical slice (build this first)

If you want the fastest path to something visibly alive, build just this slice before everything else:

1. Big Five traits for all 10 villagers → weights (Phase A).
2. `CRY_OUT` + `THUNDER` events, salience, `GAZE_AT`, `SHELTER`, `INVESTIGATE_NOISE`, hysteresis (Phase B).
3. Greeting-only conversations with affinity updates and PAD contagion (a subset of Phase D).
4. A sleep-time reflection that creates one grudge/fondness per villager per night, shown in the inspect panel (a subset of Phase C).

That combination produces: heads turning toward screams, villagers hurrying for shelter at different times, friends greeting each other, and "why does she dislike him?" answers backed by evidence, which covers all four of your goals in a demo-sized package.

---

## 10. References (verify before citing)

I do not have search access in this session and may misremember details, so please verify these:

- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023): memory stream, recency/importance/relevance retrieval, reflection.
- Mehrabian, PAD temperament model and Big Five correspondences.
- Ortony, Clore, Collins (OCC): cognitive appraisal of emotion (already used in your `appraise_observation`).
- Hatfield, Cacioppo, Rapson: emotional contagion.
- Dunbar: gossip and social grooming.
- Maslow, "A Theory of Human Motivation" (1943); Hull's drive reduction theory.
- Behavior trees with utility selectors: Dave Mark's "Infinite Axis Utility System" talks (GDC) and HTN planning literature (Erol, Hendler, Nau).
