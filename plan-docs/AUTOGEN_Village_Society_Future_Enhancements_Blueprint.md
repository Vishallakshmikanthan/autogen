# AUTOGEN Simulated Village Society: Future Enhancements & Living Agentic Civilization Blueprint

> **Document:** `plan-docs/AUTOGEN_Village_Society_Future_Enhancements_Blueprint.md`  
> **Status:** Architecture Blueprint & Evolutionary Roadmap  
> **Target Milestone:** Gate 8+ (Multi-Agent Emergence, Social Dynamics, and Generative Civilization)  
> **Governing Standards:** [AGENTS.md](file:///c:/Users/Lenovo/Downloads/autogen/AGENTS.md) — Mathematical Correctness, Layer Decoupling, Explicit Seeding, and Deterministic Provability.

---

## 1. Executive Vision: From Simulated Agents to a Living Generative Civilization

In traditional games and decorative 3D simulations, non-player characters (NPCs) are state machines with scripted dialogues and hardcoded routes. In **AUTOGEN**, humans do not exist in the village—**the agents *are* the villagers**. Every action, word, glance, trade, theft, friendship, and feud is an emergent output of:
1. **Mathematical Physiological Needs** (homeostatic survival, hunger, hydration, stamina).
2. **Subjective Cognitive Beliefs** (mental map, incomplete observation, imperfect memory).
3. **Psychological & Emotional Dynamics** (joy, grief, jealousy, guilt, spite, benevolence).
4. **Relational Social Capital** (directional PageRank trust graph, lineage reputation).
5. **Downstream Multi-Agent Persona Dialogue** (living spoken natural language reflecting internal states).

This document outlines the evolutionary enhancements to transform AUTOGEN Village from a 10-agent economic prototype into a perpetual, self-sustaining **Generative Medieval Civilization**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUTOGEN CIVILIZATION ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. BIOLOGICAL & LIFECYCLE TIER                                              │
│     Birth · Maturation · Genetic Utility Inheritance · Aging · Mortality   │
├─────────────────────────────────────────────────────────────────────────────┤
│  2. AFFECTIVE EMOTIONAL DYNAMICS (OCC + PAD Model)                          │
│     Joy · Grief · Jealousy · Malevolence · Benevolence · Guilt · Resentment │
├─────────────────────────────────────────────────────────────────────────────┤
│  3. ROLE SPECIALIZATION & CASTE DIVISION                                    │
│     Elder · Farmer · Miller · Merchant · Blacksmith · Guard · Outcast · Bard│
├─────────────────────────────────────────────────────────────────────────────┤
│  4. INSTITUTIONAL SOCIAL MECHANICS                                          │
│     Village Assemblies (Sabha) · Court Trials · Granary Rationing · Funerals│
├─────────────────────────────────────────────────────────────────────────────┤
│  5. DOWNSTREAM MULTI-AGENT DIALOGUE & CINEMATIC 3D ANIMATION                │
│     Facial & Body Gestures · Alternating Turn-Taking · Directional Gossip   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Affective Emotional Dynamics: The 3D Valence-Arousal-Dominance (PAD) + OCC Model

To eliminate robotic behavior, every agent internalizes a continuous **Affective State Vector** $\mathbf{E}_i(t) \in [-1, 1]^3$ governing their subjective experience:

$$\mathbf{E}_i(t) = \begin{bmatrix} v_i(t) \\ a_i(t) \\ d_i(t) \end{bmatrix} \quad \begin{matrix} \text{Valence: Pleasure vs. Pain} \\ \text{Arousal: Alertness vs. Lethargy} \\ \text{Dominance: Empowerment vs. Helplessness} \end{matrix}$$

### 2.1 Discrete Emergent Emotions

The combination of $\mathbf{E}_i(t)$, memory, and recent social outcomes gives rise to explicit emotion categories:

| Emotion | Trigger Conditions | Affective Coordinates $(v, a, d)$ | Behavioral Impact |
| :--- | :--- | :--- | :--- |
| **Joy / Happiness** | Fair trade cleared, granary abundance, birth of child, wedding | $(+0.8, +0.5, +0.6)$ | Generous tithing, friendly greetings, open market bids |
| **Sadness / Grief** | Death of kinship/partner, failed harvest, status collapse | $(-0.8, -0.6, -0.7)$ | Slow movement, weeping animations, visiting chapel, withdrawn dialogue |
| **Jealousy / Envy** | Neighbor wealth exceeds self while self is hungry ($\Delta W > \tau$) | $(-0.5, +0.6, -0.3)$ | Rumor-spreading, boycotting trades, prime motivation for covert theft |
| **Evilness / Malice** | Sustained low trust, high risk tolerance, opportunistic payoff | $(-0.4, +0.7, +0.8)$ | Hoarding commons, price gouging, poisoning wells, sabotage |
| **Guilt / Remorse** | Committed theft or deceit against a trusted friend ($T_{j \to i} > 0.7$) | $(-0.6, +0.4, -0.6)$ | Covert restitution, self-isolation, increased detection anxiety |
| **Righteous Anger** | Catching a thief red-handed, breaking granary seal | $(-0.6, +0.9, +0.7)$ | Sounding alarm bell, escorting offender to pillory, public denunciation |

### 2.2 Mathematical Transition Equation

$$\mathbf{E}_i(t+1) = \lambda_{\text{decay}} \mathbf{E}_i(t) + (1 - \lambda_{\text{decay}}) \mathbf{E}_{\text{baseline}} + \sum_{k} \mathbf{w}_k \cdot \text{Stimulus}_k(t)$$

Where:
- $\lambda_{\text{decay}} \in [0.85, 0.95]$ represents emotional inertia.
- $\mathbf{E}_{\text{baseline}}$ is determined by agent personality traits (e.g., cheerful vs. brooding).
- $\text{Stimulus}_k(t)$ captures events like witnessing theft, receiving charity, or enduring famine.

---

## 3. Demographics & Lifecycle: Birth, Aging, Lineage, and Death

Agents must not be immortal statues. A true society requires generational renewal and the threat of permanent mortality.

### 3.1 The Mortality Cycle
- **Starvation & Dehydration:** If hunger or thirst reaches $0.0$ for more than $N_{\text{fatal}}$ consecutive ticks, the agent undergoes physical collapse.
- **Old Age:** Beyond age $T_{\text{elder}}$ (e.g. 1,000 steps), natural stamina decays:
  $$\text{Energy}_{\text{max}}(age) = \text{Base} \times \left(1 - \frac{age - T_{\text{elder}}}{T_{\text{max}}}\right)$$
- **Death & Mourning:**
  - Upon death, the agent's inventory is either inherited by kin or moved to the village commons.
  - A funeral procession forms at the Chapel/Graveyard.
  - Close friends ($T_{j \to i} > 0.6$) enter an acute **Grief State** for 48 ticks, lowering productivity and increasing altruistic bonding.

### 3.2 Courtship & Generation Birth
- When two unmarried, fertile agents maintain high bidirectional trust ($T_{i \leftrightarrow j} > 0.75$) and resource stability (shelter + grain reserves $> 20$ units), they form a household partnership.
- **Reproduction:** After a gestating interval, a new agent is instantiated.
- **Genetic & Cultural Inheritance:**
  $$\theta_{\text{risk}}^{(\text{child})} = \frac{\theta_{\text{risk}}^{(A)} + \theta_{\text{risk}}^{(B)}}{2} + \mathcal{N}(0, \sigma^2)$$
  The child inherits the family hut, a portion of the family trust score, and episodic traditions from parent memory stores.

---

## 4. Role Specialization & Deep Character Archetypes

Beyond basic farmers and guards, the village requires structured societal roles:

```
                      ┌──────────────────────┐
                      │  Village Magistrate  │  (Law, Justice, Assemblies)
                      └──────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Primary Producers│    │ Market Craftsmen │    │ Wardens & Healers│
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ · Master Farmer  │    │ · Grain Miller   │    │ · Town Guard     │
│ · Water-Carrier  │    │ · Blacksmith     │    │ · Herbalist      │
│ · Forest Forager │    │ · Silk Merchant  │    │ · Chapel Priest  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
                      ┌──────────────────────┐
                      │ Outcasts & Vagabonds │  (Smugglers, Bards, Thieves)
                      └──────────────────────┘
```

### Role Profiles & Duty Contracts
1. **The Village Magistrate / Elder:**
   - Convokes Village Meetings when granary stocks fall below emergency thresholds.
   - Presides over disputes, assigns pillory sentences, and blesses marriages.
2. **The Blacksmith & Toolmaker:**
   - Converts iron ore and timber into sickles, plows, and water buckets.
   - Lowers task energy costs for farmers who maintain positive trade relations.
3. **The Miller:**
   - Controls the watermill wheel. Takes raw sheaves and outputs fine flour at clearing tariff rates.
4. **The Herbalist / Healer:**
   - Forages alpine herbs to concoct healing broths, curing exhaustion and illness.
5. **The Town Guard:**
   - Patrols storehouses, intercepts thieves, inspects weights at the market, and rings the alarm bell during crisis.
6. **The Outcast / Rogue:**
   - Operates in shadows, trades on the black market, builds secret caches in the forest, and recruits desperate villagers into smuggling rings.
7. **The Wandering Bard:**
   - Travels between the Tavern, Pier, and Square singing ballads that summarize the week's gossip, praising heroes and shaming caught thieves.

---

## 5. Village Assemblies & Emergent Political Institutions

### 5.1 The Town Meeting (Gram Sabha)
At periodic intervals (e.g. every Sunday dawn at 08:00) or during emergency crises (drought or serial thefts), the Town Hall bell tolls:
1. **Assembly Gathering:** All free agents pause personal tasks and route to the Town Square amphitheater.
2. **Magistrate State-of-the-Village Address:** The Magistrate announces current granary wheat levels, water table reserves, and public crime statistics.
3. **Open Floor Grievances & Gossip:**
   - Victims of theft publicly accuse suspected culprits based on episodic memory traces.
   - Accused agents give defensive or remorseful speeches.
   - The community votes on guilt by tallying trust-weighted votes.
4. **Policy Enactment:**
   - Tithing mandates (e.g. mandatory 10% grain tithe if famine threatens).
   - Sanction ratification (e.g. 12-hour pillory or expulsion beyond the palisade).

---

## 6. Crime, Corruption, Underworld, and Redemption

### 6.1 The Spectrum of Moral Deviance
- **Desperation Theft:** Starving agent takes only what is needed to survive.
- **Greed Hoarding:** Wealthy merchant buys all market grain and artificially holds back supply to drive prices from 3.00¢ to 12.00¢.
- **Sabotage & Arson:** Malicious agent spoils granary bags or cuts water ropes to create chaos.
- **Bribery & Corruption:** Wealthy culprit offers coins to the patrolling guard to turn a blind eye.

### 6.2 The Justice & Redemption Loop
1. **Crime Commission:** Actor executes covert theft, leaving forensic probability traces.
2. **Investigation & Eyewitnesses:** Guards interrogate neighbors. Eyewitness credibility is factored by PageRank trust score.
3. **Trial & Sanction:**
   - Pillory shaming in the square.
   - Asset confiscation and fine restitution to victims.
4. **The Path to Redemption:**
   - An offender is not permanently condemned.
   - By performing community service (drawing water for the elderly, tithing double grain, clearing farm weeds), their PageRank status $S_i$ slowly recovers over consecutive cycles.

---

## 7. Downstream Multi-Agent Persona & Cinematic 3D Presentation

### 7.1 Visual Expression of Emotional States
In the 3D WebGL viewport, emotional state directly alters procedural animations:
- **Joy:** Bouncy walking stride, arms open, warm smiling aura ring, cheerful greetings.
- **Grief / Sadness:** Hunched shoulders (`bp.rotation.x = 0.35`), drooping head, slow trudging gait (`speed = 0.5x`), wiping tears.
- **Suspicion / Malevolence:** Creeping crouch posture, darting head scans left and right, crimson aura flicker.
- **Righteous Authority (Guards):** Upright posture, raised spear/shield, steady rhythmic stride.

### 7.2 Conversational Staging & Dynamic Two-Shots
- When two or more agents converse, they dynamically form equilateral conversational circles.
- Speech bubbles render multi-turn dialogues with expressive icons (`❤️ Friendship`, `⚔️ Threat`, `🌾 Trade Proposal`, `📜 Gossip`).
- Audio synthesis or procedural medieval lute/flute accents punctuate salient social milestones.

---

## 8. Implementation Roadmap & Sequential Phasing

| Phase | Milestone Name | Key Capabilities Delivered | Verification Oracle |
| :---: | :--- | :--- | :--- |
| **P1** | **Emotional Core (OCC+PAD)** | Internal emotional state vector $\mathbf{E}_i(t)$, pleasure/pain/dominance tracking, emotion-driven utility weights. | Unit tests verify emotion transitions match mathematical bounds. |
| **P2** | **Lifecycles & Lineage** | Agent aging, starvation mortality, courtship pairing, child birth with inherited utility weights. | Multi-generational simulation converges to demographic equilibrium without population collapse. |
| **P3** | **Town Assemblies (Sabha)** | Village bell summoning, council circle formation, public grievance trials, trust-weighted voting. | Deterministic test verifying thief trial and community vote execution. |
| **P4** | **Deep Economic Specialization** | Blacksmith, Miller, Herbalist, Healer, Magistrate, Outcast roles with inter-dependent production graphs. | Closed-loop economy test: raw wheat $\to$ milled flour $\to$ bread $\to$ sustained health. |
| **P5** | **Justice & Redemption Loop** | Guard investigations, forensic evidence gathering, bribery resistance, public restitution. | Criminal status recovery test showing PageRank rehabilitation over 100 ticks. |
| **P6** | **Cinematic Civilization Studio** | Multi-character emotional animations, procedural soundscapes, dynamic camera director tracking village drama. | WebGL 60 FPS verified across full 24-hour day/night generational cycle. |

---

> **Architectural Guarantee:** In accordance with [AGENTS.md](file:///c:/Users/Lenovo/Downloads/autogen/AGENTS.md), all emotional transitions, generational genetics, and council votes will execute deterministically in the Python simulation kernel under seeded RNG streams, while NVIDIA Nemotron and 3D WebGL remain downstream narrative and visual decoders.
