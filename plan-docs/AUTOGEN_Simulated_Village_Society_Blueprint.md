# AUTOGEN AI — Simulated Village Society & Emergent Economy Blueprint
## A Mathematically Grounded, Socially Interactive Artificial Society with Risk-Reward Ethics

> **Project:** AUTOGEN AI Research Platform  
> **Status:** Architecture & Implementation Plan  
> **Location:** `plan-docs/AUTOGEN_Simulated_Village_Society_Blueprint.md`  
> **Target Subsystem:** Multi-Agent Emergence, Social Utility Economics, & Narrative Persona Bridge  

---

## 1. Executive Summary & Vision

### 1.1 The Concept
This blueprint defines the architecture for **AUTOGEN Village**—a persistent, multi-agent simulated society where autonomous agents live, work, socialize, and compete within a physical village environment. 

Unlike Stanford's *Smallville* (Generative Agents), where behavior is generated purely through stochastic LLM prompts without ground-truth physics or economic constraints, **AUTOGEN Village is grounded in mathematical utility theory, finite resource physics, and game-theoretic incentives**:
1. **Physical & Economic Ground Truth:** Agents experience real physiological needs (hunger, hydration, energy, shelter) and operate in a finite-resource economy (food, water, lumber, tools, currency).
2. **Quantified Societal Status:** Social standing is not a cosmetic label. It is a mathematically derived **Social Capital Score** $\mathcal{S}_i(t)$ calculated dynamically using directed trust graphs, reciprocal exchange histories, and community consensus (PageRank/Eigenvector Centrality).
3. **The Legal vs. Unethical Spectrum:** Agents do not possess hardcoded morality. Instead, they optimize an internal utility function that balances immediate physiological needs, wealth accumulation, societal prestige, and perceived risk of detection/sanction. Depending on individual risk tolerances $\theta_{\text{risk}}$ and environmental pressure (drought, famine), agents rationally choose between:
   - **Legal Pathways:** Cooperative farming, fair market trading, donating surplus, maintaining common infrastructure.
   - **Unethical / Criminal Pathways:** Hoarding scarce grain during shortages, price gouging, theft from private huts, cheating on contracts, and free-riding on the village commons.
4. **Decoupled Persona & Dialogue Bridge:** The agent's core decision-making (what action to take) is calculated **from scratch** by the AUTOGEN intelligence engine. An optional **LLM Persona Decorator** (configured securely via local `.env`) translates the agent's mathematical `DecisionTrace` and memory into rich natural language dialogues, village gossip, town hall debates, and personal diaries. When no LLM API key is present, the simulation falls back seamlessly to deterministic symbolic dialogues.

---

## 2. Core Mathematical Formulations

### 2.1 Multi-Objective Agent Utility Function
Each agent $i \in \{1, \dots, N\}$ at time $t$ evaluates actions $a \in \mathcal{A}_i$ by optimizing an expected utility function $\mathbb{E}[U_i(a, s_t)]$:

$$U_i(t) = w_{\text{phys}} \cdot U_{\text{phys}}(H_i, T_i, E_i) + w_{\text{econ}} \cdot U_{\text{econ}}(W_i) + w_{\text{soc}} \cdot \mathcal{S}_i(t) - \mathcal{R}_i(a)$$

Where:
* **$U_{\text{phys}}(H_i, T_i, E_i)$ (Physiological Survival):** Concave utility reflecting diminishing marginal returns and catastrophic penalty for critical depletion:
  $$U_{\text{phys}} = \ln(H_i + \epsilon) + \ln(T_i + \epsilon) + \ln(E_i + \epsilon) - \kappa \cdot \mathbb{I}_{[\min(H_i, T_i, E_i) < \tau]}$$
  where $H_i \in [0, 100]$ is hunger saturation, $T_i \in [0, 100]$ is hydration, $E_i \in [0, 100]$ is energy, and $\kappa$ is a starvation crisis penalty.
* **$U_{\text{econ}}(W_i)$ (Wealth Accumulation):** Total liquid coin and stored commodities valued at local market clearing prices:
  $$U_{\text{econ}} = \sqrt{C_i + \sum_{k} p_k \cdot I_{i, k}}$$
  where $C_i$ is coin balance, $I_{i, k}$ is quantity of commodity $k$, and $p_k$ is current clearing price.
* **$\mathcal{S}_i(t)$ (Societal Status / Social Capital):** Derived from the Village Trust Graph (see §2.2).
* **$\mathcal{R}_i(a)$ (Risk and Expected Penalty):** Evaluated specifically for unethical actions (see §2.3).

---

### 2.2 Village Trust Graph & Societal Status Engine
The village maintains an evolving directed trust graph $\mathbf{T}(t) \in [0, 1]^{N \times N}$, where $T_{j \to i}(t)$ represents agent $j$'s subjective trust rating of agent $i$.

#### 2.2.1 Trust Update Rule (Reciprocity & Infraction)
Following an interaction (trade, shared task, observed behavior) between agent $j$ and agent $i$:
$$T_{j \to i}(t+1) = (1 - \lambda_{\text{decay}}) T_{j \to i}(t) + \Delta T_{j \to i}$$

Where:
$$\Delta T_{j \to i} = \begin{cases}
+\alpha_{\text{coop}} \cdot (1 - T_{j \to i}(t)) & \text{if } i \text{ honored agreement / donated to commons} \\
-\beta_{\text{defect}} \cdot T_{j \to i}(t) & \text{if } i \text{ defaulted, lied, or hoarded} \\
-\gamma_{\text{crime}} & \text{if } i \text{ was caught stealing or sabotaging}
\end{cases}$$
Here $\beta_{\text{defect}} > \alpha_{\text{coop}}$, formalizing the sociological invariant that *trust is lost much faster than it is built*.

#### 2.2.2 Global Societal Status $\mathcal{S}_i$ (Trust PageRank)
An agent's global societal status $\mathcal{S}_i \in [0, 1]$ is the stationary distribution of the trust-weighted random walk with damping factor $d = 0.85$:
$$\mathbf{s} = \frac{1 - d}{N} \mathbf{1} + d \, \mathbf{\tilde{T}}^T \mathbf{s}$$
where $\mathbf{\tilde{T}}$ is row-stochastic normalized trust matrix. 
* High-status agents are trusted by agents who are themselves widely trusted.
* High status yields tangible perks: lower trade markups, credit eligibility, and political weight during town hall votes.

---

### 2.3 Ethics Engine: The Legal vs. Unethical Decision Model
At each step, agent $i$ can select from legal actions $\mathcal{A}_{\text{legal}}$ (work field, buy from market, rest) or unethical actions $\mathcal{A}_{\text{unethical}}$ (theft, price-gouging, falsifying resource discovery, commons depletion).

For an unethical action $a_{\text{unethical}}$:
$$\mathbb{E}[U(a_{\text{unethical}})] = U_{\text{gain}}(a) - P_{\text{detect}}(a, \mathbf{x}_i, \text{env}) \cdot \left[ C_{\text{penalty}} + \Delta \mathcal{S}_{\text{loss}} \cdot w_{\text{soc}} \right]$$

#### 2.3.1 Probability of Detection $P_{\text{detect}}$
$$P_{\text{detect}} = 1 - \prod_{j \in \text{Witnesses}} (1 - P_{j, \text{sight}})$$
Where $P_{j, \text{sight}}$ depends on:
* Distance $\|\mathbf{x}_i - \mathbf{x}_j\|$
* Ambient lighting (Day vs. Night cycle)
* Obstacles / Line of Sight occlusion (walls, trees)
* Presence of designated village guards or watchtowers

#### 2.3.2 Emergence of Rational Corruption
If hunger $H_i \to 0$ (imminent starvation), $U_{\text{gain}}(\text{steal bread}) \to \infty$ due to the logarithmic survival curve. Even a moral, low-risk-tolerance agent will rationally turn to theft when survival is at stake. Conversely, wealthy agents with low social weight $w_{\text{soc}}$ may engage in white-collar market cornering.

---

## 3. Village Spatial Environment & Economy

### 3.1 Spatial Layout (The Grid)
The village is instantiated on a $40 \times 40$ or $60 \times 60$ discrete coordinate grid partitioned into functional municipal zones:

```
┌─────────────────────────────────────────────────────────────┐
│                       WILD FOREST                           │
│           (Lumber, Foraged Berries, Wildlife)               │
├──────────────┬───────────────────────────────┬──────────────┤
│ PRIVATE HUTS │       CENTRAL MARKET          │ PRIVATE HUTS │
│ (Agent Beds, │ (Auction House, Trading Post, │ (Agent Beds, │
│  Personal    │  Town Notice Board)           │  Personal    │
│  Chests)     │                               │  Chests)     │
├──────────────┼───────────────────────────────┼──────────────┤
│   TOWN WELL  │       VILLAGE COMMONS         │ AGRICULTURAL │
│ (Clean Water,│ (Town Hall, Granary, Pillory, │    FIELDS    │
│  Regrowth)   │  Community Storage)           │(Wheat crops) │
└──────────────┴───────────────────────────────┴──────────────┘
```

### 3.2 Commodity Flow & Market Mechanism
1. **Raw Commodities:** Water, Wheat, Timber, Fish, Tools.
2. **Double Auction & Vickrey Exchange:** 
   - Agents place bids and asks at the Central Market.
   - Prices fluctuate dynamically based on local supply and demand (scarcity pricing).
3. **The Granary Commons:**
   - Villagers can voluntarily tithe wheat to the town granary to boost their societal status $\mathcal{S}_i$.
   - During winter or drought, the granary distributes food to prevent famine—or becomes the primary target for desperate thieves.

---

## 4. Software Architecture & Module Boundaries

Following AUTOGEN's core principle, layers are strictly isolated. Algorithms and simulation state never import rendering or dialogue code.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           APPS / CLIENT LAYER                           │
│  apps/web (CanvasRenderer, VillageInspector, SpeechBubbles, Leaderboard)│
└────────────────────────────────────▲────────────────────────────────────┘
                                     │ JSON State & SSE Events
┌────────────────────────────────────┴────────────────────────────────────┐
│                        API & REPLAY CONTROLLER                          │
│  autogen.apps.api.routes (Village Simulation Session & Stream)          │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│                       NARRATIVE & DIALOGUE LAYER                        │
│  autogen.intelligence.village.persona (Dual-Mode: Local Template / LLM) │
│  - Receives: DecisionTrace, Memory Context, Social Graph                │
│  - Emits: SpeechEvent, GossipEvent, PublicDeclaration, DiaryEntry       │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │ Reads Decisions (Does not alter)
┌────────────────────────────────────┴────────────────────────────────────┐
│                       VILLAGE INTELLIGENCE LAYER                        │
│  autogen.agents.village (VillagerAgent)                                  │
│  - Utility Engine (§2.1)                                                │
│  - Trust & Status Graph (§2.2)                                          │
│  - Risk / Crime Engine (§2.3)                                           │
│  - Episodic Memory & Spatial Cognitive Map                              │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │ Observations & Action Commands
┌────────────────────────────────────┴────────────────────────────────────┐
│                       SIMULATION ENGINE & PHYSICS                       │
│  autogen.simulation.environment.village (VillageWorld, Commons, Stores) │
│  - Enforces physical laws, movement, harvesting, inventory, detection   │
│  - Validates all actions. Agents NEVER directly mutate world state.     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Dual-Mode Dialogue & Persona Engine

To satisfy the **mathematical independence** rule in `AGENTS.md` while providing a rich Smallville-like conversational experience:

### 5.1 Mode 1: Deterministic Symbolic Dialogue (Scratch Baseline)
* Runs 100% offline with zero dependencies or API costs.
* Uses structured grammar trees driven by the agent's actual internal state:
  * *Friendly Trade:* `"I have surplus [Item]. Would you trade for [Item] at fair value?"`
  * *Deceptive / Hoarding:* `"My shelves are bare, friend. I cannot spare any grain."` (when private inventory has 20 units).
  * *Accusation / Gossip:* `"I saw [AgentB] lurking near the Granary last night. Beware!"`

### 5.2 Mode 2: LLM Persona Decorator (Smallville-Grade)
* Activated only if an API key (e.g. `GEMINI_API_KEY` or `OPENAI_API_KEY`) is found in `.env`.
* **Zero Simulation Impact:** The LLM prompt is strictly downstream. It receives:
  1. The agent's Persona (name, role, trait, risk tolerance).
  2. The mathematical `DecisionTrace` (Action chosen, Utility breakdown).
  3. The current recipient and relationship score $T_{j \to i}$.
  4. The 3 most relevant episodic memories.
* The LLM generates the natural language utterance. If the API fails, times out, or is absent, the system falls back instantly to Mode 1 without breaking the simulation loop.

---

## 6. Implementation Roadmap

### Phase 1: Environment & Commodity Mechanics (Week 1)
* [x] Create `packages/autogen/simulation/environment/village/`:
  * `village_grid.py`: Map with zones (Residential, Market, Commons, Forest, Well).
  * `amenities.py`: Private chests, Granary, Well, Market stalls.
  * `commodities.py`: Physical items, decay rates, and inventory mechanics.
* [x] Unit tests for physical movement, harvesting, and container security.

### Phase 2: Social Capital & Trust Graph (Week 2)
* [x] Create `packages/autogen/intelligence/village/social_graph.py`:
  * Asymmetric trust matrix $T_{j \to i}$ with reciprocity updates and decay.
  * Power-iteration PageRank calculation for global Societal Status $\mathcal{S}_i$.
* [x] Implement gossip and reputation propagation: agents share their trust ratings with close peers.
* [x] Mathematical verification tests for status conservation and convergence.

### Phase 3: The Rational Ethics & Decision Engine (Week 3)
* [x] Create `packages/autogen/agents/village/villager_agent.py`:
  * Multi-objective utility calculator ($U_{\text{phys}}, U_{\text{econ}}, \mathcal{S}_i, \mathcal{R}_i$).
  * Line-of-sight detection probability engine for illicit acts.
  * Action generators for both legal work and opportunistic theft/hoarding.
* [x] Verification tests for hunger-induced crime thresholds and punishment deterrence.

### Phase 4: Persona Bridge & Dialogue (Week 4)
* [x] Create `packages/autogen/intelligence/village/persona/`:
  * `symbolic_dialogue.py`: Pure rule-based communication engine.
  * `llm_bridge.py`: Async prompt wrapper reading `.env` keys with graceful fallback.
  * `village_bus.py`: Spatial speech broadcasting with distance attenuation and eavesdropping.

### Phase 5: Web UI & Visual Inspection (Week 5)
* [x] Extend `apps/web`:
  * `VillageCanvas.tsx`: Render huts, crops, town square, and animated villager avatars.
  * `SocialGraphView.tsx`: Interactive network graph visualizing trust lines and status hierarchy.
  * `SpeechBubbleOverlay.tsx`: Real-time dialogue bubbles appearing above conversing agents.
  * `VillagerInspector.tsx`: Deep-dive panel displaying an agent's utility weights, secret inventory, and real decision traces.

### Phase 6: Empirical Research & Scenarios (Week 6)
* [x] Reproducible experiments:
  1. *The Law Enforcement Threshold:* What ratio of guards/penalties is mathematically required to eliminate granary theft?
  2. *The Inequality vs. Crime Dynamic:* Does high wealth disparity trigger emergent criminal cartels?
  3. *Reputation Resilience:* Can a known fraudster rebuild status through public philanthropy?

---

## 7. Verification Strategy & Invariants

To comply with AUTOGEN's Definition of Done:

| Invariant | Mathematical Check | Test Oracle |
| :--- | :--- | :--- |
| **Trust Boundedness** | $\forall i, j: T_{j \to i}(t) \in [0, 1]$ | Monotonic bounds check after 1,000 arbitrary events. |
| **Status Distribution** | $\sum_{i=1}^N \mathcal{S}_i = 1$ | Power iteration convergence check within $\epsilon = 10^{-6}$. |
| **Detection Monotonicity** | $\frac{\partial P_{\text{detect}}}{\partial \|\mathbf{x}_i - \mathbf{x}_{\text{witness}}\|} \le 0$ | Distance sensitivity unit tests across line-of-sight. |
| **Simulation Determinism** | Same seed produces identical world state & decisions | Replay equality check across two identical runs. |
| **LLM Isolation** | Mocked or failed LLM does not alter simulation coordinates or balances | Null-bridge assertion tests. |

---

## 8. Summary & Next Step

This plan unites the narrative charm of Stanford's Smallville with the scientific rigor, mathematical purity, and deterministic foundation of AUTOGEN AI.

**To begin implementation:**
We start with **Phase 1: Environment & Commodity Mechanics** by formalizing the Village Grid specification and typed data contracts.

