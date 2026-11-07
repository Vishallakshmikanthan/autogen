# Increment 4: Conversation State Machine, Dialogue Acts Utility, and Lossy Provenance Gossip Dynamics

> **Module:** M22-H (Humanlike Villager Architecture)  
> **Source:** Derived from `plan-docs/AUTOGEN_Humanlike_Agent_Architecture.md` (§3.1 - §3.5) and `plan-docs/humanlike_agents_plan.md`.

---

## 1. Overview & Scope

Increment 4 equips villagers with rich, multi-tick conversational abilities, utility-grounded speech act arbitration, and an epidemic lossy gossip transmission system with provenance tracking.

Key capabilities delivered:
1. **Multi-tick Conversation State Machine:** Approach $\to$ Greeting $\to$ Talking $\to$ Closing / Abandonment / Interruption with dynamic engagement decay and snub relationship penalties.
2. **Utility-Guided Dialogue Act Selection:** Communicative acts (`GREET`, `SMALL_TALK`, `SHARE_NEWS`, `GOSSIP`, `ASK_FAVOR`, `COMFORT`, `COMPLAIN`, `ACCUSE`, `APOLOGIZE`, `FAREWELL`) evaluated via multi-attribute utility with risk penalties derived from overhearing probabilities ($P_{overheard} \cdot \kappa$).
3. **Lossy Gossip Transmission & Provenance Tracking:** Structured `Belief` items with source chain tracking ($A \to B \to C$), speaker emotional spin, listener credibility & skepticism evaluation, echo loop suppression, and independent corroboration via noisy-OR.
4. **Post-Conversation Settlement:** Updates directed dyadic relationship dimensions (`affinity`, `respect`, `familiarity`, `trust`), creates structured conversation episodic memories, and sets social obligations.

---

## 2. Mathematical Formulations

### 2.1 Engagement Dynamics
$$E_i(t+1) = \mathrm{clip}\left(E_i(t) + 0.15 \cdot r_{topic} + 0.10 \cdot \text{affinity}_{i \to j} - 0.08 - 0.10 \cdot b_i,\, 0.0,\, 1.0\right)$$
- When $E_i < \theta_{leave} = 0.20$, participant walks away mid-conversation (`ABANDONED`).
- Snub penalty: $\Delta \text{affinity}_{j \to i} = -0.10$, $\Delta \text{trust}_{j \to i} = -0.05$.
- Emergency bypass: if departure is triggered by an alert with $\text{threat} > 0.40$, snub penalty is zeroed.

### 2.2 Dialogue Act Utility
$$U_{act}(d) = w_{soc} \Delta \text{status}(d) + w_{rel} \Delta \text{affinity}_{i \to j}(d) + w_{info} \text{info\_value}(d) + w_{mood} \text{mood\_congruence}(d) - \text{cost}(d) - \mathbb{1}[\text{risky}] \cdot \theta_{risk} \cdot P_{overheard} \cdot \kappa_i$$
where:
- $\kappa_i = 20(0.5 + 0.5 C_i)(1 - 0.5 O_i)$ is Conscientiousness/Openness risk sensitivity.
- $P_{overheard} = \max_{k \notin \{i, j\}} \text{audibility}(i, k)$.

### 2.3 Gossip Transmission & Distortion
1. **Speaker Spin:**
   $$\text{spin} = 1.0 + 0.25 \cdot a_s + 0.20 \cdot \max(0, -\text{affinity}_{s \to B.subject})$$
   $$\text{sent\_valence} = \mathrm{clip}(B.valence \cdot \text{spin},\, -1.0,\, 1.0)$$
   $$\text{sent\_confidence} = \mathrm{clip}(B.confidence \cdot (1.0 \text{ if } B.first\_hand \text{ else } 0.90),\, 0.0,\, 1.0)$$

2. **Listener Credibility & Skepticism:**
   $$\text{credibility} = 0.50 \cdot \text{trust}_{l \to s} + 0.30 \cdot \text{status\_norm}(s) + 0.20 \cdot \text{affinity}_{l \to s}$$
   $$\text{consistency} = 1.0 \text{ if prior is None else } \left(1.0 - \frac{|\text{prior.valence} - \text{sent\_valence}|}{2.0}\right)$$
   $$\text{skepticism} = 0.50 + 0.50 \cdot (1.0 - O_l)$$
   $$x = 4.0 \cdot \left(\frac{\text{credibility} \cdot \text{consistency}}{\text{skepticism}} - 0.50\right)$$
   $$P_{accept} = \frac{1}{1 + e^{-x}}$$

3. **Noisy-OR Corroboration:**
   $$c_{merged} = 1.0 - (1.0 - c_{prior})(1.0 - \text{sent\_confidence} \cdot \text{credibility})$$

---

## 3. Architecture & Integration Plan

- **`intelligence/village/conversation.py`**: State machine transitions, act evaluations, and post-conversation relationship updates.
- **`intelligence/village/gossip.py`**: `Belief` data structures, transmission, spin, credibility, and provenance checks.
- **`intelligence/village/relationships.py`**: 4D directed relationships (`affinity`, `respect`, `familiarity`, `trust`).
- **`agents/village/villager_agent.py`**: Integration with candidate pathways, active intentions, and memory stream.
- **`agents/village/controller.py`**: Phase 4 conversation turn resolution and Phase 6 symbolic dialogue act rendering.
- **`tests/unit/village/`**: Independent unit & oracle test coverage.

