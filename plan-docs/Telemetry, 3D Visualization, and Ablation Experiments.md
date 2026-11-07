# Increment 5: Telemetry, 3D Web Visualization & Scientific Ablation Experiments

> **Module:** M22-H (Humanlike Villager Architecture)  
> **Source:** Derived from `plan-docs/AUTOGEN_Humanlike_Agent_Architecture.md` (§7, §8 Phase F, and Evaluation) and `plan-docs/humanlike_agents_plan.md`.

---

## 1. Overview & Objectives

Increment 5 is the **final increment** of Module M22-H. It elevates the humanlike villager cognitive architecture into an interactive, fully observable, and scientifically evaluated research platform:

1. **Rich Frontend Telemetry Pipeline:** Exposes complete cognitive, emotional, conversational, and memory states over the API and WebSocket streams without leaking private ground-truth states.
2. **Three.js / Web Society Visualizations (`apps/web/`):**
   - Villager head/orientation gaze cues pointing to salient environmental events.
   - Floating emotional expression badges derived from PAD dominant emotion.
   - Dynamic conversation link lines connecting speaking dyads with thickness modulated by engagement.
   - Interactive Villager Inspector panel showing Maslow needs deficit bars, behavior tree intention progress, multimodal memory retrieval breakdowns, and 4D directed relationships.
   - Epistemic Gossip Trace overlay visualizing rumor provenance chains ($v_0 \to v_1 \to \dots \to v_k$).
3. **Automated Scientific Ablation Suite (`packages/autogen/research/village_ablation.py`):**
   - Evaluates the system across multi-seed Monte Carlo simulations under 8 ablatable configurations:
     `FULL`, `NO_SALIENCE`, `NO_CONTAGION`, `NO_REFLECTION`, `NO_PERSONALITY`, `NO_NEEDS`, `NO_CONVERSATION`, `NO_GOSSIP`.
   - Calculates statistical effect sizes (Cohen's $d$, Welch's t-test) across 8 core behavioral metrics:
     reaction latency, intention flip-flop rate, gossip reach/hop count, mood variance, reciprocity clustering, forgiveness cycles, crime stability, and behavioral entropy.

---

## 2. Mathematical Formulations & Statistical Metrics

### 2.1 Behavioral Entropy (Personality Diversity)
Measures the behavioral differentiation across agent archetypes:
$$H_i = -\sum_{a \in \mathcal{A}} p_i(a) \log_2 p_i(a)$$
$$\bar{H} = \frac{1}{N} \sum_{i=1}^N H_i, \quad D_{\text{behavior}} = \frac{1}{\binom{N}{2}} \sum_{i < j} D_{\text{JS}}(P_i \parallel P_j)$$
where $D_{\text{JS}}$ is the Jensen-Shannon divergence between action probability distributions $P_i$ and $P_j$.
- Invariant: When `enable_personality=True`, $D_{\text{behavior}} > 0.15$ (archetypes differentiate). When ablated (`NO_PERSONALITY`), action distributions become statistically homogeneous ($D_{\text{behavior}} \to 0$).

### 2.2 Reaction Latency
Measures the discrete step delay between hazard emission ($t_{\text{hazard}}$) and agent intention preemption ($t_{\text{preempt}}$):
$$L_{\text{react}} = t_{\text{preempt}} - t_{\text{hazard}}$$
- Invariant: In `FULL`, high-salience storm events trigger preemption within $L_{\text{react}} \le 2$ ticks for $\ge 70\%$ of outdoor villagers. In `NO_SALIENCE`, $L_{\text{react}}$ increases significantly ($p < 0.01$).

### 2.3 Intention Hysteresis & Flip-Flop Rate
Quantifies intention stability versus jitter:
$$R_{\text{switch}} = \frac{\text{Total Intention Transitions}}{\text{Total Steps} \times N}$$
- Invariant: Intention switching cost $C_{\text{switch}}$ keeps $R_{\text{switch}} \le 0.20$ switches/tick across non-crisis steps.

### 2.4 Gossip Epidemic Reach & Provenance Distance
$$\text{Reach}(B) = \frac{|\{i \in \text{Villagers} : B \in \text{BeliefStore}_i\}|}{N}$$
$$\bar{h}(B) = \frac{1}{|\text{Believers}|} \sum_{i} \text{hop\_count}(B_i)$$

### 2.5 Statistical Effect Size (Cohen's $d$)
$$d = \frac{\bar{X}_{\text{FULL}} - \bar{X}_{\text{ABLATION}}}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

---

## 3. Files to Add & Modify

| Component | Path | Action | Description |
|---|---|---|---|
| **Research Engine** | `packages/autogen/research/village_ablation.py` | New | Automated multi-seed ablation runner, Cohen's $d$ calculation, metric aggregation |
| **Algorithm Docs** | `docs/algorithms/village/village_ablation.md` | New | 10-section documentation per AGENTS.md requirements |
| **Algorithm Registry**| `packages/autogen/intelligence/registry.py` | Modify | Register `village_ablation` with metadata and parameters |
| **Docs Linter** | `scripts/docs_lint.py` | Modify | Add allowed citations (`cohen`, `shannon`, `jensen`) |
| **Frontend Types** | `apps/web/src/types/village.ts` | Modify | Add TypeScript interfaces for gaze, intention, needs, conversation, beliefs |
| **Web Inspector UI**| `apps/web/src/components/slices/Slice6VillageSociety/VillagerInspector.tsx` | Modify | Add Needs bars, Intention card, Memory retrieval inspect, Rumor list |
| **Web Canvas UI** | `apps/web/src/components/slices/Slice6VillageSociety/VillageCanvas.tsx` | Modify | Add conversational linking lines and gaze orientation rendering |
| **Web Gossip Sync** | `apps/web/src/components/slices/Slice6VillageSociety/sync/VillageGossipOverlay.tsx` | New | Interactive provenance overlay tracing rumor transmission chains |
| **Unit Tests** | `tests/unit/village/test_humanlike_telemetry.py` | New | Verifies schema completeness, non-null guarantees, JSON serialization |
| **Ablation Tests** | `tests/unit/village/test_humanlike_ablation.py` | New | Verifies that ablations produce statistically significant behavioral deltas |
| **Demo CLI** | `packages/autogen/examples/humanlike_village_demo.py` | Modify | Support `--ablation <flag>` command line argument for single-command replication |

---

## 4. Verification Plan & Test Oracles

1. **Telemetry Schema Invariant (`test_humanlike_telemetry.py`):**
   - Every living villager in `get_full_state_payload()` possesses valid, typed `gaze`, `needs`, `intention`, `conversation`, `beliefs`, `memory`, and `relationships` blocks.
   - State serialization round-trip is strictly JSON serializable and deterministic.
2. **Ablation Sensitivity Oracles (`test_humanlike_ablation.py`):**
   - `NO_SALIENCE`: Hazard reaction latency is significantly greater than `FULL` ($d > 0.8$, large effect size).
   - `NO_CONTAGION`: Mean contagion delta is strictly $0.0$, and emotional coupling across neighbors is suppressed.
   - `NO_REFLECTION`: Nightly dispositions remain baseline ($g=0$, $v=0$) and relationship affinities do not consolidate.
   - `NO_GOSSIP`: Second-hand belief counts are strictly $0$ across all agents.
   - `NO_PERSONALITY`: Action entropy and utility weights collapse to role-uniform defaults.
3. **Full Regression & Linters:**
   - `python scripts/docs_lint.py` passes for all algorithms.
   - `pytest tests/unit/` passes 100%.
   - `ruff check` and `mypy` zero warnings.
