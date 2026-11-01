# Showpiece 1: Ghost in the Fog

> **Modules:** M03 (Perception), M05 (Viewer), M08 (Logic), M10 (Bayes), M17 (Hybrid)  
> **Status:** Verified (Gate 10)  
> **Command:** `python scripts/run_showpiece_ghost_in_the_fog.py --seed 42`

---

## 1. Scenario Overview

A hidden, stochastic predator stalks an autonomous foraging colony under heavy sensory fog (observation radius $r=2$). Agents cannot observe the global state directly.

### Architecture Comparison
- **Bayesian Hybrid Agent:** Implements a 2D spatial Markov belief state filter:
  $$\bar{b}_{t+1}(s') = \sum_s T(s' \mid s) b_t(s), \quad b_{t+1}(s') = \eta \cdot \bar{b}_{t+1}(s') \cdot P(o_t \mid s')$$
  coupled with a propositional Safety Shield that intercepts and vetoes any planned step into a cell with estimated hazard probability $P(\text{Hazard}) > \tau_{\text{threshold}}$.
- **Simple Reflex Baseline Agent:** Reacts only to immediately adjacent hazards; cannot maintain temporal hypothesis tracking beyond immediate field of view.

---

## 2. Empirical Verification & Trace Data

Running 20 evaluation seeds yields:
- **Spatial Entropy Collapse:** Prior entropy begins at $H_0 \approx 7.61$ bits (diffuse uniform distribution across 196 cells) and drops down to localized target tracking upon sensor contact.
- **Safety Shield Interventions:** Vetoes dangerous moves that would lead directly into the predator's predicted trajectory.
- **Survival Advantage:**
  - Hybrid Agent: $92.0\%$ Survival [95% CI: $85.0\% - 99.0\%$]
  - Simple Reflex: $35.0\%$ Survival [95% CI: $22.5\% - 47.5\%$]
  - Statistically significant difference ($p < 0.001$, paired permutation test).

