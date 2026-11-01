# Showpiece 7: The Changing World

> **Modules:** M13 (RL), M14 (Neural Nets), M16 (Evolution), M17 (Hybrid)  
> **Status:** Verified (Gate 10)  
> **Command:** `python scripts/run_showpiece_the_changing_world.py --seed 42`

---

## 1. Scenario Overview

Investigates cognitive plasticity and adaptation latency under periodic non-stationary environmental regime shifts.

### The Four Seasons:
1. **Abundant Spring (Ticks 0–30):** High food ($1.5\times$), low hazards ($0.2\times$).
2. **Scorching Drought (Ticks 30–60):** Severe food scarcity ($0.3\times$), high hazards ($1.8\times$).
3. **Flooded Monsoon (Ticks 60–90):** Moderate food ($0.7\times$), high obstacles.
4. **Harvest Autumn (Ticks 90–120):** Favorable foraging recovery ($1.3\times$).

---

## 2. Empirical Performance Across Regime Transitions

| Season | Adaptive Hybrid Return | Rigid Reflex Baseline Return |
|---|---|---|
| **Abundant Spring** | 163.3 | 114.0 |
| **Scorching Drought** | 16.3 (Resilient) | 0.0 (Total Collapse) |
| **Flooded Monsoon** | 64.8 | 18.3 |
| **Harvest Autumn** | 137.6 (+742.5% recovery) | 88.3 |

### Key Findings:
- Under environmental stress (Drought), rigid reflex agents experience total performance collapse ($100\%$ drop to $0.0$).
- Adaptive hybrid agents maintain positive return via Bayesian belief adjustments and safety shield protection, recovering immediately upon the arrival of Autumn.

