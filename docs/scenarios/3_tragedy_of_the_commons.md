# Showpiece 3: Tragedy of the Commons

> **Modules:** M13 (RL), M15 (Multi-Agent Systems)  
> **Status:** Verified (Gate 8, Gate 10)  
> **Command:** `python scripts/run_showpiece_tragedy_of_the_commons.py --seed 42`

---

## 1. Scenario Overview

$N=4$ autonomous learning agents harvest food from a shared, regrowing resource pool modeled with continuous logistic dynamics:
$$R_{t+1} = R_t + r R_t \left(1 - \frac{R_t}{K}\right) - \sum_{i=1}^N h_{i,t}$$

When biomass drops below critical threshold $R < R_{\text{crit}}$, regrowth halts, triggering irreversible ecosystem collapse.

---

## 2. Empirical Regime Comparison

| Metric | Selfish Policy ($\beta = 0.0$) | Prosocial Reward ($\beta = 0.5$) |
|---|---|---|
| **Depletion Horizon** | 3 ticks (Immediate collapse) | 50+ ticks (Sustainable equilibrium) |
| **Total Harvested Biomass** | 45.8 units | 137.3 units (+199.8%) |
| **Gini Inequality Index** | 0.038 | 0.010 (Reduced disparity) |

### Key Findings:
- Under individual rewards, myopic harvesting drives stock to extinction immediately.
- Shared/prosocial reward structures align incentives with community carrying capacity, multiplying total biomass yield threefold while reducing payoff inequality.

