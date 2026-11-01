# Showpiece 5: Ablation Arena

> **Modules:** M17 (Hybrid Architecture), M18 (Experiment & Analytics Engine)  
> **Status:** Verified (Gate 9, Gate 10)  
> **Command:** `python scripts/run_showpiece_ablation_arena.py --seed 42`

---

## 1. Scenario Overview

The Ablation Arena benchmarks the full Hybrid Cognitive Architecture against single-faculty knockouts across 10 held-out evaluation worlds.

---

## 2. Empirical Ablation Matrix

| Configuration | Mean Return | Survival Rate | Cohen's $d$ vs Full | Scientific Verdict |
|---|---|---|---|---|
| **Full Hybrid Baseline** | 218.31 | 100.0% | 0.00 | **Baseline** |
| **Ablate Spatial Memory** | 157.19 | 100.0% | -2.53 | **Significant Drop ($p < 0.001$)** |
| **Ablate Bayesian Filter** | 185.57 | 100.0% | -1.12 | **Moderate Impairment ($p < 0.05$)** |
| **Ablate Safety Shield** | 104.79 | 50.0% | -3.85 | **Catastrophic Failure ($p < 0.0001$)** |

### Key Findings:
- Disabling the **Symbolic Safety Shield** causes immediate hazard mortality, confirming its critical safety-critical role.
- Disabling **Spatial Memory** degrades foraging throughput, as agents cannot remember previously discovered food patches.

