# Showpiece 6: Time Machine

> **Modules:** M02 (Contracts & Core Engine), M05 (Viewer & Replay), M20 (Studio)  
> **Status:** Verified (Gate 1, Gate 10)  
> **Command:** `python scripts/run_showpiece_time_machine.py --seed 42`

---

## 1. Scenario Overview

Demonstrates deterministic history rewind, counterfactual intervention, and timeline divergence.

### The Protocol:
1. **Run Original History (Timeline Alpha):** Simulation runs forward to tick $T=25$.
2. **Snapshot Capture:** An immutable snapshot of the exact world state, RNG bit-generator seeds, and agent state is saved at $T=15$.
3. **Counterfactual Intervention:** The engine is rewound to $T=15$. A single action override (`ActionType.LEFT`) is injected for the target agent.
4. **Fork Timeline Beta:** Both timelines are stepped forward from $T=15$ to $T=25$.

---

## 2. Quantitative Divergence Evidence

$$\Delta(t) = \|\mathbf{p}_{\text{Alpha}}(t) - \mathbf{p}_{\text{Beta}}(t)\|_2$$

- Pre-fork ($t < 15$): $\Delta(t) = 0.00$ (Bit-for-bit identical history)
- Fork Step ($t = 15$): Alternate decision executed
- Post-fork ($t \ge 16$): $\Delta(t) = 1.00$ units (Permanent, deterministic trajectory divergence)

Both realities remain 100% reproducible bit-for-bit when re-run with the same seed.

