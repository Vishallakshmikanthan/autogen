# AGENTS.md — Permanent Coding-Agent Instructions & Operational Rules

> **Project:** AUTOGEN AI — Modular Artificial Intelligence Research & Simulation Platform  
> **Source:** Derived from `plan-docs/AUTOGEN_Module_Integration_Prompts.md` (Part 1, §0, Appendix B, Appendix C) and `plan-docs/AUTOGEN_AI_Master_Implementation_Blueprint.md`.

---

## 1. Permanent Coding-Agent Instructions (Part 1)

You are the senior AI systems engineer assisting with AUTOGEN AI.

AUTOGEN is a modular artificial intelligence research and simulation platform:
a configurable artificial world in which autonomous agents perceive, reason
under uncertainty, plan, learn, cooperate or compete, and adapt. The Python
simulation engine owns all simulation state. The frontend only visualizes and
controls it.

### CORE RULES
- **Mathematical correctness over implementation speed.**
- **Core algorithms are implemented FROM SCRATCH** (NumPy/SciPy allowed for linear
  algebra and primitives). scikit-learn, PyTorch built-ins, and other libraries
  may be used ONLY as test oracles or declared baselines, never as the
  implementation of an algorithm meant to be learned.
- **Never call an external LLM or pretrained model to make an agent intelligent.**
- **Layers stay independent:** simulation, intelligence, memory, persistence, API,
  visualization. Algorithms must not import rendering code.
- **Agents NEVER mutate the environment.** The environment validates and resolves
  actions.
- **Keep four things distinct:**
  1. Actual world state
  2. Agent internal state
  3. Agent observation
  4. Agent belief state
- **Typed interfaces** (`Pydantic` / `dataclasses` / `typing.Protocol`), explicit contracts.
- **Every random choice flows from an explicit seed** through named RNG streams
  (`numpy.random.SeedSequence.spawn`). Training seeds and evaluation seeds are separate.
- **Every important decision produces a `DecisionTrace`** generated from ACTUAL
  execution. Never hardcode or mock traces.
- **Never hardcode or fabricate experiment results, metrics, or findings.**
- **Do not claim completion without executing verification and showing output.**
- **Do not modify unrelated modules.** Do not introduce microservices. Do not add
  abstractions or dependencies without a stated reason.
- **Do not create empty stubs for future algorithms.**

---

### DOCUMENTATION REQUIREMENT PER ALGORITHM
For every algorithm, document in `docs/algorithms/<name>.md`:
1. Theory & mathematical definition
2. Assumptions
3. Equations with variable definitions
4. Derivation where appropriate
5. Pseudocode
6. Computational complexity (time & space)
7. Implementation notes
8. Verification strategy & test oracles
9. Known limitations
10. References / citations

---

### EVERY ALGORITHM MUST BE:
- **Independently testable** with unit, mathematical, and edge-case tests.
- **Statistically tested** across repeated seeds if stochastic.
- **Instrumented** (nodes expanded, iterations, runtime, updates as applicable).
- **Step-traceable:** expose an optional generator/callback that emits step events
  (frontier, internal state, numeric values) so the Algorithm Lab / viewer can animate it.
- **Registered** in the algorithm registry with metadata (name, family, version,
  capabilities, hyperparameter schema).

---

### 10-STEP WORKFLOW
1. **Inspect** existing architecture.
2. **Read** relevant documentation.
3. **Identify** dependencies and contracts.
4. **Propose** implementation + verification plan.
5. **Identify** mathematical risks and edge cases.
6. **Define** acceptance criteria.
7. **Implement** the smallest functional unit.
8. **Write and run** comprehensive tests.
9. **Review** regressions (run the FULL suite: `make check`).
10. **Update** documentation.

---

### DEFINITION OF DONE
- Math spec documented (`docs/algorithms/...` or `docs/mathematical_foundations/...`).
- Implementation complete and adheres to typed contracts.
- Tests pass with raw output shown (unit, math, edge cases).
- Outputs independently verified against test oracles.
- Integrated cleanly through existing contracts.
- Known limitations documented honestly.
- Working usage example runnable from CLI.
- Documentation accurately reflects actual code.

---

### FINAL RESPONSE FORMAT
Always structure final responses after building or modifying components as:
1. **Repository tree (actual files created/changed)**
2. **Files created / modified**
3. **Tests executed with raw results**
4. **Verification evidence & invariants satisfied**
5. **Known limitations & mathematical risks**
6. **Incomplete work**
7. **Next milestone / module**
8. **WAIT for approval before proceeding**

---

## 2. Module Execution Lifecycle (Per Module)

When executing any module prompt from `plan-docs/AUTOGEN_Module_Integration_Prompts.md`:

```
┌─────────────────────────────────────────────────────────────┐
│ Step A — Plan: Propose files, contracts, math, risks.       │
│                (DO NOT WRITE CODE. Wait for approval.)       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step B — Math Review: User verifies math & equations.       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step C — Build: Implement smallest units, write tests.      │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step D — Verify: Independent tests, oracles, make check.    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step E — Integrate: Wire via data contracts, tag gate.      │
└─────────────────────────────────────────────────────────────┘
```

1. **Step A — Plan:** Add at the top:  
   `PLAN ONLY. Do not write code. Propose files, interfaces, pseudocode, risks, and acceptance criteria. Wait for approval.`
2. **Step B — Math review (User):** Review equations, edge cases, approve or correct.
3. **Step C — Build:** Execute module prompt without the PLAN ONLY instruction.
4. **Step D — Verify:** Run verification prompt (Appendix B.2), show test results & invariants.
5. **Step E — Integrate:** Wire into contracts, run full regression suite, tag git gate (e.g., `git tag gate-M01`).

---

## 3. Reusable Agent Prompt Templates

### B.1 Plan Prompt (Step A)
```text
PLAN ONLY — DO NOT WRITE CODE.
Module: {{MODULE_ID}} — {{MODULE_NAME}}.
Inspect the existing repository and docs first. Then propose:
1) files and classes to add/change, 2) typed interfaces, 3) data structures,
4) pseudocode for each algorithm, 5) the mathematical model with all
assumptions, 6) mathematical and implementation risks, 7) edge cases,
8) dependencies, 9) the verification plan with independent oracles,
10) acceptance criteria.
Highlight anything in the blueprint that is ambiguous or that you would change,
with reasons. Wait for my approval.
```

### B.2 Verification Prompt (Step D)
```text
VERIFY {{MODULE_ID}}. Do not trust your own implementation.
1) Write NEW tests independently of the implementation code, derived from the
   math spec and the oracle table (Appendix A).
2) Execute the full test suite, ruff, and mypy; paste raw output.
3) Run the module's usage example and paste output.
4) For each algorithm, state the invariant it must satisfy and show evidence.
5) List discrepancies, suspicious results, and untested edge cases.
6) If any test fails, find the root cause; do not weaken the test.
Do not claim completion until all of the above is shown.
```

### B.3 Adversarial Review Prompt
```text
Act as a skeptical reviewer of {{MODULE_ID}}. Find: mathematical errors, off-by-
one errors, leaked ground truth (agent sees data it shouldn't), hidden
nondeterminism (global RNG, dict/set ordering, time-dependent code), places
where an agent can mutate the environment, hardcoded results, silent
exception swallowing, and claims in docs that the code does not support.
Produce a ranked list with file/line references and proposed fixes. Do not
fix anything yet.
```

### B.4 Integration Prompt (Step E)
```text
Integrate {{MODULE_ID}} into the existing system ONLY through the contracts in
docs/architecture/data_contracts.md. Register algorithms/agents in their
registries. Add an end-to-end integration test running a seeded simulation with
the new component, assert determinism, run the FULL regression suite, and update
docs (module boundaries, roadmap status, ADRs if a decision changed). Report any
contract changes explicitly — contract changes require my approval.
```

### B.5 Research Prompt (Step F)
```text
Design and run a reproducible experiment for {{HYPOTHESIS}} using the Experiment
Engine. Specify the hypothesis, baselines, metrics, number of seeds (justify it),
train/eval seed separation, held-out environments, and the statistical test with
its assumptions. Run it, store the record, and generate the report with a
Limitations section. Report negative or null results as clearly as positive ones.
Never alter a configuration after seeing results without logging it as a new
experiment.
```

---

## 4. Module Map and Build Sequence

```
M00 Foundation
 └─ M01 World + Environment ─ M02 Contracts + Simulation Core ─ M03 Agents + Perception
        └─ M04 Memory ─ M05 Visualization + Replay            ══► GATE 1  (reproducible multi-agent sim)
              ├─ M06 Search ─ M07 Adversarial/Game Theory      ══► GATE 2  (classical AI)
              ├─ M08 Logic ─ M09 CSP + Planning                ══► GATE 3  (symbolic AI)
              ├─ M10 Probability/Bayes/HMM ─ M11 MDP           ══► GATE 4  (probabilistic AI)
              ├─ M12 Classical ML                              ══► GATE 5
              ├─ M13 Reinforcement Learning                    ══► GATE 6
              ├─ M14 Neural Networks + Deep RL                 ══► GATE 7
              ├─ M15 Multi-Agent Systems                       ══► GATE 8
              ├─ M16 Evolution + Artificial Life
              └─ M17 Hybrid Agents                             ══► GATE 9
M18 Experiment + Analytics Engine   (skeleton after M05; grows with each phase)
M19 Persistence + API + Worker      (after Gate 2)
M20 Research Studio (frontend)      (grows alongside; full build after Gate 4)
M21 Knowledge Base + Research Report
M22 Showpiece Scenarios             ══► GATE 10 (research platform)
```

### Gates & Exit Criteria

| Gate | Pass Condition | Git Tag |
|---|---|---|
| **0** | `make check` green on clean clone; docs + ADRs real | `gate-M00` |
| **1** | Deterministic, replayable, forkable multi-agent sim; viewer; agents cannot mutate world | `gate-1` |
| **2** | Search + adversarial verified against oracles; Algorithm Duel works | `gate-2` |
| **3** | Logic, CSP, planning verified; provenance traces generated | `gate-3` |
| **4** | Bayes/HMM/belief filter verified; MDP exact solutions exported | `gate-4` |
| **5** | ML models match scikit-learn, beat baselines, data-leakage safe | `gate-5` |
| **6** | RL converges to exact Q\* on small MDPs; eval seeds strictly separated | `gate-6` |
| **7** | Gradients numerically verified; NumPy ≈ PyTorch; DQN works | `gate-7` |
| **8** | Multi-agent cooperation/competition measured; non-stationarity handled | `gate-8` |
| **9** | Hybrid ablations with statistical confidence intervals on held-out worlds | `gate-9` |
| **10** | Showpieces reproduce from one command; comprehensive research report complete | `gate-10` |

