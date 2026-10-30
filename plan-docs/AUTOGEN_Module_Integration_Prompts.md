# AUTOGEN AI — Module-Wise Integration Prompts

**Companion to:** `AUTOGEN_AI_Master_Implementation_Blueprint.md`
**Purpose:** Copy-paste prompts for a coding agent (Antigravity, Kiro, Cursor, Claude Code…) that build AUTOGEN one module at a time, wire every module into the previous ones, and end with a project that is actually *fascinating to watch and defend*.

> The blueprint says: *"Be able to open any module, explain its mathematics, inspect its implementation, run an experiment, and defend the results without depending on an LLM."*
> These prompts are written to protect that goal. The agent writes code. **You** verify the math.

---

## 0. How to use this file

### 0.1 The loop (per module)

1. Paste **Part 1 (Permanent Instructions)** once as the agent's system/rules file (e.g. `.kiro/steering`, `AGENTS.md`, `CLAUDE.md`, Antigravity rules).
2. Open the module below. Run its prompt in this order:
   - **Step A — Plan:** paste the module prompt but add at the top: `PLAN ONLY. Do not write code. Propose files, interfaces, pseudocode, risks, and acceptance criteria. Wait for approval.`
   - **Step B — Math review (you):** read the equations, check edge cases, approve or correct.
   - **Step C — Build:** paste the module prompt again without the PLAN ONLY line.
   - **Step D — Verify:** paste the **Verification Prompt** from Appendix B.
   - **Step E — Integrate:** paste the module's **Integration Hook** (included at the end of each module).
3. Commit only when the module's **Gate** is green. Tag it: `git tag gate-M06`.

### 0.2 Module map and build order

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
M18 Experiment + Analytics Engine   (start the skeleton right after M05; grow it every phase)
M19 Persistence + API + Worker      (after Gate 2 is enough)
M20 Research Studio (frontend)      (grows alongside; full build after Gate 4)
M21 Knowledge Base + Research Report
M22 Showpiece Scenarios             ══► GATE 10 (research platform)
```

Hard dependencies (from the blueprint): A\* needs only M01–M05. Bayes is independent of RL. Neural nets can be built standalone. **DQN needs M13 + M14.** **MARL needs M13 + a working multi-agent env.** **Neuroevolution needs M16 + a policy representation.**

### 0.3 What makes AUTOGEN fascinating (design hooks baked into the prompts)

| Hook | What the viewer experiences | Built in |
|---|---|---|
| **Glass-box agents** | Click any agent → see its real observation, belief, candidate actions with scores, chosen action, and *why*. | M03, M05, M20 |
| **Truth vs. Belief split-screen** | Left: the real world. Right: what the agent *thinks* the world is (probability heatmap). Watch belief collapse onto a hidden predator. | M10, M05, M20 |
| **Time Machine** | Scrub backwards through a run, **fork** it at tick *t*, change one action, and watch history diverge — deterministically. | M02, M05, M22 |
| **Algorithm Duel** | A\*, UCS, Greedy, BFS race on the *same map and seed*; frontiers expand side by side with live node counts. | M06, M20 |
| **Ecosystem Lab** | Evolving populations with lineages, speciation, famine, and boom–bust cycles. | M16, M22 |
| **Tragedy of the Commons** | MARL agents over-harvest a shared resource, then learn (or fail) to cooperate. | M15, M22 |
| **Ablation Arena** | Remove memory / Bayes / A\* / Q-learning from the hybrid agent and see exactly what each part is worth — with confidence intervals. | M17, M18, M22 |
| **Math Lab** | Every algorithm page: formula, derivation, live parameters, step-through. | M20, M21 |

### 0.4 Naming note

The blueprint's directory tree is titled `nexus-ai/` while the product is **AUTOGEN**. These prompts use `autogen-ai/` as the root. Rename if you prefer — nothing else depends on it.

---

## Part 1 — Permanent Coding-Agent Instructions (paste once)

```text
You are the senior AI systems engineer assisting with AUTOGEN AI.

AUTOGEN is a modular artificial intelligence research and simulation platform:
a configurable artificial world in which autonomous agents perceive, reason
under uncertainty, plan, learn, cooperate or compete, and adapt. The Python
simulation engine owns all simulation state. The frontend only visualizes and
controls it.

CORE RULES
- Mathematical correctness over implementation speed.
- Core algorithms are implemented FROM SCRATCH (NumPy/SciPy allowed for linear
  algebra and primitives). scikit-learn, PyTorch built-ins, and other libraries
  may be used ONLY as test oracles or declared baselines, never as the
  implementation of an algorithm meant to be learned.
- Never call an external LLM or pretrained model to make an agent intelligent.
- Layers stay independent: simulation, intelligence, memory, persistence, API,
  visualization. Algorithms must not import rendering code.
- Agents NEVER mutate the environment. The environment validates and resolves
  actions.
- Keep four things distinct: actual world state, agent internal state, agent
  observation, agent belief state.
- Typed interfaces (Pydantic / dataclasses / Protocols), explicit contracts.
- Every random choice flows from an explicit seed through named RNG streams
  (numpy SeedSequence.spawn). Training seeds and evaluation seeds are separate.
- Every important decision produces a DecisionTrace generated from ACTUAL
  execution. Never hardcode or mock traces.
- Never hardcode or fabricate experiment results, metrics, or findings.
- Do not claim completion without executing verification and showing output.
- Do not modify unrelated modules. Do not introduce microservices. Do not add
  abstractions or dependencies without a stated reason.
- Do not create empty stubs for future algorithms.

FOR EVERY ALGORITHM, DOCUMENT (docs/algorithms/<name>.md)
Theory, definition, assumptions, equations with variable definitions,
derivation where appropriate, pseudocode, complexity, implementation notes,
verification strategy, limitations, references.

EVERY ALGORITHM MUST BE
- Independently testable with unit, mathematical, and edge-case tests.
- Statistically tested across repeated seeds if stochastic.
- Instrumented (nodes expanded, iterations, time, updates as applicable).
- Step-traceable: expose an optional generator/callback that emits step events
  (frontier, internal state, numeric values) so the Algorithm Lab can animate it.
- Registered in the algorithm registry with metadata (name, family, version,
  capabilities, hyperparameter schema).

WORKFLOW
1 Inspect existing architecture. 2 Read relevant docs. 3 Identify dependencies.
4 Propose implementation + verification plan. 5 Identify mathematical risks.
6 Define acceptance criteria. 7 Implement the smallest functional unit.
8 Write and run tests. 9 Review regressions (run the FULL suite).
10 Update documentation.

DEFINITION OF DONE
Math spec documented · implementation complete · tests pass (show output) ·
outputs independently verified · integrated through existing contracts ·
limitations documented · working usage example · docs reflect actual code.

FINAL RESPONSE FORMAT
Repository tree (actual), files created/changed, tests executed with results,
verification evidence, known limitations, incomplete work, next milestone.
Then WAIT for approval.

Build a verifiable AI research system, not merely impressive-looking code.
```

---

## Part 2 — Module Prompts

> **Convention:** every module prompt starts with `MODULE`, lists `PREREQUISITES`, and ends with an `INTEGRATION HOOK` that wires it into earlier modules. Replace nothing in angle brackets unless you want to customize.

---

### M00 — Foundation (Gate 0)

**Blueprint refs:** §2, §9, §15, §16 · **Subsystems:** all (scaffolding only)

```text
MODULE M00 — Project Foundation
PREREQUISITES: none.

TASK: Initialize the AUTOGEN AI monorepo as a modular monolith. Do not implement
any AI algorithm in this task.

1. REPOSITORY
   Create the structure from the blueprint (apps/api, apps/web, packages/
   {simulation, agents, intelligence, memory, experiments, evaluation, analytics,
   visualization}, research/, docs/, tests/{unit,integration,mathematical,
   property_based,regression}, configs/, scripts/, benchmarks/, docker/).
   Create only directories that have a purpose today. Add package __init__ files
   and a single importable top-level layout. Propose the import strategy
   (e.g. `autogen.simulation`, `autogen.agents`) and justify it before creating.

2. TOOLING
   pyproject.toml: Python 3.12+, numpy, scipy, pydantic, pytest, pytest-cov,
   hypothesis (dev), ruff, mypy (strict for packages/, relaxed for tests).
   A single command surface: `make check` (ruff, mypy, pytest -q), `make test`,
   `make lint`. Add .gitignore, .editorconfig, pre-commit config.
   Logging: structured logging setup (one module), levels configurable.
   Config: Pydantic Settings + YAML loading with schema validation.

3. DOCS (real content, not placeholders)
   README.md; docs/architecture/{system_overview, module_boundaries,
   data_contracts, decision_records}.md; docs/mathematical_foundations/
   world_model.md (stub header only — filled in M01); docs/implementation/
   roadmap.md with every phase: learning objectives, math prerequisites,
   algorithms, modules, interfaces, tests, acceptance criteria, dependencies.
   Record ADR-001 (modular monolith), ADR-002 (Python owns simulation state),
   ADR-003 (from-scratch policy for core algorithms), ADR-004 (seeded RNG streams).

4. CI
   GitHub Actions workflow: ruff, mypy, pytest on push. Benchmarks excluded.

ACCEPTANCE
- `make check` passes on a clean clone.
- A trivial smoke test proves the package imports and config validation rejects
  a malformed YAML.
- Docs describe the repo as it ACTUALLY exists.

INTEGRATION HOOK: none (root module). Tag: gate-M00.
```

---

### M01 — Mathematical World & Environment Engine (SYS-02)

**Blueprint refs:** §3.1–3.5 · **Gate 1 (part 1)**

```text
MODULE M01 — World Model and Environment Engine
PREREQUISITES: M00.

GOAL: Implement the discrete 2D world exactly as mathematically specified, with
no agents' intelligence yet.

SPECIFY FIRST (docs/mathematical_foundations/world_model.md)
- Grid G = {0..H-1} x {0..W-1}; cell C_ij = (terrain, resource, danger,
  movement_cost). Terrain: grassland, water, forest, desert, mountain, obstacle.
- Per-terrain properties: walkable?, movement cost, resource capacity, regen rate.
- World state S_t = (E_t, A_t, R_t, W_t).
- Agent state s_i = (position, energy, health, carried_food, internal_memory_ref).
- Action space A = {UP, DOWN, LEFT, RIGHT, STAY, COLLECT}.
- Transition S_{t+1} = T(S_t, a_t, xi_t); deterministic mode omits xi_t.
- Resource dynamics: F_{t+1} = min(F_max, F_t + G_t - C_t).
- Agent energy: E_{t+1} = min(E_max, max(0, E_t - C_a + R_f)).
- Starvation and health damage are SEPARATE rules (never conflate energy/health).
- Population dynamics N_{t+1} = N_t + B_t - D_t (births disabled until M16).
- Reward and termination conditions (document each term).

IMPLEMENT (packages/simulation/environment, state)
- GridWorld with typed WorldState (immutable snapshots; mutation only inside the
  environment transition function).
- Seeded procedural generator: random obstacles/resources from configurable
  densities; guarantee connectivity of walkable cells (document the method).
- Resource regeneration and consumption.
- Weather/seasonal hooks as an optional stochastic component xi_t using its own
  RNG stream (off by default).
- Config schema (width, height, densities, terrain weights, E_max, costs, seed).
- Canonical state hashing (stable across runs/platforms) for reproducibility tests.

TESTS
- Movement into walls/obstacles/out-of-bounds is rejected; STAY always valid.
- Energy never leaves [0, E_max]; regeneration never exceeds F_max.
- Starvation damages health only after the documented rule triggers.
- Same seed => identical world hash; different seeds => different hashes.
- Property-based: random action sequences never violate invariants.
- Connectivity guarantee holds across 1000 generated worlds.

ACCEPTANCE: world fully specified, deterministic under seed, invariants tested.

INTEGRATION HOOK: expose BaseEnvironment-compatible methods (reset, validate,
apply) but do NOT define the engine loop yet — that is M02.
```

---

### M02 — Core Contracts & Simulation Engine (SYS-01)

**Blueprint refs:** §3.6, §4.1, §9.1, §16 · **Gate 1 (part 2)**

```text
MODULE M02 — Contracts and Deterministic Simulation Core
PREREQUISITES: M00, M01.

GOAL: Define the typed contracts and a deterministic discrete-time engine.

CONTRACTS (packages/simulation/*, agents/base)
BaseEnvironment, BaseAgent (observe, update_belief, decide, learn, reset),
Observation, Action (Command pattern), WorldState, AgentState, SimulationEngine,
SimulationEvent, DecisionTrace, AgentCapabilities (declared explicitly, e.g.
{memory, belief, planning, learning}). Use Protocols/ABCs + Pydantic models.

ENGINE: documented tick order (changing it changes outcomes — record as ADR)
1 generate observations  2 update beliefs/memory  3 request decisions
4 validate actions  5 resolve simultaneous actions  6 apply environment
transition  7 compute rewards/costs  8 update metrics + events
9 publish new state.

DESIGN DECISIONS TO DOCUMENT AND IMPLEMENT
- Conflict resolution for two agents moving into the same cell and for
  contested COLLECT: choose a policy (e.g. both blocked / priority by seeded
  random permutation per tick), make it a config option, test every branch.
- Invalid actions: environment substitutes STAY and emits an
  InvalidActionEvent (agent is never allowed to bypass validation).
- Event bus (Observer pattern): typed events, subscribers, no rendering imports.
- RNG: one SeedSequence root; spawned streams per subsystem and per agent.

TIME-MACHINE SUPPORT (important for later "fascination" features)
- Full-state snapshot() / restore() that is exact (bit-identical hash).
- Replay log: seed + config + per-tick joint actions => exact reproduction.
- fork(snapshot, tick, overridden_actions) => a new independent run.
- Snapshot cadence configurable (don't snapshot every tick by default).

TESTS
- Run 500 ticks twice with the same seed => identical state hash every tick.
- snapshot at tick 200, restore, continue => identical to uninterrupted run.
- Replay log reproduces a run exactly; a fork with an unchanged action list
  equals the original; a fork with one changed action diverges.
- Simultaneous-conflict resolution tests for all configured policies.
- Agents cannot reach env internals (test via a hostile agent that tries).

ACCEPTANCE: a deterministic, replayable, forkable headless simulation runs with
dummy agents; contracts documented in docs/architecture/data_contracts.md.

INTEGRATION HOOK: M01 world plugged in through BaseEnvironment only; add an
integration test that runs M01 + engine + two scripted dummy agents for 1000 ticks.
```

---

### M03 — Agent Framework, Perception & Baseline Agents (SYS-03, SYS-04)

**Blueprint refs:** §4, Phase 01 · **Gate 1 (part 3)**

```text
MODULE M03 — Agent Framework, Perception Engine, Baseline Agents
PREREQUISITES: M01, M02.

PERCEPTION ENGINE
- Converts true WorldState into a PERMITTED Observation per agent.
- Modes: fully observable; local window radius r; line-of-sight with obstacle
  occlusion; optional sensor noise (own RNG stream); optional limited sensing
  of other agents.
- Test that an agent can never see outside its permitted observation (property
  test: inject sentinel values outside the window and assert they never leak).

AGENT FRAMEWORK
- Agent registry + factory from config (Registry + Factory patterns).
- Capability declaration; strategy-swappable decision component.
- DecisionTrace builder used by every agent: observation summary, belief
  summary, candidate actions with numeric scores, selected action, mechanism
  name, human-readable reason — all populated from real computation.

AGENTS (implement exactly these, nothing more)
1 RandomAgent (seeded)
2 SimpleReflexAgent: a_t = pi(o_t) — condition-action rules (food visible ->
  COLLECT/move toward, obstacle ahead -> turn, else move).
3 ModelBasedReflexAgent: maintains an internal map built from observations
  (this is internal state, not belief probabilities yet).

PERFORMANCE EVALUATOR
Discounted performance J = sum gamma^t r_t, survival time, resources collected,
energy efficiency, exploration coverage. Per-agent and aggregate.

EXPERIMENTS (use the M18 skeleton if it exists; otherwise a minimal script)
random vs reflex; full vs partial observability; sensor radius sweep; resource
density sweep. Seeds fixed; results written to disk, not printed only.

TESTS: agents obey the interface; decision traces are present and non-empty for
every tick; ModelBased agent's map equals ground truth only where observed;
reflex beats random in a controlled setup across >= 30 seeds (statistical test,
not a single run).

ACCEPTANCE: agents operate independently, receive only permitted observations,
decisions are inspectable, metrics recorded.

INTEGRATION HOOK: register all agents in the registry; run the M02 engine with
20 mixed agents on a 50x50 world and confirm throughput + determinism.
```

---

### M04 — Memory Engine (SYS-05)

**Blueprint refs:** §1.2 (Memory), Phase 01/13 hybrid agents

```text
MODULE M04 — Memory Engine
PREREQUISITES: M02, M03.

GOAL: Give agents retrievable experience, then measure whether it helps.

IMPLEMENT (packages/memory)
- Working memory: small bounded buffer (FIFO/LRU) of recent observations.
- Episodic memory: experience tuples (t, o, a, r, o', done) with metadata.
- Spatial memory: remembered cells (terrain, last-seen tick, last-seen resource
  level, confidence that decays with staleness).
- Retrieval: by recency, by spatial proximity, by similarity (cosine on feature
  vectors) — each as a separate, testable strategy.
- Forgetting: configurable capacity + decay policy.
- Memory Explorer data API: read-only view of memory contents for the UI.

DOCUMENT: what a memory fact is, how staleness is modelled, why remembered ≠
believed (belief arrives in M10; memory stores evidence, belief interprets it).

AGENT INTEGRATION: MemoryAugmentedAgent (wrapper/decorator, not a rewrite) that
adds memory to any baseline agent.

EXPERIMENT: memory vs no-memory on dynamic-resource worlds; capacity sweep;
forgetting-rate sweep. Report effect sizes with confidence intervals (M18 stats
helpers when available).

TESTS: capacity limits respected; retrieval returns expected items on crafted
cases; decay is monotonic; memory never contains data the agent didn't observe.

INTEGRATION HOOK: wire MemoryAugmentedAgent into the factory; DecisionTrace gains
a "memory used" field listing which retrieved items influenced the decision.
```

---

### M05 — First Visualization, Decision Traces & Time Machine (SYS-18 v0)

**Blueprint refs:** §7.4, §7.5, §15 · **Gate 1 complete**

```text
MODULE M05 — Pygame Viewer, Decision Trace Mode, Replay
PREREQUISITES: M02–M04.

ARCHITECTURE RULE: the viewer is a CONSUMER. It reads snapshots/events from the
engine (queue or log file). It never advances simulation state on its own and
never contains AI logic. Rendering speed is independent of simulation speed.

FEATURES
- Grid rendering: terrain colors, resource intensity, agents with IDs.
- Controls: play, pause, single-step, speed multiplier, reset, seed display.
- Agent selection (click) -> Decision Trace panel showing the REAL trace:
  observation, belief/internal summary, candidate actions + scores, selected
  action, mechanism, reason, memory used.
- Overlays (toggle): fog-of-war / agent field of view, movement trails,
  resource heatmap, danger heatmap, collision markers, event timeline.
- TRUTH vs AGENT-VIEW split screen: left = true world, right = what the selected
  agent currently knows (its internal map now; its belief heatmap after M10).
- TIME MACHINE: replay a recorded run with a scrubber (rewind/forward by
  restoring snapshots), and "fork here": pick a tick, optionally override one
  agent's action, spawn a new run that visibly diverges.
- Headless mode still works without pygame installed.

TESTS: render-free tests on the data adapter (what the viewer would show);
replay equals original run; fork produces divergence only after the fork tick.
A manual checklist in docs/implementation/m05_manual_qa.md.

ACCEPTANCE (Gate 1): reproducible multi-agent simulation with a working viewer,
inspectable decisions, replay + fork. Documentation matches reality.

INTEGRATION HOOK: provide a stable ViewerState DTO — M20's React app will
consume the same DTO over WebSocket later, so keep it JSON-serializable.
```

---

### M06 — Search Engine (SYS-06) — Gate 2

**Blueprint refs:** Phase 02

```text
MODULE M06 — Search Algorithms and Problem Solving
PREREQUISITES: M01–M05.

GENERIC FRAMEWORK
SearchProblem protocol: initial_state, is_goal, successors(state) ->
(action, next_state, cost), heuristic(state). Pure functions; no dependency on
rendering or the simulator internals (adapter builds a SearchProblem from an
agent's belief/known map).

IMPLEMENT FROM SCRATCH
Uninformed: BFS (FIFO, visited, parents, path reconstruction), DFS (stack +
recursive, cycle detection, depth limit), Depth-Limited, Iterative Deepening,
Bidirectional BFS, Uniform Cost Search (g(n) = sum of edge costs).
Informed: Greedy Best-First f=h, A* f=g+h, Weighted A*, IDA*.
Local/stochastic: Hill Climbing (with restarts), Simulated Annealing
(accept worse with P = exp(-dE/T), configurable cooling schedule), Local Beam,
Beam Search.

MATH TO DOCUMENT AND TEST
- Complexities (BFS/DFS O(V+E), etc.), completeness, optimality conditions.
- Heuristics: Manhattan for 4-connected grids. IMPORTANT: with terrain costs,
  Manhattan is admissible only if scaled by the MINIMUM step cost; document it
  and test h(n) <= h*(n) numerically. Also consistency (h(n) <= c(n,n') +
  h(n')) and how the implementation handles reopening if consistency fails.
- Weighted A* suboptimality bound w * C*.

INSTRUMENTATION: nodes expanded, nodes generated, max frontier size, path cost,
runtime, peak memory. Every algorithm also yields STEP EVENTS (generator) with
frontier/explored snapshots for the Algorithm Lab and the Algorithm Duel.

AGENTS: GoalBasedAgent using a pluggable search strategy, replanning when new
observations invalidate its path; path cache with invalidation.

VERIFICATION (independent oracles)
- Returned paths are valid action sequences on the grid.
- UCS and A* costs equal scipy.sparse.csgraph.dijkstra on 500 random grids.
- BFS = shortest in edge count on unweighted grids.
- Admissible-heuristic A* never returns costlier than UCS; inadmissible
  heuristic test demonstrates suboptimality exists (so the test has teeth).
- All algorithms in a comparison use identical maps and seeds.

EXPERIMENT: algorithm comparison (path cost, nodes expanded, runtime, memory)
across obstacle densities and map sizes; statistical comparison with CIs.

INTEGRATION HOOK: register algorithms; GoalBasedAgent in factory; ALGORITHM
DUEL command: run N algorithms on the same map+seed in lockstep and emit
combined step events for the viewer (M05) and later the UI (M20).
```

---

### M07 — Adversarial Search & Game Theory (SYS-10)

**Blueprint refs:** Phase 03

```text
MODULE M07 — Adversarial Search and Game Theory
PREREQUISITES: M06 (search framework), M02.

MODELING NOTE: the simulator is simultaneous-move. Provide two adapters and
document when each is valid: (a) a turn-based AdversarialGame abstraction for
sub-games (predator vs prey lookahead with alternating moves), and
(b) a simultaneous-move normal-form (matrix) game for local interactions.

IMPLEMENT FROM SCRATCH
- Minimax with depth limit + evaluation function; Alpha-Beta (prune when
  alpha >= beta) with move ordering; Expectimax for stochastic opponents.
- Evaluation functions for predator-prey (distance, escape routes, energy).
- Normal-form games G=(N,A,U): payoff matrices; best responses; dominant
  strategies; pure Nash by enumeration; mixed Nash for 2x2 analytically and
  small bimatrix games via support enumeration; verify the equilibrium
  condition U_i(s_i*, s_-i*) >= U_i(s_i, s_-i*).
- Games for resource competition: Prisoner's Dilemma (and iterated with
  tit-for-tat variants), Hawk-Dove, Stag Hunt.
- Agents: PredatorAgent (alpha-beta), PreyAgent (minimax/expectimax),
  StochasticOpponent, StrategicForager.

VERIFICATION
- Alpha-Beta value == Minimax value on 1000 random game trees; node count
  <= Minimax; with best-first ordering approaches O(b^(d/2)) on crafted trees.
- Tic-Tac-Toe solved value = draw. Hand-checked textbook trees.
- Matching Pennies mixed Nash = (0.5, 0.5); Prisoner's Dilemma unique pure Nash;
  equilibrium checker rejects non-equilibria.

EXPERIMENT: predator depth vs prey success; minimax vs expectimax vs random
opponent; payoff-matrix tournaments. Measurable differences with CIs.

INTEGRATION HOOK: predator/prey scenario config in the engine; "Hunt" preset in
M05 viewer showing the lookahead tree for the selected agent (from real data).
```

---

### M08 — Knowledge Representation & Logical Reasoning (SYS-07)

**Blueprint refs:** Phase 04 · **Gate 3 (part 1)**

```text
MODULE M08 — Logic Engine
PREREQUISITES: M02, M03.

IMPLEMENT FROM SCRATCH
- Propositional logic: expression AST, parser (documented grammar), pretty
  printer, truth-table evaluation, CNF conversion, satisfiability, entailment.
- Inference: modus ponens; forward chaining and backward chaining on Horn
  clauses; resolution refutation; DPLL (optional but recommended).
- First-order logic: terms, predicates, quantifier handling via skolemization
  for resolution; unification with occurs check; substitution composition.
- Knowledge base: add/retract facts and rules, contradiction detection.
- PROVENANCE: every derived conclusion stores a proof tree (which facts/rules
  produced it). Expose this to the Reasoning Inspector.

WORLD-LINKED KNOWLEDGE (agent-facing)
Predicates such as Track(x,y), PredatorNear(r), Safe(x,y), Water(x,y).
Rules like Track(x,y) -> PredatorMayBeNear(x,y). A SafetyRuleLayer agents can use
as a hard filter on candidate actions ("never enter a cell not provably safe
when energy is above T").

VERIFICATION
- Resolution and truth-table entailment agree on 5000 random small formulas.
- Forward/backward chaining agree on Horn KBs.
- Unification: Knows(x, Food) with Knows(Agent1, Food) gives {x/Agent1};
  occurs-check test (x vs f(x)) fails correctly.
- Contradictory KB detected; provenance reconstructs a valid proof.

INTEGRATION HOOK: LogicAgent wrapper that filters/justifies another agent's
actions; DecisionTrace gains "symbolic justification" with the proof tree id.
```

---

### M09 — Constraint Satisfaction & Automated Planning (SYS-08)

**Blueprint refs:** Phase 05 · **Gate 3 complete**

```text
MODULE M09 — CSP and Planning
PREREQUISITES: M06 (search), M08 (logic representations).

CSP = (X, D, C) FROM SCRATCH
Backtracking; MRV; Degree heuristic; Least Constraining Value; Forward Checking;
Arc consistency (AC-3: remove x_i if no supporting x_j in D_j). Problems:
map coloring, N-Queens, Sudoku, and a world-linked one: patrol/foraging schedule
assignment for multiple agents.

PLANNING P = (S, A, Pre, Eff, G)
- STRIPS representation: predicates, ground actions, preconditions, add/delete
  effects, state transitions, goal test.
- Forward state-space planner (BFS and A*) with domain-independent heuristics
  (h_max, h_add via delete relaxation).
- Independent PLAN VALIDATOR (separate code path from the planner).
- Detect unsolvable problems (exhaustive search proves it, or relaxed
  reachability proves it).
- Optional/advanced: planning graph + Graphplan, partial-order planning, HTN.

WORLD-LINKED DOMAIN: a crafting/survival domain (gather wood -> build shelter ->
store food -> survive winter) executed by an agent inside the simulator; the
plan's primitive steps compile to grid actions via the search module.

VERIFICATION
- CSP: solutions satisfy all constraints (independent checker); AC-3 prunes
  exactly what a brute-force support check says; MRV/LCV reduce backtracks
  measurably vs naive ordering (recorded, not assumed).
- Planning: validator accepts planner output and rejects mutated plans; known
  unsolvable instance is reported unsolvable; optimal plan length matches BFS.

INTEGRATION HOOK: PlanningAgent in factory; Gate 3 experiment comparing
reflex vs search vs planner on multi-step goals.
```

---

### M10 — Probability, Bayesian Reasoning & Belief States (SYS-09)

**Blueprint refs:** Phase 06 · **Gate 4 (part 1)**

```text
MODULE M10 — Probabilistic Engine
PREREQUISITES: M03, M04 (memory), M08 optional.

IMPLEMENT FROM SCRATCH
- Discrete distributions with normalization guarantees; Bernoulli, Binomial,
  Gaussian, Poisson utilities (verified against scipy.stats as an oracle).
- Bayes' theorem updater: P(H|E) = P(E|H)P(H)/P(E), with log-space option.
- Bayesian networks: DAG, CPTs, joint factorization, exact inference by
  enumeration and variable elimination, approximate inference by likelihood
  weighting and Gibbs/rejection sampling.
- Markov chains and Hidden Markov Models: forward, backward, Viterbi,
  forward-backward smoothing; Baum-Welch (optional).
- GRID BELIEF FILTER (histogram filter): b_{t+1}(s') = eta * O(o|s') *
  sum_s T(s'|s,a) b_t(s). Used to track a hidden predator / unseen resource
  levels over the grid with a motion model and a sensor model.
- POMDP foundations: belief state object, belief update, expected-utility action
  selection one step ahead.

KEEP DISTINCT: actual state vs observation vs belief. A BeliefState is a
probability distribution over world states and is the ONLY thing a
belief-based agent consults for hidden variables.

VERIFICATION
- All distributions sum to 1; probabilities in [0,1] (property tests).
- Variable elimination == enumeration on random small networks.
- Sampling methods converge to exact posteriors (statistical test with tolerance
  that shrinks with samples).
- HMM forward probability equals brute-force sum over all hidden sequences on
  small chains; Viterbi equals brute-force argmax.
- Belief filter: with a perfect sensor, belief collapses to truth; with a
  useless sensor, belief equals the motion-model prediction.
- Entropy and KL(truth || belief) decrease on average with informative sensing.

AGENT: BayesianHazardAgent (estimates danger/predator location, chooses actions
maximizing expected utility under belief).

INTEGRATION HOOK: expose belief heatmap in the ViewerState; M05's split screen
right panel now shows the agent's probability map next to the true world
("Ghost in the Fog"). DecisionTrace gains belief summary: top-k cells with
probabilities, entropy.
```

---

### M11 — Decision Theory & MDPs

**Blueprint refs:** Phase 07 · **Gate 4 complete**

```text
MODULE M11 — Decision Theory and MDPs
PREREQUISITES: M01, M10.

IMPLEMENT
- Expected utility EU(a) = sum_s P(s|a)U(s); argmax selection; utility
  functions U(s) = w_e E + w_h H + w_f F - w_r R with weights as experimental
  parameters.
- MDP = (S, A, T, R, gamma) with explicit T and R tensors (small, finite).
- Builder that converts a SMALL grid world (e.g. 6x6 with slip probability) into
  an exact MDP.
- Iterative policy evaluation (Bellman expectation), Value Iteration (Bellman
  optimality), Policy Iteration; convergence monitoring (max-norm residual,
  theoretical bound with gamma).
- Policy and value-function visualization data (arrows + heatmap) for M05/M20.

VERIFICATION
- Analytically solvable 2-state and 3-state MDPs checked by hand-derived values.
- Policy evaluation equals the direct linear solve (I - gamma P_pi)^-1 R_pi.
- Value Iteration and Policy Iteration reach the same V* and policy.
- Classic textbook gridworld results reproduced.

IMPORTANT: this module produces EXACT V*, Q*, pi* that M13 will use as ground
truth to verify learning algorithms. Export them in a documented format.

INTEGRATION HOOK: MDPAgent that follows pi* in the small world; UtilityAgent
using EU over belief states from M10 (Bayesian + decision-theoretic agent).
```

---

### M12 — Classical Machine Learning (SYS-11)

**Blueprint refs:** Phase 08 · **Gate 5**

```text
MODULE M12 — Classical ML Engine
PREREQUISITES: M01–M05 (data comes from simulation logs), M10 optional.

DATA PIPELINE
Dataset builder that turns simulator logs into supervised datasets with
leakage-safe splits (train/validation/test split by RUN/SEED, never by random
row, because adjacent ticks are correlated). Feature scaling fit on train only.

IMPLEMENT FROM SCRATCH (NumPy)
Linear regression (normal equation; batch/SGD/mini-batch GD; regularization;
polynomial features) · Logistic regression (sigmoid, BCE, GD) · Decision trees
(entropy/information gain ID3, Gini/CART, pruning) · Random forest · KNN ·
SVM (soft-margin; Pegasos or SMO; kernels) · K-Means (k-means++) · PCA
(covariance eigendecomposition, explained variance) · Gaussian Mixture Models +
EM · optional: hierarchical clustering, DBSCAN.
Metrics: precision, recall, F1, ROC/AUC, MAE, MSE, RMSE, R^2, confusion matrix,
k-fold cross-validation.

AUTOGEN APPLICATIONS (build at least three)
Predict resource availability · classify cell danger · cluster agent behaviour
(feeds M15 emergent-behaviour detection) · predict energy requirement ·
learn resource distribution patterns.

VERIFICATION
- Each model compared with the scikit-learn equivalent on the same data
  (predictions/coefficients within documented tolerance) — sklearn lives only
  in tests/benchmarks.
- Every model beats a trivial baseline (majority class / mean predictor) or the
  report explains why not.
- Leakage test: a deliberately leaky split must be detected by the splitter's
  guard.
- PCA reconstruction error decreases with components; K-Means inertia is
  non-increasing per iteration; EM log-likelihood is non-decreasing.

INTEGRATION HOOK: LearnedPerceptionComponent (e.g. danger classifier) that agents
can plug in; model artifacts saved with metadata (data hash, seed, version).
```

---

### M13 — Reinforcement Learning (SYS-12)

**Blueprint refs:** Phase 09 · **Gate 6**

```text
MODULE M13 — Reinforcement Learning Engine
PREREQUISITES: M01–M05, M11 (exact V*/Q* as ground truth), M12 optional.

ADAPTER: a single-agent Gym-style wrapper over the simulator (Adapter pattern):
reset(seed), step(action) -> (obs, reward, terminated, truncated, info). Keep it
a THIN adapter; the simulator remains the source of truth. State encoder
interface (tabular key vs feature vector).

IMPLEMENT FROM SCRATCH
- Multi-armed bandits: greedy, epsilon-greedy (with decay
  eps_t = max(eps_min, eps_0 * lambda^t)), UCB, Thompson Sampling.
- Tabular control: Q-learning (off-policy), SARSA (on-policy), Expected SARSA,
  Double Q-learning.
- Prediction/control: Monte Carlo (first-visit, every-visit), TD(0), n-step TD,
  eligibility traces (TD(lambda)).
- Model-based: Dyna-Q.
- TERMINAL HANDLING: target = r on terminal transitions, no bootstrap
  (explicit test).
- Training harness: separate TRAIN seeds and EVAL seeds; periodic greedy
  evaluation; checkpointing; logs of TD error, epsilon, returns, episode length.

VERIFICATION
- On small MDPs from M11, learned Q converges toward Q* within tolerance
  (report the error curve); policy matches pi* on states visited enough.
- Terminal target test: Target = r exactly.
- CliffWalking: SARSA learns the safer path, Q-learning the risky optimal path
  (qualitative behavioural test).
- Bandit regret: UCB/Thompson regret grows sublinearly on stationary arms.
- Do NOT accept a pretty reward curve as proof; require comparison to Q*,
  random baseline, and multiple seeds with CIs.

AGENT: QLearningAgent, SarsaAgent, DynaAgent in factory (learn() implemented).

INTEGRATION HOOK: the Learning Studio data API (training curves, Q-value
heatmaps, policy arrows) in the ViewerState/metrics stream; DecisionTrace shows
Q-values of each action and whether the choice was exploratory or greedy.
```

---

### M14 — Neural Networks & Deep RL (SYS-13) — Gate 7

**Blueprint refs:** Phase 10, DQN from Phase 09

```text
MODULE M14 — Neural Network Engine
PREREQUISITES: M12 (datasets/metrics), M13 (RL harness).

PART 1 — NUMPY MLP FROM SCRATCH (no autograd)
Layers, activations (sigmoid, tanh, ReLU, Leaky ReLU, softmax), losses (MSE,
BCE, categorical CE), forward pass, full backpropagation using
delta^(l) = (W^(l+1)T delta^(l+1)) ⊙ sigma'(z^(l)); dL/dW^(l) = delta^(l)
a^(l-1)T. Optimizers: SGD, Momentum, RMSProp, Adam (with bias correction),
learning-rate schedules, weight decay, dropout, early stopping, checkpointing.

PART 2 — PYTORCH (after Part 1 is verified)
- PyTorch twin of the NumPy MLP.
- DQN: replay buffer, target network theta^-, y = r + gamma max Q(s',a';theta^-)
  with y = r on terminal, gradient clipping, training diagnostics.
- Policy gradient family: REINFORCE with baseline, Actor-Critic/A2C, then PPO.
- Neural survival predictor; small CNN terrain classifier.
- Training visualization data (loss, gradient norms, Q statistics).

VERIFICATION
- NUMERICAL GRADIENT CHECK in float64:
  dL/dtheta_i ≈ (L(theta_i+eps) - L(theta_i-eps)) / (2 eps); relative error
  < 1e-6 for every layer/activation/loss combination.
- NumPy MLP and PyTorch twin with identical weights/data produce identical
  outputs and gradients (tolerance documented).
- Learns XOR; fits sklearn digits above a documented accuracy.
- DQN: terminal target test, replay buffer sampling statistics, learns on a
  tabular-sized env where M11 provides Q*, then beats tabular on a larger
  partially observable variant (report honestly if it does not).
- Do not "rush to deep learning": if Part 1 gradient checks fail, stop.

INTEGRATION HOOK: DQNAgent and PPOAgent through the same BaseAgent contract;
CNN terrain classifier exposed as a perception component; checkpoints stored via
model_checkpoints metadata (M19).
```

---

### M15 — Multi-Agent Systems & Collective Intelligence (SYS-14)

**Blueprint refs:** Phase 11 · **Gate 8**

```text
MODULE M15 — Multi-Agent Engine
PREREQUISITES: M02, M03, M10, M13, M14 (for CTDE), M12 (behaviour clustering).

ENVIRONMENT FEATURES
- Shared scarce resources with a commons dynamic (regrowth depends on how much
  is left — over-harvest collapses it): the Tragedy of the Commons scenario.
- Communication channel: messages m_i^t = Message(s_i^t) with configurable
  bandwidth, range, noise, and ENERGY COST (communication must not be free).
- Reward modes: individual r_i, shared R_team = sum r_i, and mixed
  (alpha-weighted). Document each.

IMPLEMENT
- Independent Q-learning; independent policy gradient.
- Centralized Training with Decentralized Execution (centralized critic).
- Value decomposition (VDN, then QMIX-lite).
- Competitive self-play with opponent pool / frozen snapshots.
- Coordination protocols: contract-net task allocation, stigmergy (pheromone
  fields), simple auctions (mechanism design).
- Emergent-behaviour detector: cluster per-agent behaviour features with the
  M12 K-Means/GMM to find roles (scouts, hoarders, free-riders).

METRICS: individual and group return, group survival rate, Gini coefficient of
resources, cooperation index, commons depletion time, communication volume.

NON-STATIONARITY: evaluate learners against frozen opponent snapshots and
population mixes; log learning curves per agent; document that independent
learners face a moving target and test for it.

VERIFICATION
- Reduces to single-agent behaviour when N=1 (regression test vs M13).
- Matrix-game sanity: IQL on Prisoner's Dilemma / Stag Hunt behaves as
  theory predicts under given exploration schedules.
- Shared vs individual reward experiment shows a measurable, statistically
  tested difference (or honestly reports none).
- Determinism holds with N agents under seed.

INTEGRATION HOOK: Multi-Agent Arena preset; ViewerState exposes per-agent role
labels, message arrows, and commons level over time.
```

---

### M16 — Evolutionary Computation & Artificial Life (SYS-15)

**Blueprint refs:** Phase 12

```text
MODULE M16 — Evolution Engine
PREREQUISITES: M02–M05, M13/M14 (for neuroevolution policy representations).

IMPLEMENT FROM SCRATCH
Genetic algorithm (chromosome, fitness, roulette/tournament/rank selection,
crossover variants, mutation with p_m, elitism, replacement) · Evolution
Strategies (mu+lambda, self-adaptive sigma) · Differential Evolution · Particle
Swarm (v update with omega, c1, c2) · small Genetic Programming (expression
trees) · Neuroevolution with a FIXED topology first, then NEAT-lite (structural
mutation + speciation).

ARTIFICIAL LIFE LAYER
Genome -> traits (vision radius, metabolism, speed/energy trade-off, aggression,
cooperation tendency). Births and deaths enable N_{t+1} = N_t + B_t - D_t with
documented reproduction cost, mutation, and carrying capacity. Lineage tracking
(parent ids) for a family-tree visualization. Diversity metrics (genotype
variance, species count, behavioural entropy).

VERIFICATION
- GA solves OneMax; DE/PSO/ES minimize sphere and Rastrigin within tolerance
  statistically over seeds. Selection pressure test: higher pressure increases
  mean fitness faster but reduces diversity.
- Elitism guarantees best fitness is non-decreasing.
- Evolved policies are evaluated on UNSEEN environments (generalization gap
  reported). Optimizing a chosen fitness is not "general intelligence".
- Population conservation: births - deaths accounting matches N_t exactly.

EXPERIMENTS: evolution of navigation strategies; mutation-rate and selection
pressure sweeps; neuroevolution vs gradient-based learning (M14) with equal
compute budget.

INTEGRATION HOOK: EvolvedAgent via factory; Ecosystem preset that runs
generations headless and exports lineage + diversity series for the viewer.
```

---

### M17 — Advanced Planning, Meta-Learning & Hybrid Agents — Gate 9

**Blueprint refs:** Phase 13, §6.4

```text
MODULE M17 — Hybrid Intelligence
PREREQUISITES: M04, M06, M08, M10, M13 (M14/M16 optional).

IMPLEMENT
- Monte Carlo Tree Search with UCT_i = w_i/n_i + c sqrt(ln N / n_i).
- Model Predictive Control: plan over horizon H with a model, execute the first
  action, observe, re-plan.
- Options framework omega = (I, pi, beta) and a small hierarchical agent
  (high-level picks options like "go to water", "forage", "flee").
- Model-based RL (Dyna from M13) over a learned model.
- HYBRID AGENT with swappable components:
    A* for navigation · Bayesian belief for hazard estimation · symbolic
    safety rules as a SHIELD · Q-learning for high-level action selection ·
    memory for past experience.
  Each component can be toggled by config.

ABLATION STUDIES (mandatory)
Run Full, minus-Memory, minus-Bayes, minus-A*, minus-Q-learning, plus the
baselines (Random, Reflex, best single-method agent). Same seeds, held-out
evaluation worlds, paired comparisons, effect sizes, multiple-comparison
correction.

VERIFICATION
- MCTS on a known-bandit tree converges to the best arm; with infinite budget
  approaches the minimax value on small games.
- Safety shield property test: over thousands of random states the shield
  never permits an action that violates a safety rule.
- Hybrid vs simpler baselines: report honestly. More components do not
  automatically mean more intelligence — a negative result is valid evidence.

INTEGRATION HOOK: DecisionTrace of the hybrid agent shows which component
contributed what at each tick (navigation target from A*, belief value from
Bayes, shield vetoes, Q-values). This is the flagship "glass-box agent".
```

---

### M18 — Experiment Engine & Analytics (SYS-16, SYS-17)

**Blueprint refs:** §6, §12 · *Start the skeleton right after M05; extend every phase.*

```text
MODULE M18 — Experiment and Analytics Engine
PREREQUISITES: M02 (skeleton after M05), grows with every later module.

EXPERIMENT ENGINE
- YAML experiment configs validated by Pydantic (name, seed, episodes,
  environment, agents, metrics, evaluation protocol, ablation matrix).
- Algorithm Registry lookup by name+version.
- Runner: independent runs parallelized across seeds (multiprocessing first;
  parallelize independent runs before coupled simulations). Separate train and
  evaluation seed ranges; held-out environment sets.
- Reproducibility record per run: config hash, seeds, algorithm versions,
  environment version, software versions (python, numpy, torch), git commit,
  checkpoints, metrics, timing/execution metadata.
- Ablation runner that expands a component matrix into runs.

METRICS: survival rate SR = N_survived/N_initial, success rate, average
cumulative reward, path efficiency PE = L_optimal/L_actual, resource efficiency,
population growth rate, runtime, memory, nodes expanded, model updates,
simulation throughput (ticks/s).

STATISTICS (from scratch with scipy as oracle)
mean, sample variance (N-1), standard error, 95% normal CI, median, IQR,
bootstrap CI, paired comparisons, effect sizes (Cohen's d, Cliff's delta),
permutation test, paired t-test with assumption diagnostics (otherwise fall back
to permutation/Wilcoxon), Holm-Bonferroni correction.

REPORTING: auto-generated markdown report per experiment (tables + figures),
including a mandatory LIMITATIONS section filled from run metadata (e.g. number
of seeds, worlds, known confounds). Never write conclusions the data does not
support.

VERIFICATION
- Re-running an experiment from its saved record reproduces identical metrics.
- Statistics match scipy on random data; bootstrap coverage test on known
  distributions.
- Config schema rejects malformed/ambiguous experiments.
- Fast commit CI excludes expensive benchmarks (separate marker/workflow).

INTEGRATION HOOK: every previous module's experiments run through this engine;
results land in the persistence layer (M19) and Analytics UI (M20).
```

---

### M19 — Persistence, API & Simulation Worker

**Blueprint refs:** §8

```text
MODULE M19 — Persistence, FastAPI, WebSocket Stream, Worker
PREREQUISITES: M02, M05 (ViewerState DTO), M18. Useful after Gate 2.

ARCHITECTURE: modular monolith. HTTP handlers NEVER run simulations; a separate
worker process owns running simulations and experiments. Use a job queue
(multiprocessing/queue first; keep an interface so a real broker can replace it).

PERSISTENCE (Repository pattern, SQLite now, PostgreSQL-compatible schema)
Tables: users (optional), environments, environment_versions, agents,
agent_versions, algorithms, simulations, simulation_events, agent_states
(snapshots only), experiments, experiment_runs, metrics, model_checkpoints,
research_notes. Do NOT store every agent state every tick: configurable
snapshot cadence, event logs, compressed trajectories, aggregate metrics;
large artifacts on disk/object storage with references + metadata in rows.
Migrations (Alembic).

API (Pydantic v2 schemas, explicit validation, OpenAPI)
POST/GET /api/v1/environments, /agents; POST /api/v1/simulations and
/{id}/start|pause|step|reset; GET /{id}/state and /metrics;
POST /api/v1/experiments, /{id}/run; GET /{id}/results.
WebSocket /api/v1/simulations/{id}/stream: ViewerState frames, delta-encoded,
throttled with backpressure (drop frames, never block the engine); control
messages (play/pause/speed) validated.

VERIFICATION
- API tests with TestClient incl. validation errors; worker lifecycle tests
  (start/pause/step/reset/cancel); WebSocket test receives frames in order.
- Killing the worker mid-run leaves a consistent DB and resumable state.
- Determinism: a simulation run via API equals the same run headless.

INTEGRATION HOOK: export OpenAPI JSON and generate the typed TS client for M20.
```

---

### M20 — AUTOGEN Research Studio (SYS-18)

**Blueprint refs:** §7 · *Build in vertical slices; don't wait for all modules.*

```text
MODULE M20 — Research Studio Frontend
PREREQUISITES: M19 (API), generated TS client. Slices can start earlier against
recorded ViewerState files.

STACK: React + TypeScript + Vite + Tailwind CSS + Plotly (+ KaTeX for math).
The browser is a VIEWER. It never decides agent behaviour and is never the
source of truth.

BUILD IN SLICES (each slice shippable)
Slice 1  Live Simulation: canvas renderer, play/pause/step/speed, agent
         selection, trails, heatmaps, event timeline, WebSocket hook.
Slice 2  Decision Trace Mode (real traces), Truth-vs-Belief split screen.
Slice 3  World Builder: grid size, terrain painting, resources, obstacles,
         agent spawning, weather, visibility, presets, procedural generation,
         seed control.
Slice 4  Algorithm Laboratory: for any algorithm show formula, definition,
         assumptions, input, frontier/internal state, step-by-step execution
         from REAL step events, numeric values, output, complexity,
         limitations, source link, tests, references. ALGORITHM DUEL view.
Slice 5  Experiment Manager + Analytics: configure/run experiments, Plotly
         comparisons with CIs, ablation tables, effect sizes.
Slice 6  Agent Laboratory, Learning Studio (curves, Q-heatmaps, policy arrows),
         Memory Explorer, Reasoning Inspector (proof trees, belief tables),
         Multi-Agent Arena.
Slice 7  Mathematics Lab (derivations with adjustable parameters),
         Documentation, Research Notebook, Command Center.
Slice 8  TIME MACHINE UI: scrubber, fork-here, side-by-side divergent runs.

DESIGN DIRECTION: a research instrument, not an admin dashboard — dense,
legible, dark theme, monospace for numerics, consistent color semantics
(truth vs belief vs memory), keyboard shortcuts for stepping. Use the
frontend-design guidance if your IDE supports skills.

VERIFICATION: TypeScript strict; component tests for data adapters; Playwright
smoke tests for each slice; every displayed trace/metric comes from the API (a
test greps the codebase for hardcoded demo data).

INTEGRATION HOOK: none beyond API; keep UI feature flags per module so slices
light up as engine modules land.
```

---

### M21 — Knowledge Base & Research Report

**Blueprint refs:** §17, §18

```text
MODULE M21 — Knowledge Base and Research Output
PREREQUISITES: all implemented modules (iterative).

KNOWLEDGE BASE (MkDocs)
Per-algorithm record with exactly: Theory · Mathematics · Algorithm (pseudocode +
complexity) · Implementation · Verification · Experiments · Limitations ·
References. Generate API docs from docstrings. A docs-lint script fails CI if an
algorithm registered in the registry lacks its record.

CITATION SAFETY: cite only sources you can name precisely (textbooks from the
blueprint: Russell & Norvig; Sutton & Barto; Bishop; Goodfellow et al.; Weiss).
Do not invent papers or page numbers; mark anything unverified as
"TO VERIFY".

RESEARCH REPORT (research/findings/)
Template: research question, hypothesis, method, environments, seeds, results
(real, from M18 records), statistical analysis, ablations, negative results,
threats to validity, limitations, future work. No novelty claims without a
literature review and reproducible evidence. Failed experiments are evidence.

VERIFICATION: every number in the report is traceable to an experiment record id;
a script re-generates all tables/figures from stored records.
```

---

### M22 — Showpiece Scenarios (Gate 10)

**This is where the project becomes fascinating.** Each scenario is a config preset + experiment + UI view built entirely from real engine output.

```text
MODULE M22 — Showpiece Scenarios
PREREQUISITES: the modules listed per scenario.

Build each as: configs/showpieces/<name>.yaml + scripts/run_showpiece_<name>.py +
a scenario page in docs and in the Studio. All outputs from real runs.

1 GHOST IN THE FOG  (M03, M05, M08, M10, M17)
  A hidden predator stalks a foraging colony. Agents see only a small radius.
  Right panel: belief heatmap; left: truth. Show belief entropy over time, the
  moment the filter "locks on", and the symbolic safety shield vetoing a lethal
  move. Compare Bayesian vs reflex survival with CIs.

2 ALGORITHM DUEL  (M06)
  BFS, UCS, Greedy, A*, Weighted A* race on one map. Live frontiers, node counts,
  path costs. Include a map where Greedy is fooled and one where Manhattan
  scaling matters.

3 TRAGEDY OF THE COMMONS  (M13, M15)
  N learners share a regrowing resource. Show collapse under individual reward,
  and what changes with shared/mixed reward, communication, or a punishment
  mechanism. Report Gini, depletion time, survival.

4 EVOLUTION OF TRIBES  (M16, M12)
  Hundreds of generations; lineage tree; speciation; behavioural clusters
  (scouts/hoarders/raiders) via K-Means on behaviour features; boom-bust
  cycles. Evaluate champions on unseen worlds.

5 ABLATION ARENA  (M17, M18)
  The hybrid agent with components removed, on held-out worlds, with effect
  sizes and a plain-language verdict that is allowed to be "no difference".

6 TIME MACHINE  (M02, M05, M20)
  Pick a catastrophe tick (e.g. a famine death), rewind 30 ticks, change one
  decision, fork, and display divergence metrics between the two histories.

7 THE CHANGING WORLD  (M13, M14, M16, M17)
  Non-stationary environment (seasons, droughts, moving water). Measure
  adaptation speed, forgetting vs transfer, and which architectures recover.

ACCEPTANCE (Gate 10)
- Every scenario reproduces from a single command and a seed.
- Scenario pages show REAL traces/metrics; none hardcoded.
- A reproducible benchmark suite and the research report (M21) exist.
- You can open any module, explain its mathematics, run its experiment, and
  defend the result without asking an LLM.
```

---

## Appendix A — Independent verification oracles (the "no blind trust" table)

| Module | Algorithm | Independent oracle / check |
|---|---|---|
| M01 | World transitions | Property-based invariants; state hashing across seeds |
| M02 | Determinism | Hash-per-tick equality; snapshot/restore equivalence |
| M06 | A\*, UCS, BFS | `scipy.sparse.csgraph.dijkstra`; brute-force on tiny graphs |
| M07 | Alpha-Beta | Equals Minimax on random trees; known game values |
| M07 | Nash | Equilibrium inequality checker; analytic 2×2 solutions |
| M08 | Resolution / chaining | Truth-table entailment |
| M09 | CSP / STRIPS | Independent constraint checker; independent plan validator |
| M10 | Variable elimination | Full enumeration; `scipy.stats` for distributions |
| M10 | HMM | Brute-force sum over hidden sequences |
| M11 | Value/Policy iteration | Linear solve of the Bellman equations; hand-solved MDPs |
| M12 | Classical ML | scikit-learn on identical data (tests only) |
| M13 | Q-learning / SARSA | Exact Q\* from M11; CliffWalking behaviour |
| M14 | Backprop | Numerical gradient check; PyTorch twin |
| M14 | DQN target | Terminal target equals reward |
| M16 | GA / DE / PSO | OneMax, sphere, Rastrigin benchmarks |
| M17 | MCTS | Known bandit tree; small-game minimax value |
| M18 | Statistics | `scipy.stats` oracle; bootstrap coverage test |

---

## Appendix B — Reusable prompt templates

### B.1 Plan prompt (Step A)

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

### B.2 Verification prompt (Step D)

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

### B.3 Adversarial review prompt

```text
Act as a skeptical reviewer of {{MODULE_ID}}. Find: mathematical errors, off-by-
one errors, leaked ground truth (agent sees data it shouldn't), hidden
nondeterminism (global RNG, dict/set ordering, time-dependent code), places
where an agent can mutate the environment, hardcoded results, silent
exception swallowing, and claims in docs that the code does not support.
Produce a ranked list with file/line references and proposed fixes. Do not
fix anything yet.
```

### B.4 Integration prompt (Step E)

```text
Integrate {{MODULE_ID}} into the existing system ONLY through the contracts in
docs/architecture/data_contracts.md. Register algorithms/agents in their
registries. Add an end-to-end integration test running a seeded simulation with
the new component, assert determinism, run the FULL regression suite, and update
docs (module boundaries, roadmap status, ADRs if a decision changed). Report any
contract changes explicitly — contract changes require my approval.
```

### B.5 Research prompt (Step F)

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

## Appendix C — Gate checklist

| Gate | Pass condition | Tag |
|---|---|---|
| 0 | `make check` green on clean clone; docs + ADRs real | `gate-M00` |
| 1 | Deterministic, replayable, forkable multi-agent sim; viewer; agents can't mutate world | `gate-1` |
| 2 | Search + adversarial verified against oracles; Algorithm Duel works | `gate-2` |
| 3 | Logic, CSP, planning verified; provenance traces | `gate-3` |
| 4 | Bayes/HMM/belief filter verified; MDP exact solutions exported | `gate-4` |
| 5 | ML models match sklearn, beat baselines, leakage-safe | `gate-5` |
| 6 | RL converges to exact Q\* on small MDPs; eval seeds separate | `gate-6` |
| 7 | Gradients numerically verified; NumPy ≈ PyTorch; DQN works | `gate-7` |
| 8 | Multi-agent cooperation/competition measured; non-stationarity handled | `gate-8` |
| 9 | Hybrid ablations with statistics on held-out worlds | `gate-9` |
| 10 | Showpieces reproduce from one command; research report complete | `gate-10` |

**Advance only when the gate is green. A failed experiment is useful evidence — record it, don't hide it.**

---

## Appendix D — Suggested first week-of-work order (completion-based, not calendar)

1. M00 → M01 → M02 (stop and personally read the tick-order ADR and the conflict-resolution policy).
2. M03 → M05 (get the viewer and decision traces running as early as possible — motivation matters).
3. M18 skeleton (just config → runs → CSV + basic stats).
4. M06 (A\* + Algorithm Duel is your first "wow" moment).
5. M10 belief filter + split screen (second "wow": Ghost in the Fog).
6. Then follow the dependency map in §0.2.
