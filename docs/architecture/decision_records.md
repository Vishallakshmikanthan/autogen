# Architecture Decision Records (ADRs)

---

## ADR-001: Modular Monolith Repository Architecture

### Context
AUTOGEN AI is a comprehensive AI research and simulation laboratory encompassing discrete simulation, autonomous agents, search, probabilistic inference, machine learning, deep neural networks, and analytical tools. A microservice architecture was considered, where simulation, agents, and algorithms run as independent network services.

### Decision
We adopt a **modular monolith** within a single repository (`autogen/`).
- Subsystems are organized as cleanly partitioned packages under `packages/autogen/`.
- Communication across subsystem boundaries occurs via strictly typed Python contracts, in-memory function calls, and protocols.
- Microservices, network serialization overhead, and distributed consensus are avoided during algorithm and simulation core development.

### Consequences
- **Positive:** Zero serialization/network overhead during high-frequency simulation ticks; atomic refactoring; unified type-checking with Mypy; simplified single-command CI (`make check`).
- **Negative:** Requires strict discipline and automated linting to prevent cyclic or illegal package dependencies.

---

## ADR-002: Python Simulation Engine Owns Simulation State

### Context
Interactive simulation platforms often blur the line between client-side rendering (web browsers, game engines) and server-side state calculation. When state is distributed or owned by frontend frameworks, reproducibility, determinism, and headless benchmarking become compromised.

### Decision
**The Python simulation engine (`autogen.simulation`) exclusively owns all ground-truth simulation state.**
- The environment transitions state deterministically according to strict mathematical rules.
- Autonomous agents can only propose actions; they never mutate the world state directly.
- Frontend interfaces (React / WebSockets) and Algorithm Lab visualizations are purely passive consumers and inspectors of world state diffs and decision traces.

### Consequences
- **Positive:** Headless simulation runs at thousands of ticks per second; 100% reproducible execution and replay; mathematical verification is entirely decoupled from GUI code.
- **Negative:** Full world state snapshots must be serialized into differential updates for client consumption over WebSockets.

---

## ADR-003: From-Scratch Policy for Core AI Algorithms

### Context
Modern AI development heavily relies on off-the-shelf high-level libraries (e.g. Scikit-learn, PyTorch, Hugging Face, Ray). While rapid for production deployment, third-party libraries obscure algorithmic primitives, make internal search trees opaque, introduce hidden nondeterminism, and prevent deep educational and inspectable instrumentation.

### Decision
**All foundational AI algorithms must be implemented FROM SCRATCH using only standard library, NumPy, and SciPy.**
- This applies to graph search (BFS, DFS, Dijkstra, A*, IDA*), adversarial search (Minimax, Alpha-Beta, MCTS), logic engines, constraint satisfaction, exact Bayesian inference, MDP solvers (Value Iteration, Policy Iteration), tabular RL, neural autograd graphs, and evolutionary algorithms.
- Third-party libraries (e.g. `scikit-learn`, `torch`) are strictly permitted **only as test oracles and baselines** in test suites, never as the implementation itself.

### Consequences
- **Positive:** Complete visibility into every computation step; step-traceable generator hooks for algorithm visualizers; absolute educational and scientific transparency; zero external model opacity.
- **Negative:** Requires rigorous mathematical derivation, careful performance optimization with vectorized NumPy, and exhaustive verification suites.

---

## ADR-004: Seeded RNG Streams via NumPy SeedSequence

### Context
Stochastic algorithms (e.g. environment resource generation, random exploration, stochastic search, genetic mutations) require pseudo-random number generation. Using global RNG (`random.seed()`, `numpy.random.seed()`) introduces hidden state coupling, race conditions, non-reproducibility across test ordering, and data leakage between training and evaluation splits.

### Decision
**Every random choice must flow from an explicit seed through named RNG streams spawned via `numpy.random.SeedSequence`.**
- A centralized `SeedManager` maintains a root `SeedSequence`.
- Subsystems spawn distinct child sequences for each domain: `env`, `agent_i`, `train_run_k`, `eval_run_k`.
- Global random calls (`random.choice()`, `np.random.rand()`) are strictly forbidden and caught in code review.

### Consequences
- **Positive:** Guarantee of bit-identical replayability across different runs given the same master seed; complete mathematical independence between training environments and evaluation benchmarks.
- **Negative:** Explicit passing of PRNG streams or `SeedManager` references to stochastic components.

---

## ADR-005: Discrete-Time 9-Step Simulation Tick Execution Order

### Context
In multi-agent discrete-time simulation, changing the order of observation, decision request, conflict arbitration, and transition fundamentally alters outcomes (e.g. who observes food first, whether actions are evaluated on current or intermediate state).

### Decision
We standardize an immutable **9-phase simulation tick lifecycle**:
1. **Generate observations:** Construct `Observation` for each active agent using bounded sensory filtering.
2. **Update beliefs/memory:** Agents update internal memory and probabilistic beliefs from observations.
3. **Request decisions:** Agents propose joint actions and emit verifiable `DecisionTrace` records.
4. **Validate actions:** Environment validates legality. Illegal actions are cleanly substituted with `STAY`, emitting `InvalidActionEvent`.
5. **Resolve simultaneous conflicts:** Contested movement and collection disputes are arbitrated via configured policy.
6. **Apply environment transition:** Physical world state advances deterministically to $S_{t+1}$.
7. **Compute rewards/costs:** Rewards calculated and dispatched to `agent.learn()`.
8. **Update metrics and events:** `StepCompletedEvent` emitted, actions logged to replay, snapshots saved per cadence.
9. **Publish new state:** The new immutable `WorldState` is published.

### Consequences
- **Positive:** Deterministic, reproducible behavior across all runs; explicit arbitration prevents race conditions; full step-traceability.
- **Negative:** Strict ordering must be maintained across all custom environments.

---

## ADR-006: Contested Movement and Resource Arbitration Policies

### Context
When multiple agents simultaneously attempt to move into the exact same cell or collect limited food from the same cell during a tick, the simulation engine requires deterministic arbitration rules.

### Decision
We support two explicit, configurable conflict resolution policies:
1. `ConflictPolicy.CANCEL_ALL`: When two or more agents simultaneously attempt to move into the same cell, all moving claimants are blocked and assigned `STAY` (bouncing back).
2. `ConflictPolicy.SEEDED_PRIORITY`: Claimants are permuted via a seeded pseudo-random stream. The first claimant in the permutation successfully moves; remaining claimants are assigned `STAY`.
3. Contested food collection supports `CollectPolicy.PROPORTIONAL` (split evenly among gathering agents) and `CollectPolicy.SEEDED_PRIORITY`.

### Consequences
- **Positive:** Prevents spatial stacking or physical clipping; fully deterministic under a given seed; enables experimental analysis of different conflict dynamics.
- **Negative:** Agents must learn to anticipate or adapt to contested space collisions.
