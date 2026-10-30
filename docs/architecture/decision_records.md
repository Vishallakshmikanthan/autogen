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

