# AUTOGEN AI — Modular Artificial Intelligence Research & Simulation Platform

> A configurable artificial world in which autonomous agents perceive, reason under uncertainty, plan, learn, cooperate or compete, and adapt.

---

## 1. Project Vision

AUTOGEN is a modular artificial intelligence research and simulation laboratory. It is engineered to bridge theoretical artificial intelligence, rigorous mathematical verification, and autonomous agent systems within a single, reproducible codebase.

Unlike toy frameworks or high-level API orchestrators, **AUTOGEN implements its core intelligence algorithms from scratch using NumPy and SciPy**, enabling precise formal inspection, step-traceable execution, and exact mathematical invariants.

---

## 2. Core Operational Principles

1. **Mathematical Correctness Over Speed:** Every algorithm must be accompanied by its mathematical definition, assumptions, derivations, and invariant tests.
2. **From-Scratch Algorithm Implementations:** All search algorithms, planners, probabilistic belief filters, machine learning classifiers, and reinforcement learning engines are written without third-party black-boxes. Libraries such as scikit-learn or PyTorch serve exclusively as external test oracles.
3. **Four Independent State Concepts:**
   - **Actual World State ($S_t$):** Full, unmasked ground truth possessed only by the simulation engine.
   - **Agent Internal State ($I_t$):** The agent's private computational and memory state.
   - **Agent Observation ($O_t$):** Partial, noisy, or masked perception granted to an agent through the Perception Engine.
   - **Agent Belief State ($B_t$):** The agent's internal probability distribution over world states ($P(S_t \mid O_{1:t})$).
4. **Environment Controls State Transitions:** Agents propose actions; the simulation engine validates, resolves collisions, and executes deterministic transitions. Agents never mutate the environment.
5. **Deterministic Replayability:** Every stochastic event flows strictly from explicit `np.random.SeedSequence` streams. Training and evaluation seeds are partitioned to prevent test contamination.
6. **Execution-Generated Decision Traces:** Every strategic decision generates a typed `DecisionTrace` reflecting actual internal reasoning for analysis in the Algorithm Lab.

---

## 3. Repository Architecture

AUTOGEN is structured as a modular monolith:

```text
autogen/
├── apps/
│   ├── api/                            # FastAPI backend & WebSocket simulation streams
│   └── web/                            # React + Vite Research Studio & Algorithm Lab
├── packages/
│   └── autogen/
│       ├── core/                       # Foundational types, logging, RNG, config
│       ├── simulation/                 # World state, physics, spatial grid (SYS-01, SYS-02)
│       ├── agents/                     # Agent base classes, architectures, reflexes (SYS-03, SYS-04)
│       ├── intelligence/               # Search, reasoning, planning, ML, RL (SYS-06 - SYS-15)
│       ├── memory/                     # Spatial, episodic, and semantic memory (SYS-05)
│       ├── experiments/                # Reproducible experiment runner & configs (SYS-16)
│       ├── evaluation/                 # Baselines, benchmarks, and test harnesses
│       ├── analytics/                  # Statistical metrics & verification tools (SYS-17)
│       └── visualization/              # Replay encoders & renderers
├── configs/                            # Canonical YAML scenario and simulation configurations
├── docs/                               # Mathematical foundations, architecture, and ADRs
├── scripts/                            # Operational automation scripts (cross-platform verification)
└── tests/                              # Unit, integration, mathematical, property, and regression tests
```

---

## 4. Quickstart

### Prerequisites
- Python 3.12+ (tested up to Python 3.14)
- Git

### Installation
Clone the repository and install dependencies in editable mode:
```bash
git clone https://github.com/autogen-ai/autogen.git
cd autogen
pip install -e .[dev]
```

### Verification & Tooling
AUTOGEN provides a single unified command surface for verification across Linux, macOS, and Windows:

```bash
# Cross-platform runner (recommended on Windows/PowerShell)
python scripts/check.py

# Or via Makefile (Linux / macOS / Git Bash)
make check

# Or via Windows cmd batch wrapper
make.bat check
```

Other available targets:
- `make test` / `python scripts/check.py --test`: Runs pytest suite.
- `make lint` / `python scripts/check.py --lint`: Runs Ruff and Mypy in strict mode.
- `make format` / `python scripts/check.py --format`: Automatically formats code and fixes linter warnings.

---

## 5. Development Roadmap & Documentation
For detailed architectural blueprints, mathematical specifications, and milestone exit gates:
- [Architecture System Overview](docs/architecture/system_overview.md)
- [Module Boundaries & Layer Rules](docs/architecture/module_boundaries.md)
- [Data Contracts & Types](docs/architecture/data_contracts.md)
- [Architecture Decision Records (ADRs)](docs/architecture/decision_records.md)
- [Implementation Roadmap & Milestones](docs/implementation/roadmap.md)

