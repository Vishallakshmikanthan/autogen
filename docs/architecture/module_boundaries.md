# Architecture: Module Boundaries & Import Rules

> **Standard:** Strict Layered Architecture  
> **Source Ref:** AGENTS.md §1, Blueprint §2, §9.1

---

## 1. Architectural Layers & Separation

The AUTOGEN codebase is strictly divided into distinct vertical layers and horizontal packages. No module may bypass its declared boundary.

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Applications (apps/api, apps/web)                  │
│ Can import: packages/autogen/*                              │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Experiments & Analytics (experiments, analytics)   │
│ Can import: Layer 3, Layer 2, Layer 1, Layer 0              │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Agents & Intelligence (agents, intelligence)       │
│ Can import: Layer 2 (Memory), Layer 1 (Contracts), Layer 0  │
│ FORBIDDEN to import: Simulation Engine, Apps, Visualization │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Memory & Knowledge Representation (memory)         │
│ Can import: Layer 1 (Contracts), Layer 0 (Core)             │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Simulation Engine & Physics (simulation)           │
│ Can import: Layer 0 (Core)                                  │
│ FORBIDDEN to import: Agents, Intelligence algorithms        │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│ Layer 0: Core Foundation (autogen.core)                     │
│ Self-contained: types, logging, rng, config                 │
│ FORBIDDEN to import: any higher layer                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Permitted & Prohibited Dependencies Matrix

| Origin Module | Allowed Imports | Strictly Prohibited Imports | Rationale |
| :--- | :--- | :--- | :--- |
| `autogen.core` | Standard library, NumPy, Pydantic, PyYAML | Any other `autogen.*` package | Core foundation must remain a zero-dependency substrate. |
| `autogen.simulation` | `autogen.core` | `autogen.agents.*`, `autogen.intelligence.*`, `apps.*` | Environment cannot depend on agent decision code; environment exists independently. |
| `autogen.agents` | `autogen.core`, `autogen.memory.*` | `autogen.simulation.engine.*`, `apps.*`, `autogen.visualization.*` | Agents must never have access to ground-truth simulation internals or rendering code. |
| `autogen.intelligence` | `autogen.core`, NumPy, SciPy | `autogen.simulation.*`, `autogen.visualization.*`, `apps.*` | Intelligence algorithms (e.g. A*, Q-learning, MCTS) are pure domain-agnostic mathematical engines. |
| `autogen.memory` | `autogen.core`, NumPy | `autogen.simulation.*`, `apps.*` | Memory structures are agent-internal cognitive assets. |
| `autogen.visualization`| `autogen.core` | `autogen.intelligence.*`, `autogen.agents.*` | Visualization formats traces and states; it does not alter or drive intelligence logic. |
| `apps.api` | Any `autogen.*` package | — | API hosts and exposes services to frontend clients. |

---

## 3. Ground Truth Isolation Rules

1. **No Ground Truth Leakage:**
   - An agent's decision logic receives an `Observation` object via its `perceive(obs: Observation)` method.
   - Agents are forbidden from receiving references to `WorldState`, `Environment`, or simulation engine pointers.
2. **Action Arbitration Safety:**
   - Agent methods (`act()`, `plan()`) return pure immutable data objects (e.g. `Action(type=ActionType.MOVE, direction=Direction.NORTH)`).
   - Agents cannot directly adjust their coordinates, replenish their energy, or modify terrain cells.
3. **Purity of Algorithmic Modules:**
   - Modules in `autogen.intelligence` must operate over standard mathematical abstractions (graphs, cost functions, transition matrices, reward functions, tensor arrays).
   - They must never import simulation-specific objects (e.g., `GridCell` or `Inventory`). Problem-specific adapters translate simulation entities into mathematical state spaces.
