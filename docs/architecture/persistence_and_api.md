# Persistence, FastAPI, WebSocket Stream & Worker Architecture (SYS-18, Module M19)

## 1. Architectural Overview

The AUTOGEN Research Platform follows a modular monolith architecture. A critical invariant of this design is that **HTTP and WebSocket handlers NEVER execute simulations**. All simulation stepping, physics resolution, and experiment suite runs are owned exclusively by decoupled simulation worker processes communicating across non-blocking job queues.

```
┌─────────────────────────────────────────────────────────────┐
│                   React Frontend / Web UI                   │
│             (Viewer Only — Never source of truth)           │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                     │
│  - REST Endpoints (/environments, /agents, /simulations)    │
│  - WebSocket Hub (/api/v1/simulations/{id}/stream)          │
│  - Pydantic v2 validation & response envelopes              │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼ Command Queue                 ▼ Queries
┌──────────────────────────────┐┌──────────────────────────────┐
│   Simulation Worker Loop     ││  SQLite Persistence (WAL)    │
│  - Decoupled process/thread  ││  - Repository pattern        │
│  - Owns SimulationEngine     ││  - Periodic snapshots        │
│  - Delta-encodes frames      ││  - M18 Experiment records    │
│  - Manages playback speed    ││  - Event log telemetry       │
└──────────────┬───────────────┘└──────────────────────────────┘
               │ Event Queue (Delta frames)
               └───────────────────────────────► Forwarded to WS
```

---

## 2. Database Schema (PostgreSQL-Compatible SQLite DDL)

| Table | Purpose |
| :--- | :--- |
| `users` | Optional user accounts and credentials |
| `environments` | Environment configurations (dimensions, baseline parameters) |
| `environment_versions` | Immutable versioned environment definitions with SHA-256 config hashes |
| `agents` | Agent architectures and canonical names |
| `agent_versions` | Versioned agent hyperparameter and policy specifications |
| `algorithms` | Registered algorithm metadata, capabilities, and version tags |
| `simulations` | Live and archived simulation sessions with status and latest state hash |
| `simulation_events` | Append-only event log (collisions, deaths, food collections) |
| `agent_states` | Selected state snapshots saved at configurable cadence (default: every 25 ticks) |
| `experiments` | Declarative experiment suite definitions |
| `experiment_runs` | Individual episode run records with execution timings and seed metadata |
| `metrics` | Numerical metrics tracked across steps |
| `model_checkpoints` | Neural and tabular model checkpoint references on disk |
| `research_notes` | Hypotheses, findings, and Markdown research memos |

---

## 3. WebSocket Streaming & Delta-Encoding

### 3.1 Delta-Compression Strategy
Broadcasting full `ViewerState` frames (containing all grid cells) at 30–60 FPS generates unnecessary wire overhead. The `DeltaEncoder`:
1. Transmits an initial full **Keyframe** (`is_keyframe=True`).
2. Subsequent ticks calculate differential changes:
   - Updated agents (position, energy, carried food, health changes).
   - Modified cells (cells whose food resource or terrain changed).
   - Deliberation traces and dynamic overlay markers.
3. Automatically transmits periodic keyframes every 20 ticks or upon client resynchronization.

### 3.2 Backpressure & Frame-Dropping
If a client browser experiences UI thread lag or network throttling:
- The streaming worker maintains an event queue.
- If queue depth exceeds the threshold ($> 10$ frames), intermediate delta frames are dropped immediately without blocking the simulation engine.
- A fresh full keyframe is emitted upon buffer clearance to resynchronize the client display.

---

## 4. REST API Specification

### Environments
- `POST /api/v1/environments` — Register new environment configuration.
- `GET /api/v1/environments` — List all environments with version tags.
- `GET /api/v1/environments/{id}` — Retrieve environment configuration and hash.

### Agents
- `POST /api/v1/agents` — Register agent architecture.
- `GET /api/v1/agents` — List registered agents.
- `GET /api/v1/agents/{id}` — Retrieve agent policy version.

### Simulations
- `POST /api/v1/simulations` — Spawn a new simulation session.
- `GET /api/v1/simulations/{id}` — Query status, step progress, and state hash.
- `POST /api/v1/simulations/{id}/start` — Dispatch start/resume execution command.
- `POST /api/v1/simulations/{id}/pause` — Dispatch pause command.
- `POST /api/v1/simulations/{id}/step` — Advance simulation by exactly one tick.
- `POST /api/v1/simulations/{id}/reset` — Reset world to tick 0.
- `GET /api/v1/simulations/{id}/state` — Fetch latest snapshot.
- `GET /api/v1/simulations/{id}/metrics` — Fetch tracked numerical metrics.

### Experiments
- `POST /api/v1/experiments` — Register declarative experiment YAML/JSON.
- `GET /api/v1/experiments/{id}` — Retrieve experiment definition.
- `POST /api/v1/experiments/{id}/run` — Dispatch experiment suite across seeds.
- `GET /api/v1/experiments/{id}/results` — Retrieve runs and aggregated metrics.

---

## 5. Verification & Test Oracles

1. **REST Endpoints:** Tested via `fastapi.testclient.TestClient` validating HTTP 201, 200, 404, and 422 schemas.
2. **Delta Encoding Exactness:** `DeltaEncoder.decode_and_apply(prev_state, delta)` must produce a `ViewerState` bit-identical to the true target state.
3. **Simulation Determinism Invariant:** Running an identical seed through the API produces the exact same `state_hash` as running headless via `SimulationEngine`.
4. **WebSocket Backpressure:** Verified by simulating slow consumers and asserting that frame buffers drop gracefully and recover via keyframe without stalling the simulation.

