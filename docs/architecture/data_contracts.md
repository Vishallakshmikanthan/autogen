# Architecture: Data Contracts & Protocols

> **Standard:** Strict Typed Pydantic & Protocol Specifications  
> **Source Ref:** AGENTS.md §1, Blueprint §2.2, §3

---

## 1. Foundational Contracts

All cross-module communication in AUTOGEN is mediated by immutable data classes (`Pydantic` models with `frozen=True` or `typing.Protocol`).

### 1.1 Action Contract
Agents propose actions; the environment executes them.
```python
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class ActionType(str, Enum):
    IDLE = "idle"
    MOVE = "move"
    GATHER = "gather"
    COMMUNICATE = "communicate"
    ACTIVATE = "activate"

class Action(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    action_type: ActionType
    parameters: dict[str, int | float | str | bool] = Field(default_factory=dict)
```

### 1.2 Observation Contract
Observations are partial representations of the world state filtered through perception sensors.
```python
class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    step_id: int
    local_grid: list[list[dict[str, int | float | str]]]
    local_agents: list[dict[str, str | int | float]]
    internal_sensors: dict[str, float]  # Energy, health, temperature
```

### 1.3 Decision Trace Contract
Every algorithm deliberation produces a verifiable audit trail.
```python
class DecisionTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    step_id: int
    timestamp_ns: int
    algorithm_name: str
    decision_type: str
    metrics: dict[str, float | int | str]  # nodes expanded, depth, runtime_ms
    rationale: dict[str, Any]             # chosen branch, utility score, probabilities
```

---

## 2. Structural Protocols

### 2.1 Agent Protocol
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class AgentProtocol(Protocol):
    agent_id: str

    def perceive(self, observation: Observation) -> None:
        """Updates internal belief and state based on sensory input."""
        ...

    def act(self) -> tuple[Action, DecisionTrace]:
        """Deliberates and returns a chosen action paired with its decision trace."""
        ...

    def reset(self) -> None:
        """Resets internal state for a new episode."""
        ...
```

### 2.2 Environment Protocol
```python
@runtime_checkable
class EnvironmentProtocol(Protocol):
    def step(self, actions: list[Action]) -> tuple[dict[str, Observation], dict[str, float], bool, dict[str, Any]]:
        """Resolves actions, transitions world state, and returns observations, rewards, done, info."""
        ...

    def reset(self, seed: int | None = None) -> dict[str, Observation]:
        """Resets environment to initial state using deterministic PRNG."""
        ...
```

---

## 3. Serialization Standards

1. **JSON Schema Compliance:** All models must serialize cleanly to and from JSON using standard Pydantic schemas.
2. **Replay Persistence:** Timelines of simulation runs serialize sequences of `(step_id, WorldStateDiff, list[Action], list[DecisionTrace])`.
3. **No Dynamic Class Pickling:** Serialization must use structured JSON/YAML or Arrow/Parquet tables for metrics, never raw Python pickles, ensuring long-term replay compatibility across versions.

