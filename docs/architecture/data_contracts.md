# Architecture: Data Contracts & Protocols

> **Standard:** Strict Typed Pydantic & Protocol Specifications  
> **Source Ref:** AGENTS.md §1, Blueprint §2.2, §3, Module M02

---

## 1. Foundational Contracts

All cross-module communication in AUTOGEN is mediated by immutable data classes (`Pydantic` models with `frozen=True` or `typing.Protocol`).

### 1.1 Action Contract (Command Pattern)
Agents propose actions; the simulation engine validates and arbitrates them before applying transitions.
```python
class Action(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    action_type: ActionType  # UP, DOWN, LEFT, RIGHT, STAY, COLLECT
    parameters: dict[str, Any] = Field(default_factory=dict)
```

### 1.2 Observation Contract
Observations are partial representations of the world state filtered through perception sensors.
```python
class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    step: int = Field(ge=0)
    position: Position
    energy: float = Field(ge=0.0)
    health: float = Field(ge=0.0)
    carried_food: float = Field(ge=0.0)
    is_alive: bool = True
    visible_cells: tuple[CellState, ...]
    visible_agents: tuple[tuple[str, Position], ...]
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
    metrics: dict[str, float | int | str]
    rationale: dict[str, Any]
```

### 1.4 Agent Capabilities Contract
```python
class AgentCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory: bool = False
    belief: bool = False
    planning: bool = False
    learning: bool = False
```

---

## 2. Structural Protocols

### 2.1 BaseAgent Interface
```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, capabilities: AgentCapabilities | None = None) -> None:
        self.agent_id = agent_id
        self.capabilities = capabilities or AgentCapabilities()

    @abstractmethod
    def observe(self, observation: Observation) -> None: ...

    @abstractmethod
    def update_belief(self) -> None: ...

    @abstractmethod
    def decide(self) -> tuple[Action, DecisionTrace]: ...

    @abstractmethod
    def learn(self, reward: float, terminal: bool) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...
```

### 2.2 BaseEnvironment Protocol
```python
@runtime_checkable
class BaseEnvironment(Protocol):
    @property
    def state(self) -> WorldState: ...

    def reset(self, seed: int | None = None) -> WorldState: ...

    def get_observation(self, agent_id: str) -> Observation: ...

    def validate_action(self, agent_id: str, action: ActionType) -> bool: ...

    def apply_transitions(self, actions: dict[str, ActionType]) -> WorldState: ...
```

---

## 3. Simulation Events & Telemetry

Dispatched across the synchronous `EventBus`:

| Event Class | Attributes | Trigger |
| :--- | :--- | :--- |
| `InvalidActionEvent` | `agent_id`, `attempted_action`, `substituted_action`, `reason` | Phase 4 when an agent attempts an illegal/blocked move |
| `StepCompletedEvent` | `step`, `state_hash`, `active_agents_count` | Phase 8 at completion of each simulation tick |
| `DecisionTraceEmittedEvent` | `agent_id`, `algorithm_name`, `decision_type`, `metrics` | Phase 3 when an agent finishes deliberation |
