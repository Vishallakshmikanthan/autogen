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

---

## 4. Perception Contracts (Module M03)

Sensory filters mediating between world state and observations:

```python
class PerceptionMode(str, Enum):
    FULL = "full"
    LOCAL_WINDOW = "local_window"
    LINE_OF_SIGHT = "line_of_sight"

class NoiseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    resource_noise_std: float = Field(default=0.0, ge=0.0)
    agent_detection_probability: float = Field(default=1.0, ge=0.0, le=1.0)

class AgentSensingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    anonymous: bool = False
    sensing_radius: int = Field(default=3, ge=0)

class PerceptionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: PerceptionMode = PerceptionMode.LOCAL_WINDOW
    radius: int = Field(default=3, ge=0)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    agent_sensing: AgentSensingConfig = Field(default_factory=AgentSensingConfig)
    custom_parameters: dict[str, Any] = Field(default_factory=dict)
```

---

## 5. Agent Factory & Performance Evaluation Contracts (Module M03)

### 5.1 Agent Configuration Contract
```python
class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    agent_type: str
    agent_id: str
    seed: int = 42
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    custom_params: dict[str, Any] = Field(default_factory=dict)
```

### 5.2 Performance Evaluation Metrics Contract
```python
class AgentPerformance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    agent_id: str
    discounted_return: float
    survival_time: int
    resources_collected: float
    energy_efficiency: float
    exploration_coverage: float
    total_steps: int
    is_alive_at_end: bool
```

---

## 6. Memory Contracts (Module M04)

```python
class DecayPolicy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"

class EvictionPolicy(str, Enum):
    FIFO = "fifo"
    LRU = "lru"
    LEAST_CONFIDENT = "least_confident"

class MemoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    working_memory_capacity: int = Field(default=10, ge=1, le=100)
    episodic_memory_capacity: int = Field(default=500, ge=10, le=100_000)
    spatial_memory_capacity: int = Field(default=1000, ge=10, le=100_000)
    decay_policy: DecayPolicy = DecayPolicy.EXPONENTIAL
    decay_rate: float = Field(default=0.02, ge=0.0)
    confidence_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    eviction_policy: EvictionPolicy = EvictionPolicy.LEAST_CONFIDENT

class EpisodicTransition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    step: int = Field(ge=0)
    state_observation: Observation
    action: Action
    reward: float
    next_observation: Observation | None = None
    terminal: bool = False
    feature_vector: tuple[float, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

class SpatialMemoryCell(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    position: Position
    terrain: TerrainType
    walkable: bool
    last_seen_food: float = Field(ge=0.0)
    last_observed_tick: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)

class MemorySnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    agent_id: str
    current_step: int
    working_memory_size: int
    episodic_memory_size: int
    spatial_cells_count: int
    mean_spatial_confidence: float
    working_memory_steps: list[int]
    spatial_cells: list[SpatialMemoryCell]
```

---

## 7. Visualization & Viewer Contracts (Module M05)

```python
class ViewerCellState(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int
    y: int
    terrain: TerrainType
    food: float
    hazard: float
    walkable: bool

class ViewerAgentState(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_id: str
    x: int
    y: int
    energy: float
    health: float
    is_alive: bool
    heading: tuple[int, int] = (0, 1)

class ViewerOverlayData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fog_of_war_visible: list[tuple[int, int]] = Field(default_factory=list)
    movement_trails: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    collision_markers: list[tuple[int, int]] = Field(default_factory=list)
    resource_intensity: dict[str, float] = Field(default_factory=dict)

class ViewerTimeMachineState(BaseModel):
    model_config = ConfigDict(frozen=True)
    current_step: int
    total_steps: int
    available_snapshots: list[int]
    is_paused: bool = True
    playback_speed: float = 1.0
    can_fork: bool = True

class ViewerAgentMindCell(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int
    y: int
    terrain: TerrainType
    food: float
    confidence: float
    last_seen_step: int

class ViewerState(BaseModel):
    model_config = ConfigDict(frozen=True)
    step: int
    width: int
    height: int
    state_hash: str
    grid_cells: list[ViewerCellState]
    agents: list[ViewerAgentState]
    selected_agent_id: str | None = None
    selected_agent_trace: DecisionTrace | None = None
    agent_mind: list[ViewerAgentMindCell] | None = None
    overlays: ViewerOverlayData = Field(default_factory=ViewerOverlayData)
    time_machine: ViewerTimeMachineState | None = None
```

---

## 8. Search & Problem Solving Contracts (Module M06)

```python
class SearchProblem(Protocol[S, A]):
    @property
    def initial_state(self) -> S: ...
    def is_goal(self, state: S) -> bool: ...
    def successors(self, state: S) -> list[tuple[A, S, float]]: ...
    def heuristic(self, state: S) -> float: ...

@dataclass(slots=True)
class SearchNode(Generic[S, A]):
    state: S
    parent: SearchNode[S, A] | None = None
    action: A | None = None
    g_cost: float = 0.0
    h_cost: float = 0.0
    depth: int = 0

@dataclass(slots=True)
class SearchStepEvent(Generic[S, A]):
    algorithm: str
    step: int
    current_state: S
    frontier_states: list[S] = field(default_factory=list)
    explored_states: list[S] = field(default_factory=list)
    f_cost: float = 0.0
    g_cost: float = 0.0
    nodes_expanded: int = 0

@dataclass(slots=True)
class SearchResult(Generic[S, A]):
    success: bool
    path: list[A]
    states: list[S]
    total_cost: float
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int
    runtime_seconds: float


---

## 9. Adversarial Search & Game Theory Contracts (Module M07)

### 9.1 Adversarial Sequential Game Protocol
```python
class AdversarialGame(Protocol[S, A]):
    @property
    def initial_state(self) -> S: ...
    def player(self, state: S) -> int: ...
    def legal_actions(self, state: S) -> list[A]: ...
    def successor(self, state: S, action: A) -> S: ...
    def is_terminal(self, state: S) -> bool: ...
    def utility(self, state: S, player: int) -> float: ...
    def chance_outcomes(self, state: S, action: A) -> list[tuple[S, float]]: ...

class AdversarialAlgorithm(Protocol[S, A]):
    def search(
        self,
        game: AdversarialGame[S, A],
        state: S,
        max_depth: int,
        eval_fn: EvaluationFunction[S],
    ) -> AdversarialSearchResult[A]: ...
```

### 9.2 Normal-Form Game Contract
```python
@dataclass(slots=True)
class NormalFormGame:
    players: list[str]
    actions: list[list[str]]
    payoff_matrices: list[np.ndarray]

    @property
    def num_players(self) -> int: ...
    def payoff(self, player_idx: int, action_indices: Sequence[int]) -> float: ...
    def expected_payoff(self, player_idx: int, strategy_profile: Sequence[np.ndarray]) -> float: ...

@dataclass(slots=True)
class MixedStrategyProfile:
    probabilities: list[np.ndarray]
    expected_payoffs: list[float] = field(default_factory=list)

    @property
    def is_pure(self) -> bool: ...
    def pure_action_indices(self) -> list[int] | None: ...
```

```

---

## 10. Knowledge Representation & Logical Reasoning Contracts (Module M08, SYS-07)

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping, Protocol, Sequence, runtime_checkable

class ProofRule(str, Enum):
    AXIOM = "axiom"
    MODUS_PONENS = "modus_ponens"
    RESOLUTION = "resolution"
    AND_INTRO = "and_intro"
    AND_ELIM = "and_elim"
    OR_INTRO = "or_intro"
    UNIT_PROPAGATION = "unit_propagation"

@dataclass(frozen=True, slots=True)
class ProofStep:
    step_id: int
    conclusion: str
    rule: ProofRule
    premises: tuple[int, ...] = ()
    substitution: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ProofTree:
    query: str
    is_proved: bool
    steps: tuple[ProofStep, ...]
    root_step_id: int | None = None

    def pretty_print(self) -> str: ...

@dataclass(frozen=True, slots=True)
class Literal:
    variable: str
    is_negated: bool = False
    @property
    def complement(self) -> "Literal": ...

@dataclass(frozen=True, slots=True)
class Clause:
    literals: frozenset[Literal]
    @property
    def is_empty(self) -> bool: ...
    @property
    def is_unit(self) -> bool: ...
    @property
    def is_tautology(self) -> bool: ...

@runtime_checkable
class Formula(Protocol):
    def evaluate(self, model: Mapping[str, bool]) -> bool: ...
    def variables(self) -> frozenset[str]: ...

@runtime_checkable
class TheoremProver(Protocol):
    def entails(self, kb_formula: Formula, query: Formula) -> tuple[bool, ProofTree | None]: ...

@runtime_checkable
class SATSolver(Protocol):
    def is_satisfiable(self, clauses: frozenset[Clause]) -> tuple[bool, dict[str, bool]]: ...
```

