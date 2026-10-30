"""Agent architectural contracts, capability declarations, and abstract base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation


class AgentCapabilities(BaseModel):
    """Explicit declaration of cognitive subsystems supported by an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory: bool = False
    belief: bool = False
    planning: bool = False
    learning: bool = False


class BaseAgent(ABC):
    """Abstract base class for all autonomous agents in AUTOGEN."""

    def __init__(self, agent_id: str, capabilities: AgentCapabilities | None = None) -> None:
        self.agent_id = agent_id
        self.capabilities = capabilities or AgentCapabilities()
        self._last_observation: Observation | None = None

    @abstractmethod
    def observe(self, observation: Observation) -> None:
        """Phase 1: Ingests the sensory observation from the Perception Engine."""
        self._last_observation = observation

    @abstractmethod
    def update_belief(self) -> None:
        """Phase 2: Updates internal memory, belief state, or Bayesian filter."""
        ...

    @abstractmethod
    def decide(self) -> tuple[Action, DecisionTrace]:
        """Phase 3: Deliberates and selects an action, producing a verified DecisionTrace."""
        ...

    @abstractmethod
    def learn(self, reward: float, terminal: bool) -> None:
        """Phase 7: Updates policy or value weights based on observed outcome and reward."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Resets agent internal state for a new episode or simulation replay."""
        self._last_observation = None
