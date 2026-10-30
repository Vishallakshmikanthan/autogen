"""Agent Registry and Factory Pattern implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.baseline.model_based_reflex_agent import ModelBasedReflexAgent
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.goal_based.goal_based_agent import GoalBasedAgent
from autogen.agents.memory_augmented_agent import MemoryAugmentedAgent
from autogen.memory.contracts import MemoryConfig
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position


class AgentConfig(BaseModel):
    """Declarative specification for configuring and instantiating an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_type: str
    agent_id: str
    seed: int = 42
    capabilities: AgentCapabilities | None = None
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    custom_params: dict[str, Any] = Field(default_factory=dict)


AgentFactory = Callable[[AgentConfig], BaseAgent]


class AgentRegistry:
    """Thread-safe registry of agent architectures and builders."""

    _registry: dict[str, type[BaseAgent] | AgentFactory] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory_or_cls: type[BaseAgent] | AgentFactory,
    ) -> None:
        """Registers an agent type or constructor factory under a canonical name."""
        canonical_name = name.strip().lower()
        if canonical_name in cls._registry:
            raise ValueError(f"Agent type '{canonical_name}' is already registered.")
        cls._registry[canonical_name] = factory_or_cls

    @classmethod
    def get(cls, name: str) -> type[BaseAgent] | AgentFactory:
        """Retrieves the factory or class registered under name."""
        canonical_name = name.strip().lower()
        if canonical_name not in cls._registry:
            raise KeyError(
                f"Agent type '{canonical_name}' not found. Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[canonical_name]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Lists all registered agent names."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clears all registered agents (primarily for test teardown)."""
        cls._registry.clear()

    @classmethod
    def create(cls, config: AgentConfig) -> BaseAgent:
        """Instantiates an agent from its declarative configuration."""
        factory_or_cls = cls.get(config.agent_type)
        if isinstance(factory_or_cls, type) and issubclass(factory_or_cls, BaseAgent):
            # Inspect constructor arguments
            if factory_or_cls is RandomAgent:
                return factory_or_cls(
                    agent_id=config.agent_id,
                    seed=config.seed,
                    capabilities=config.capabilities,
                )
            return factory_or_cls(
                agent_id=config.agent_id,
                capabilities=config.capabilities,
            )
        # Callable factory
        factory_fn: AgentFactory = factory_or_cls  # type: ignore[assignment]
        return factory_fn(config)


# Register canonical baseline agents
AgentRegistry.register("random", RandomAgent)
AgentRegistry.register("simple_reflex", SimpleReflexAgent)
AgentRegistry.register("model_based_reflex", ModelBasedReflexAgent)


def _create_memory_augmented_agent(config: AgentConfig) -> BaseAgent:
    inner_type = config.custom_params.get("inner_agent_type", "simple_reflex")
    inner_cfg = AgentConfig(
        agent_type=inner_type,
        agent_id=config.agent_id,
        seed=config.seed,
    )
    inner_agent = AgentRegistry.create(inner_cfg)
    mem_dict = config.custom_params.get("memory_config", {})
    mem_config = MemoryConfig(**mem_dict) if isinstance(mem_dict, dict) else MemoryConfig()
    return MemoryAugmentedAgent(inner_agent=inner_agent, config=mem_config)


AgentRegistry.register("memory_augmented", _create_memory_augmented_agent)


def _create_goal_based_agent(config: AgentConfig) -> BaseAgent:
    fixed_goal_data = config.custom_params.get("fixed_goal")
    fixed_goal = Position(*fixed_goal_data) if fixed_goal_data else None
    return GoalBasedAgent(
        agent_id=config.agent_id,
        capabilities=config.capabilities,
        fixed_goal=fixed_goal,
    )


AgentRegistry.register("goal_based", _create_goal_based_agent)
