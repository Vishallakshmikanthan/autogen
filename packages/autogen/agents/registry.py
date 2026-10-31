"""Agent Registry and Factory Pattern implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.agents.adversarial.predator_agent import PredatorAgent
from autogen.agents.adversarial.prey_agent import PreyAgent
from autogen.agents.adversarial.stochastic_opponent import StochasticOpponent
from autogen.agents.adversarial.strategic_forager import StrategicForager
from autogen.agents.base.contracts import AgentCapabilities, BaseAgent
from autogen.agents.decision.mdp_agent import MDPAgent
from autogen.agents.decision.utility_agent import UtilityAgent
from autogen.agents.planning.planning_agent import PlanningAgent
from autogen.agents.probabilistic.bayesian_hazard_agent import BayesianHazardAgent
from autogen.agents.symbolic.logic_agent import LogicAgent
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
            if issubclass(factory_or_cls, RandomAgent):
                return RandomAgent(
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


def _create_predator_agent(config: AgentConfig) -> BaseAgent:
    return PredatorAgent(
        agent_id=config.agent_id,
        capabilities=config.capabilities,
        max_lookahead_depth=config.custom_params.get("max_lookahead_depth", 3),
        target_prey_id=config.custom_params.get("target_prey_id"),
    )


AgentRegistry.register("predator", _create_predator_agent)


def _create_prey_agent(config: AgentConfig) -> BaseAgent:
    return PreyAgent(
        agent_id=config.agent_id,
        capabilities=config.capabilities,
        max_lookahead_depth=config.custom_params.get("max_lookahead_depth", 3),
        use_expectimax=config.custom_params.get("use_expectimax", False),
        predator_id=config.custom_params.get("predator_id"),
    )


AgentRegistry.register("prey", _create_prey_agent)


def _create_stochastic_opponent(config: AgentConfig) -> BaseAgent:
    return StochasticOpponent(
        agent_id=config.agent_id,
        seed=config.seed,
        capabilities=config.capabilities,
        action_weights=config.custom_params.get("action_weights"),
    )


AgentRegistry.register("stochastic_opponent", _create_stochastic_opponent)


def _create_strategic_forager(config: AgentConfig) -> BaseAgent:
    return StrategicForager(
        agent_id=config.agent_id,
        capabilities=config.capabilities,
        game_type=config.custom_params.get("game_type", "hawk_dove"),
        strategy_mode=config.custom_params.get("strategy_mode", "mixed_nash"),
        seed=config.seed,
    )


AgentRegistry.register("strategic_forager", _create_strategic_forager)


def _create_logic_agent(config: AgentConfig) -> BaseAgent:
    return LogicAgent(
        agent_id=config.agent_id,
        capabilities=config.capabilities,
        energy_safety_threshold=config.custom_params.get("energy_safety_threshold", 25.0),
    )


AgentRegistry.register("logic_agent", _create_logic_agent)


def _create_planning_agent(config: AgentConfig) -> BaseAgent:
    return PlanningAgent(
        agent_id=config.agent_id,
        planning_problem=config.custom_params.get("planning_problem"),
        capabilities=config.capabilities,
    )


AgentRegistry.register("planning_agent", _create_planning_agent)


def _create_bayesian_hazard_agent(config: AgentConfig) -> BaseAgent:
    return BayesianHazardAgent(
        agent_id=config.agent_id,
        grid_width=config.custom_params.get("grid_width", 15),
        grid_height=config.custom_params.get("grid_height", 15),
        capabilities=config.capabilities,
        hazard_penalty=config.custom_params.get("hazard_penalty", 100.0),
        food_reward=config.custom_params.get("food_reward", 10.0),
    )


AgentRegistry.register("bayesian_hazard_agent", _create_bayesian_hazard_agent)


def _create_mdp_agent(config: AgentConfig) -> BaseAgent:
    return MDPAgent(
        agent_id=config.agent_id,
        solution=config.custom_params.get("solution"),
        mdp=config.custom_params.get("mdp"),
        capabilities=config.capabilities,
    )


AgentRegistry.register("mdp_agent", _create_mdp_agent)


def _create_utility_agent(config: AgentConfig) -> BaseAgent:
    return UtilityAgent(
        agent_id=config.agent_id,
        grid_width=config.custom_params.get("grid_width", 15),
        grid_height=config.custom_params.get("grid_height", 15),
        weights=config.custom_params.get("weights"),
        capabilities=config.capabilities,
    )


AgentRegistry.register("utility_agent", _create_utility_agent)


def _create_learned_perception_agent(config: AgentConfig) -> BaseAgent:
    from autogen.agents.ml.learned_perception_agent import LearnedPerceptionAgent
    from autogen.intelligence.ml.applications import LearnedPerceptionComponent

    comp = config.custom_params.get("perception_component")
    if comp is None:
        comp = LearnedPerceptionComponent()
    return LearnedPerceptionAgent(
        agent_id=config.agent_id,
        perception_component=comp,
        capabilities=config.capabilities,
        danger_threshold=config.custom_params.get("danger_threshold", 0.40),
    )


AgentRegistry.register("learned_perception_agent", _create_learned_perception_agent)


def _create_q_learning_agent(config: AgentConfig) -> BaseAgent:
    from autogen.agents.rl.q_learning_agent import QLearningAgent
    from autogen.intelligence.rl.tabular import QLearning

    ql = config.custom_params.get("q_learning")
    if ql is None:
        ql = QLearning(
            alpha=config.custom_params.get("alpha", 0.1),
            gamma=config.custom_params.get("gamma", 0.99),
            epsilon=config.custom_params.get("epsilon", 0.1),
            seed=config.seed,
        )
    return QLearningAgent(
        agent_id=config.agent_id,
        q_learning=ql,
        capabilities=config.capabilities,
    )


AgentRegistry.register("q_learning_agent", _create_q_learning_agent)


def _create_sarsa_agent(config: AgentConfig) -> BaseAgent:
    from autogen.agents.rl.sarsa_agent import SarsaAgent
    from autogen.intelligence.rl.tabular import SARSA

    sarsa = config.custom_params.get("sarsa")
    if sarsa is None:
        sarsa = SARSA(
            alpha=config.custom_params.get("alpha", 0.1),
            gamma=config.custom_params.get("gamma", 0.99),
            epsilon=config.custom_params.get("epsilon", 0.1),
            seed=config.seed,
        )
    return SarsaAgent(
        agent_id=config.agent_id,
        sarsa=sarsa,
        capabilities=config.capabilities,
    )


AgentRegistry.register("sarsa_agent", _create_sarsa_agent)


def _create_dyna_agent(config: AgentConfig) -> BaseAgent:
    from autogen.agents.rl.dyna_agent import DynaAgent
    from autogen.intelligence.rl.dyna import DynaQ

    dyna = config.custom_params.get("dyna_q")
    if dyna is None:
        dyna = DynaQ(
            alpha=config.custom_params.get("alpha", 0.1),
            gamma=config.custom_params.get("gamma", 0.99),
            epsilon=config.custom_params.get("epsilon", 0.1),
            n_planning_steps=config.custom_params.get("n_planning_steps", 10),
            seed=config.seed,
        )
    return DynaAgent(
        agent_id=config.agent_id,
        dyna_q=dyna,
        capabilities=config.capabilities,
    )


AgentRegistry.register("dyna_agent", _create_dyna_agent)


def _create_dqn_agent(config: AgentConfig) -> BaseAgent:
    from autogen.agents.neural.dqn_agent import DQNAgent
    from autogen.intelligence.neural.dqn import DeepQNetwork

    dqn = config.custom_params.get("dqn")
    if dqn is None:
        dqn = DeepQNetwork(
            state_dim=config.custom_params.get("state_dim", 4),
            num_actions=config.custom_params.get("num_actions", 4),
            seed=config.seed,
        )
    return DQNAgent(
        agent_id=config.agent_id,
        dqn=dqn,
        state_dim=config.custom_params.get("state_dim", 4),
        replay_capacity=config.custom_params.get("replay_capacity", 5000),
        batch_size=config.custom_params.get("batch_size", 32),
        initial_epsilon=config.custom_params.get("initial_epsilon", 1.0),
        min_epsilon=config.custom_params.get("min_epsilon", 0.05),
        epsilon_decay=config.custom_params.get("epsilon_decay", 0.995),
        capabilities=config.capabilities,
    )


AgentRegistry.register("dqn_agent", _create_dqn_agent)


def _create_ppo_agent(config: AgentConfig) -> BaseAgent:
    from autogen.agents.neural.ppo_agent import PPOAgent
    from autogen.intelligence.neural.policy_gradient import PPO

    ppo = config.custom_params.get("ppo")
    if ppo is None:
        ppo = PPO(
            state_dim=config.custom_params.get("state_dim", 4),
            num_actions=config.custom_params.get("num_actions", 4),
            seed=config.seed,
        )
    return PPOAgent(
        agent_id=config.agent_id,
        ppo=ppo,
        state_dim=config.custom_params.get("state_dim", 4),
        update_interval=config.custom_params.get("update_interval", 32),
        capabilities=config.capabilities,
    )


AgentRegistry.register("ppo_agent", _create_ppo_agent)



