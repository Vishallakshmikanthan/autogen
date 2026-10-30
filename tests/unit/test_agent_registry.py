"""Unit tests for AgentRegistry and Factory Pattern."""

import pytest

from autogen.agents.base.contracts import BaseAgent
from autogen.agents.baseline.model_based_reflex_agent import ModelBasedReflexAgent
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.core.types import DecisionTrace
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType


class CustomTestAgent(BaseAgent):
    def observe(self, observation: Observation) -> None:
        super().observe(observation)

    def update_belief(self) -> None:
        pass

    def decide(self) -> tuple[Action, DecisionTrace]:
        return Action(agent_id=self.agent_id, action_type=ActionType.STAY), DecisionTrace(
            agent_id=self.agent_id,
            step_id=0,
            algorithm_name="custom_stay",
            decision_type="action_selection",
        )

    def learn(self, reward: float, terminal: bool) -> None:
        pass

    def reset(self) -> None:
        super().reset()


def test_agent_registry_pre_registered_canonical_agents():
    registered = AgentRegistry.list_registered()
    assert "random" in registered
    assert "simple_reflex" in registered
    assert "model_based_reflex" in registered


def test_agent_factory_instantiation():
    cfg_rand = AgentConfig(agent_type="random", agent_id="r1", seed=99)
    agent_rand = AgentRegistry.create(cfg_rand)
    assert isinstance(agent_rand, RandomAgent)
    assert agent_rand.agent_id == "r1"
    assert agent_rand.seed == 99

    cfg_refl = AgentConfig(agent_type="simple_reflex", agent_id="s1")
    agent_refl = AgentRegistry.create(cfg_refl)
    assert isinstance(agent_refl, SimpleReflexAgent)
    assert agent_refl.agent_id == "s1"

    cfg_mb = AgentConfig(agent_type="model_based_reflex", agent_id="m1")
    agent_mb = AgentRegistry.create(cfg_mb)
    assert isinstance(agent_mb, ModelBasedReflexAgent)
    assert agent_mb.capabilities.memory is True


def test_custom_agent_registration_and_errors():
    AgentRegistry.register("custom_test", CustomTestAgent)
    assert "custom_test" in AgentRegistry.list_registered()

    agent = AgentRegistry.create(AgentConfig(agent_type="custom_test", agent_id="c1"))
    assert isinstance(agent, CustomTestAgent)

    # Duplicate registration raises ValueError
    with pytest.raises(ValueError, match="already registered"):
        AgentRegistry.register("custom_test", CustomTestAgent)

    # Unknown agent raises KeyError
    with pytest.raises(KeyError, match="not found"):
        AgentRegistry.create(AgentConfig(agent_type="non_existent", agent_id="x"))
