"""Adversarial and game-theoretic autonomous agents (SYS-10, Module M07)."""

from autogen.agents.adversarial.predator_agent import PredatorAgent
from autogen.agents.adversarial.prey_agent import PreyAgent
from autogen.agents.adversarial.stochastic_opponent import StochasticOpponent
from autogen.agents.adversarial.strategic_forager import StrategicForager

__all__ = [
    "PredatorAgent",
    "PreyAgent",
    "StochasticOpponent",
    "StrategicForager",
]

