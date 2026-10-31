"""Data contracts and protocols for Multi-Agent Systems & Collective Intelligence (SYS-14, Module M15)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from autogen.simulation.state.types import Position


class RewardMode(str, Enum):
    """Reward allocation scheme across multi-agent collective."""

    INDIVIDUAL = "individual"  # r_i only
    SHARED = "shared"          # R_team = sum r_i
    MIXED = "mixed"            # alpha * r_i + (1 - alpha) * R_team


class Message(BaseModel):
    """Discrete communication packet transmitted across physical communication bus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sender_id: str
    recipient_id: str | None = None  # None indicates broadcast
    payload: dict[str, Any]
    sent_at_step: int
    sender_position: Position


@dataclass(frozen=True, slots=True)
class JointTransition:
    """Joint transition tuple for Centralized Training with Decentralized Execution (CTDE)."""

    global_state: np.ndarray
    observations: Mapping[str, np.ndarray]
    joint_actions: Mapping[str, int]
    rewards: Mapping[str, float]
    team_reward: float
    next_global_state: np.ndarray
    next_observations: Mapping[str, np.ndarray]
    terminated: bool
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MultiAgentMetricSummary:
    """Statistical summary of multi-agent society dynamics and resource allocations."""

    mean_individual_return: float
    team_return: float
    group_survival_rate: float
    gini_coefficient: float
    cooperation_index: float
    commons_stock: float
    communication_volume: int


@runtime_checkable
class MARLAlgorithm(Protocol):
    """Protocol for Multi-Agent Reinforcement Learning controllers."""

    def select_actions(
        self,
        observations: Mapping[str, np.ndarray],
        explore: bool = True,
    ) -> dict[str, int]:
        """Selects decentralized actions for each agent in observation map."""
        ...

    def update(self, transition: JointTransition) -> dict[str, float]:
        """Updates internal models given a joint transition."""
        ...
