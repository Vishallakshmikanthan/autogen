"""State Encoders for Reinforcement Learning (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.intelligence.rl.contracts import StateEncoder
from autogen.simulation.core.contracts import Observation


class TabularGridEncoder(StateEncoder):
    """Encodes Observation into discrete (x, y) coordinates or flat integer state index."""

    def __init__(self, width: int = 15, height: int = 15) -> None:
        self.width = width
        self.height = height

    def encode(self, observation: Any) -> tuple[int, int]:
        if isinstance(observation, Observation):
            return (observation.position.x, observation.position.y)
        elif isinstance(observation, tuple) and len(observation) == 2:
            return (int(observation[0]), int(observation[1]))
        elif hasattr(observation, "position"):
            return (observation.position.x, observation.position.y)
        raise ValueError(f"Cannot encode observation of type {type(observation)}")

    def encode_index(self, observation: Any) -> int:
        x, y = self.encode(observation)
        return y * self.width + x

    @property
    def num_states(self) -> int:
        return self.width * self.height


class FeatureVectorEncoder(StateEncoder):
    """Encodes Observation into normalized float64 vector [x_norm, y_norm, energy_norm, health_norm]."""

    def __init__(self, width: int = 15, height: int = 15) -> None:
        self.width = width
        self.height = height

    def encode(self, observation: Any) -> np.ndarray:
        if isinstance(observation, Observation):
            return np.array([
                float(observation.position.x) / max(1, self.width),
                float(observation.position.y) / max(1, self.height),
                float(observation.energy) / 100.0,
                float(observation.health) / 100.0,
            ], dtype=np.float64)
        elif isinstance(observation, tuple) and len(observation) == 2:
            return np.array([
                float(observation[0]) / max(1, self.width),
                float(observation[1]) / max(1, self.height),
                1.0,
                1.0,
            ], dtype=np.float64)
        raise ValueError(f"Cannot encode observation of type {type(observation)}")

    @property
    def num_states(self) -> int | None:
        return None

