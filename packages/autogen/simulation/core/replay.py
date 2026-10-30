"""Replay logging and deterministic reproduction of simulation runs."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.state.types import ActionType


class ReplayStep(BaseModel):
    """Actions executed by agents at a single simulation tick."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    actions: dict[str, ActionType]
    state_hash: str


class ReplayLog(BaseModel):
    """Complete log required to bit-identically reproduce a simulation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_seed: int
    config: EnvironmentConfig
    steps: tuple[ReplayStep, ...] = Field(default_factory=tuple)

    def save(self, filepath: Path | str) -> None:
        """Serializes replay log to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Path | str) -> ReplayLog:
        """Loads replay log from a JSON file."""
        path = Path(filepath)
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw_data)
