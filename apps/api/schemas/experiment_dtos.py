"""Experiment orchestration request and response schemas (M19, SYS-18)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.experiments.config import ExperimentConfig


class CreateExperimentRequest(BaseModel):
    """Payload to register a new experiment specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    config: ExperimentConfig


class ExperimentResponse(BaseModel):
    """Metadata response for a registered experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    config_hash: str
    config: dict[str, Any]


class ExperimentRunResultResponse(BaseModel):
    """Aggregated results from an executed experiment suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exp_id: str
    total_runs: int
    runs: list[dict[str, Any]]

