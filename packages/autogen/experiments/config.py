"""Declarative experiment configurations validated with Pydantic (SYS-16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class EnvironmentSpec(BaseModel):
    """Specification of grid world environment for experiment runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(ge=5, le=500, default=20)
    height: int = Field(ge=5, le=500, default=20)
    obstacle_density: float = Field(ge=0.0, le=0.8, default=0.15)
    resource_density: float = Field(ge=0.0, le=0.8, default=0.08)
    hazard_density: float = Field(ge=0.0, le=0.8, default=0.05)
    energy_max: float = Field(gt=0.0, default=100.0)
    max_steps: int = Field(ge=1, le=10000, default=200)


class AgentSpec(BaseModel):
    """Specification of an agent architecture to instantiate and benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    agent_type: str
    policy_params: dict[str, Any] = Field(default_factory=dict)


class EvaluationProtocol(BaseModel):
    """Protocol defining seed ranges, held-out environments, and confidence levels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_seed_range: tuple[int, int] = (1000, 1099)
    eval_seed_range: tuple[int, int] = (2000, 2049)
    confidence_level: float = Field(ge=0.80, le=0.999, default=0.95)
    significance_alpha: float = Field(ge=0.001, le=0.20, default=0.05)


class ExperimentConfig(BaseModel):
    """Root configuration for a reproducible experiment suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    seed: int = 42
    episodes: int = Field(ge=1, default=30)
    parallel_workers: int = Field(ge=1, default=4)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    agents: list[AgentSpec]
    metrics: list[str] = Field(
        default_factory=lambda: [
            "survival_rate",
            "average_return",
            "path_efficiency",
            "resource_efficiency",
            "throughput_ticks_per_sec",
        ]
    )
    protocol: EvaluationProtocol = Field(default_factory=EvaluationProtocol)
    ablation_matrix: dict[str, list[Any]] | None = None

    @classmethod
    def from_yaml(cls, source: str | Path) -> ExperimentConfig:
        """Loads and validates an ExperimentConfig from YAML string or file path."""
        if isinstance(source, Path) or (isinstance(source, str) and ("\n" not in source and Path(source).exists())):
            with open(source, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(source)

        if not isinstance(data, dict):
            raise ValueError("YAML document must contain a top-level dictionary.")

        # Support optional 'experiment' wrapper namespace
        if "experiment" in data and isinstance(data["experiment"], dict):
            payload = dict(data["experiment"])
            for key in ["environment", "agents", "metrics", "protocol", "ablation_matrix"]:
                if key in data and key not in payload:
                    payload[key] = data[key]
            data = payload

        return cls.model_validate(data)

