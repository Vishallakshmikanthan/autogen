"""Immutable reproducibility records for experimental simulation runs (SYS-16)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import scipy
from pydantic import BaseModel, ConfigDict, Field

from autogen.analytics.hypothesis import HypothesisTestResult
from autogen.analytics.statistics import SampleStats
from autogen.experiments.config import ExperimentConfig


def get_git_commit() -> str:
    """Retrieves current git commit hash, or 'unknown-commit' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown-commit"


def compute_config_hash(config: ExperimentConfig) -> str:
    """Computes SHA-256 digest of serialized canonical config JSON."""
    raw = config.model_dump_json()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RunRecord(BaseModel):
    """Reproducibility and performance record for a single simulation seed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    config_hash: str
    seed: int
    git_commit: str
    software_versions: dict[str, str]
    metrics: dict[str, float]
    timing_seconds: float
    ticks_per_second: float
    timestamp: str


class ExperimentSuiteRecord(BaseModel):
    """Aggregated experimental record containing all run records and statistical analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_id: str
    experiment_name: str
    config_hash: str
    total_runs: int
    git_commit: str
    software_versions: dict[str, str]
    aggregated_metrics: dict[str, SampleStats]
    comparisons: list[HypothesisTestResult] = Field(default_factory=list)
    ablation_summary: list[dict[str, Any]] = Field(default_factory=list)
    records: list[RunRecord] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    created_at: str

    def save(self, output_path: str | Path) -> None:
        """Saves record as human-readable JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, input_path: str | Path) -> ExperimentSuiteRecord:
        """Loads and validates record from JSON file."""
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


def capture_environment_versions() -> dict[str, str]:
    """Captures key runtime package versions for provenance tracking."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pydantic": pydantic.__version__,
        "os": platform.platform(),
    }
