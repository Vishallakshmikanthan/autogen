"""Showpiece scenario contracts and scientific summary structures (SYS-19, Gate 10, Module M22).

Standardizes configuration, execution traces, and empirical verdicts across the
seven canonical demonstration scenarios of AUTOGEN AI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ShowpieceID(str, Enum):
    """Canonical identifiers for the seven Gate 10 showpiece research scenarios."""

    GHOST_IN_THE_FOG = "1_ghost_in_the_fog"
    ALGORITHM_DUEL = "2_algorithm_duel"
    TRAGEDY_OF_THE_COMMONS = "3_tragedy_of_the_commons"
    EVOLUTION_OF_TRIBES = "4_evolution_of_tribes"
    ABLATION_ARENA = "5_ablation_arena"
    TIME_MACHINE = "6_time_machine"
    THE_CHANGING_WORLD = "7_the_changing_world"


class ShowpieceRunResult(BaseModel):
    """Immutable result contract produced by a showpiece scenario execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    showpiece_id: ShowpieceID
    title: str
    seed: int
    execution_time_seconds: float
    git_commit: str
    metrics: dict[str, float]
    traces: list[dict[str, Any]] = Field(default_factory=list)
    verdict: str
    scientific_summary: str
