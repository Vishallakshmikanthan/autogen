"""Core foundational types and data protocols for AUTOGEN AI."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionTrace(BaseModel):
    """Immutable record of an agent decision step.

    Required by AGENTS.md: Every important decision must produce a DecisionTrace
    generated from actual execution, enabling explainability and replay in the
    Algorithm Lab and Research Studio.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(description="Unique identifier of the deciding agent")
    step_id: int = Field(ge=0, description="Simulation step tick")
    timestamp_ns: int = Field(
        default_factory=time.time_ns,
        description="POSIX timestamp in nanoseconds when decision was reached",
    )
    algorithm_name: str = Field(description="Canonical algorithm identifier")
    decision_type: str = Field(
        description="Categorical decision class (e.g., action_selection, heuristic_evaluation)"
    )
    metrics: dict[str, float | int | str] = Field(
        default_factory=dict,
        description="Instrumented search/evaluation metrics (nodes expanded, runtime, depth)",
    )
    rationale: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal state trace or explanation supporting the decision",
    )
