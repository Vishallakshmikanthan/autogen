"""Pydantic schemas for empirical research findings and traceable claims (SYS-19, Module M21).

Enforces that every number in a scientific report links to an immutable
experiment record hash and raw execution runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceableMetricClaim(BaseModel):
    """Binds an empirical claim in a research report to an immutable experiment record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(..., description="Unique claim identifier, e.g. 'CLM-HYBRID-SURVIVAL'")
    metric_name: str = Field(..., description="Target metric key, e.g. 'survival_rate'")
    reported_mean: float = Field(..., description="Reported empirical mean")
    reported_ci_lower: float = Field(..., description="Reported 95% CI lower bound")
    reported_ci_upper: float = Field(..., description="Reported 95% CI upper bound")
    sample_size: int = Field(..., description="Number of independent evaluation seeds")
    suite_id: str = Field(..., description="Associated experiment suite ID")
    config_sha256: str = Field(..., description="SHA-256 of the configuration used")
    run_record_ids: list[str] = Field(
        default_factory=list, description="List of individual run IDs"
    )


class ResearchHypothesis(BaseModel):
    """Formal hypothesis specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str
    null_hypothesis: str
    alternative_hypothesis: str
    significance_level: float = 0.05


class AblationFinding(BaseModel):
    """Component ablation comparison with effect size."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    factor_name: str
    baseline_mean: float
    ablated_mean: float
    cohens_d: float
    p_value: float
    verdict: str  # "Significant Drop", "Catastrophic Failure", or "No Difference"


class ResearchReportDocument(BaseModel):
    """Complete structured research finding document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str
    title: str
    authors: list[str] = Field(default_factory=lambda: ["AUTOGEN AI Empirical Research Engine"])
    date: str
    git_commit: str
    hypothesis: ResearchHypothesis
    environments: list[str]
    claims: list[TraceableMetricClaim]
    ablations: list[AblationFinding] = Field(default_factory=list)
    threats_to_validity: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
