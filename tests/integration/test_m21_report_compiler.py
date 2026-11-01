"""Integration tests for Research Report Compiler and Cryptographic Traceability Verifier (SYS-19, Module M21)."""

import pytest

from autogen.analytics.statistics import confidence_interval, mean
from autogen.experiments.config import AgentSpec, EnvironmentSpec, ExperimentConfig
from autogen.experiments.record import ExperimentSuiteRecord
from autogen.experiments.runner import ExperimentRunner
from autogen.research.finding_schema import (
    ResearchHypothesis,
    ResearchReportDocument,
    TraceableMetricClaim,
)
from autogen.research.report_compiler import ReportCompiler, TraceabilityVerifier


@pytest.fixture(scope="module")
def sample_suite_record() -> ExperimentSuiteRecord:
    """Runs a quick 6-seed simulation suite to generate verifiable run records."""
    config = ExperimentConfig(
        name="test_m21_suite",
        description="M21 integration test suite",
        seed=2001,
        episodes=6,
        parallel_workers=1,
        environment=EnvironmentSpec(
            width=10,
            height=10,
            obstacle_density=0.1,
            resource_density=0.1,
            max_steps=20,
        ),
        agents=[
            AgentSpec(
                agent_id="test_agent",
                agent_type="simple_reflex",
            )
        ],
    )
    runner = ExperimentRunner(use_parallel=False)
    return runner.run(config)


def test_report_compilation_and_traceability(sample_suite_record: ExperimentSuiteRecord):
    """Verify that generated claims match raw run records and verification passes."""
    returns = [r.metrics["average_return"] for r in sample_suite_record.records]
    mean_val = float(mean(returns))
    ci_lower, ci_upper = confidence_interval(returns, confidence=0.95)

    claim = TraceableMetricClaim(
        claim_id="CLM-TEST-RETURN",
        metric_name="average_return",
        reported_mean=mean_val,
        reported_ci_lower=ci_lower,
        reported_ci_upper=ci_upper,
        sample_size=len(sample_suite_record.records),
        suite_id=sample_suite_record.suite_id,
        config_sha256=sample_suite_record.config_hash,
        run_record_ids=[r.run_id for r in sample_suite_record.records],
    )

    doc = ResearchReportDocument(
        report_id="REP-TEST-001",
        title="Test Scientific Report",
        date="2026-11-01",
        git_commit=sample_suite_record.git_commit,
        hypothesis=ResearchHypothesis(
            id="HYP-001",
            description="Test hypothesis",
            null_hypothesis="H0: mu = 0",
            alternative_hypothesis="H1: mu > 0",
            significance_level=0.05,
        ),
        environments=["GridWorld-10x10"],
        claims=[claim],
    )

    # 1. Compile report
    markdown = ReportCompiler.compile_markdown_report(doc, [sample_suite_record])
    assert "## 3. Quantitative Results & Verified Claims" in markdown
    assert "CLM-TEST-RETURN" in markdown

    # 2. Verify traceability
    errors = TraceabilityVerifier.verify(
        markdown, {sample_suite_record.suite_id: sample_suite_record}
    )
    assert len(errors) == 0


def test_tamper_detection(sample_suite_record: ExperimentSuiteRecord):
    """Verify that tampering with any reported number causes verification to fail."""
    returns = [r.metrics["average_return"] for r in sample_suite_record.records]
    mean_val = float(mean(returns))
    ci_lower, ci_upper = confidence_interval(returns, confidence=0.95)

    # Fabricate a modified/falsified claim
    tampered_claim = TraceableMetricClaim(
        claim_id="CLM-TAMPERED",
        metric_name="average_return",
        reported_mean=mean_val + 50.0,  # Fabricated inflated return!
        reported_ci_lower=ci_lower,
        reported_ci_upper=ci_upper,
        sample_size=len(sample_suite_record.records),
        suite_id=sample_suite_record.suite_id,
        config_sha256=sample_suite_record.config_hash,
        run_record_ids=[r.run_id for r in sample_suite_record.records],
    )

    doc = ResearchReportDocument(
        report_id="REP-TAMPER-001",
        title="Tampered Report",
        date="2026-11-01",
        git_commit=sample_suite_record.git_commit,
        hypothesis=ResearchHypothesis(
            id="HYP-001",
            description="Test hypothesis",
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            significance_level=0.05,
        ),
        environments=["GridWorld-10x10"],
        claims=[tampered_claim],
    )

    markdown = ReportCompiler.compile_markdown_report(doc, [sample_suite_record])

    errors = TraceabilityVerifier.verify(
        markdown, {sample_suite_record.suite_id: sample_suite_record}
    )
    assert len(errors) > 0
    assert any("DISCREPANCY: Reported mean" in e for e in errors)
