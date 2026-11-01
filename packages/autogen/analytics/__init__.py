"""Analytics package: metrics computation and statistical validation (SYS-17)."""

from autogen.analytics.hypothesis import (
    HypothesisTestResult,
    compare_samples,
    diagnostic_normality_check,
    holm_bonferroni,
    paired_permutation_test,
    paired_t_test,
    two_sample_permutation_test,
    wilcoxon_signed_rank,
)
from autogen.analytics.metrics import (
    PathEfficiencyComputer,
    PopulationGrowthComputer,
    ResourceEfficiencyComputer,
    SimulationThroughputComputer,
    SurvivalRateComputer,
)
from autogen.analytics.report import MarkdownReportGenerator
from autogen.analytics.statistics import (
    SampleStats,
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    confidence_interval,
    interquartile_range,
    mean,
    median,
    sample_stats,
    sample_variance,
    standard_deviation,
    standard_error,
)

__all__ = [
    "HypothesisTestResult",
    "MarkdownReportGenerator",
    "PathEfficiencyComputer",
    "PopulationGrowthComputer",
    "ResourceEfficiencyComputer",
    "SampleStats",
    "SimulationThroughputComputer",
    "SurvivalRateComputer",
    "bootstrap_ci",
    "cliffs_delta",
    "cohens_d",
    "compare_samples",
    "confidence_interval",
    "diagnostic_normality_check",
    "holm_bonferroni",
    "interquartile_range",
    "mean",
    "median",
    "paired_permutation_test",
    "paired_t_test",
    "sample_stats",
    "sample_variance",
    "standard_deviation",
    "standard_error",
    "two_sample_permutation_test",
    "wilcoxon_signed_rank",
]
