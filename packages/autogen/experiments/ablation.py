"""Systematic component ablation and factor-matrix runner (SYS-16, SYS-17)."""

from __future__ import annotations

from typing import Any

from autogen.analytics.hypothesis import compare_samples
from autogen.experiments.config import ExperimentConfig
from autogen.experiments.record import ExperimentSuiteRecord
from autogen.experiments.runner import ExperimentRunner


class AblationRunner:
    """Orchestrates factorial or component-removal ablation studies against a baseline."""

    def __init__(self, runner: ExperimentRunner | None = None) -> None:
        self.runner = runner or ExperimentRunner(use_parallel=False)

    def run_ablation(
        self,
        base_config: ExperimentConfig,
        ablation_matrix: dict[str, list[Any]],
    ) -> ExperimentSuiteRecord:
        """Executes the full baseline and each ablated variation across identical seeds."""
        # 1. Run baseline
        baseline_record = self.runner.run(base_config)
        baseline_returns = [r.metrics["average_return"] for r in baseline_record.records]
        baseline_stats = baseline_record.aggregated_metrics["average_return"]

        ablation_summary: list[dict[str, Any]] = [{
            "name": "baseline_full",
            "mean_return": baseline_stats.mean,
            "ci_str": f"[{baseline_stats.ci_lower:.2f}, {baseline_stats.ci_upper:.2f}]",
            "survival_rate": baseline_record.aggregated_metrics["survival_rate"].mean,
            "cohens_d": 0.0,
            "status": "Baseline",
        }]

        all_comparisons = list(baseline_record.comparisons)
        all_run_records = list(baseline_record.records)

        # 2. Iterate through ablation matrix factors
        for factor_name, variations in ablation_matrix.items():
            for var_val in variations:
                variant_name = f"ablation_{factor_name}_{var_val}"
                cfg_dict = base_config.model_dump()
                cfg_dict["name"] = variant_name

                # Modify agent policy params or agent specs
                for ag_dict in cfg_dict["agents"]:
                    ag_dict["policy_params"][factor_name] = var_val

                variant_config = ExperimentConfig.model_validate(cfg_dict)
                variant_record = self.runner.run(variant_config)

                variant_returns = [r.metrics["average_return"] for r in variant_record.records]
                variant_stats = variant_record.aggregated_metrics["average_return"]

                # Statistical comparison against baseline
                test_result = compare_samples(
                    sample_baseline=baseline_returns,
                    sample_candidate=variant_returns,
                    paired=True,
                    alpha=base_config.protocol.significance_alpha,
                    seed=base_config.seed,
                )
                all_comparisons.append(test_result)
                all_run_records.extend(variant_record.records)

                status = (
                    "Significant Drop"
                    if test_result.is_significant and variant_stats.mean < baseline_stats.mean
                    else ("Significant Gain" if test_result.is_significant else "No Difference")
                )

                ablation_summary.append({
                    "name": variant_name,
                    "mean_return": variant_stats.mean,
                    "ci_str": f"[{variant_stats.ci_lower:.2f}, {variant_stats.ci_upper:.2f}]",
                    "survival_rate": variant_record.aggregated_metrics["survival_rate"].mean,
                    "cohens_d": test_result.effect_size,
                    "status": status,
                })

        return ExperimentSuiteRecord(
            suite_id=f"ablation_{base_config.name}_{base_config.seed}",
            experiment_name=f"Ablation: {base_config.name}",
            config_hash=baseline_record.config_hash,
            total_runs=len(all_run_records),
            git_commit=baseline_record.git_commit,
            software_versions=baseline_record.software_versions,
            aggregated_metrics=baseline_record.aggregated_metrics,
            comparisons=all_comparisons,
            ablation_summary=ablation_summary,
            records=all_run_records,
            limitations=baseline_record.limitations,
            created_at=baseline_record.created_at,
        )
