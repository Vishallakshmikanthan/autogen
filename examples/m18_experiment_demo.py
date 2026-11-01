"""CLI runnable example demonstrating Module M18 Experiment & Analytics Engine (SYS-16, SYS-17)."""

from pathlib import Path

from autogen.analytics.report import MarkdownReportGenerator
from autogen.experiments.ablation import AblationRunner
from autogen.experiments.config import (
    AgentSpec,
    EnvironmentSpec,
    EvaluationProtocol,
    ExperimentConfig,
)
from autogen.experiments.runner import ExperimentRunner


def main() -> None:
    print("=" * 70)
    print("AUTOGEN AI — M18 Experiment & Analytics Engine Demo (SYS-16, SYS-17)")
    print("=" * 70)

    # 1. Define declarative experiment
    config = ExperimentConfig(
        name="demo_reflex_vs_random",
        description="Comparing SimpleReflexAgent against RandomAgent across seeds",
        seed=42,
        episodes=6,
        parallel_workers=2,
        environment=EnvironmentSpec(
            width=12,
            height=12,
            obstacle_density=0.10,
            resource_density=0.12,
            max_steps=20,
        ),
        agents=[
            AgentSpec(agent_id="reflex_agent", agent_type="simple_reflex"),
            AgentSpec(agent_id="random_agent", agent_type="random"),
        ],
        protocol=EvaluationProtocol(
            eval_seed_range=(1001, 1006),
            confidence_level=0.95,
        ),
    )

    print(f"\n[1] Launching Experiment Suite: '{config.name}'")
    print(f"    - Episodes: {config.episodes}")
    print(f"    - Grid: {config.environment.width}x{config.environment.height}")
    print(f"    - Seed Range: {config.protocol.eval_seed_range}")

    runner = ExperimentRunner(use_parallel=False)
    record = runner.run(config)

    print("\n[2] Execution Complete!")
    print(f"    - Total Runs: {record.total_runs}")
    print(f"    - Config Hash: {record.config_hash[:16]}...")
    print(f"    - Git Commit: {record.git_commit[:10]}")

    print("\n[3] Empirical Metrics Summary:")
    for metric_name, stats in record.aggregated_metrics.items():
        print(f"    - {metric_name:25s}: Mean = {stats.mean:8.3f} | 95% CI = [{stats.ci_lower:7.3f}, {stats.ci_upper:7.3f}]")

    # 2. Run an ablation study
    print("\n[4] Running Component Ablation Matrix...")
    ablation_matrix = {
        "sensory_range": [1, 2],
    }
    ablation_runner = AblationRunner(runner=runner)
    ablation_record = ablation_runner.run_ablation(config, ablation_matrix)

    print("\n[5] Ablation Summary Table:")
    for row in ablation_record.ablation_summary:
        print(f"    - {row['name']:28s}: Return = {row['mean_return']:6.2f} | Survival = {row['survival_rate']*100:5.1f}% | Cohen's d = {row['cohens_d']:5.2f} | Status: {row['status']}")

    # 3. Generate Markdown report
    report_md = MarkdownReportGenerator.generate_report(
        experiment_name=record.experiment_name,
        config_hash=record.config_hash,
        git_commit=record.git_commit,
        total_episodes=record.total_runs,
        metrics=record.aggregated_metrics,
        comparisons=ablation_record.comparisons,
        ablation_table=ablation_record.ablation_summary,
        runtime_seconds=1.5,
        throughput_ticks_per_sec=120.0,
    )

    report_path = Path("reports/m18_demo_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n[6] Publication-ready report written to: {report_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

