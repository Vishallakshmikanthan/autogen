"""Integration tests for M18 Experiment Engine, YAML validation, and reproducibility (SYS-16)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from autogen.analytics.report import MarkdownReportGenerator
from autogen.experiments.config import (
    AgentSpec,
    EnvironmentSpec,
    EvaluationProtocol,
    ExperimentConfig,
)
from autogen.experiments.record import ExperimentSuiteRecord
from autogen.experiments.runner import ExperimentRunner


def test_yaml_config_validation_and_schema_enforcement(tmp_path: Path):
    yaml_valid = """
    name: reflex_vs_random_benchmark
    description: Empirical comparison of reflex against random exploration
    seed: 42
    episodes: 5
    parallel_workers: 2
    environment:
      width: 12
      height: 12
      obstacle_density: 0.10
      resource_density: 0.15
      max_steps: 25
    agents:
      - agent_id: reflex_1
        agent_type: simple_reflex
      - agent_id: random_1
        agent_type: random
    protocol:
      eval_seed_range: [100, 104]
      confidence_level: 0.95
    """
    cfg_file = tmp_path / "exp_valid.yaml"
    cfg_file.write_text(yaml_valid, encoding="utf-8")

    cfg = ExperimentConfig.from_yaml(cfg_file)
    assert cfg.name == "reflex_vs_random_benchmark"
    assert cfg.episodes == 5
    assert len(cfg.agents) == 2

    # Schema must forbid unexpected fields
    yaml_invalid = yaml_valid + "\n    unauthorized_key: dangerous_value"
    with pytest.raises(ValidationError):
        ExperimentConfig.from_yaml(yaml_invalid)


def test_experiment_suite_execution_and_reproducibility():
    config = ExperimentConfig(
        name="reproducibility_test",
        seed=123,
        episodes=4,
        parallel_workers=2,
        environment=EnvironmentSpec(
            width=10,
            height=10,
            max_steps=20,
        ),
        agents=[
            AgentSpec(agent_id="test_reflex", agent_type="simple_reflex"),
        ],
        protocol=EvaluationProtocol(
            eval_seed_range=(500, 503),
            confidence_level=0.95,
        ),
    )

    runner = ExperimentRunner(use_parallel=False)
    suite_record_1 = runner.run(config)
    suite_record_2 = runner.run(config)

    # Identical config hash
    assert suite_record_1.config_hash == suite_record_2.config_hash
    assert len(suite_record_1.records) == 4

    # Bit-identical metrics for every seed
    for r1, r2 in zip(suite_record_1.records, suite_record_2.records, strict=True):
        assert r1.seed == r2.seed
        assert r1.metrics["survival_rate"] == r2.metrics["survival_rate"]
        assert r1.metrics["average_return"] == r2.metrics["average_return"]
        assert r1.metrics["total_ticks"] == r2.metrics["total_ticks"]


def test_suite_record_persistence_and_report_generation(tmp_path: Path):
    config = ExperimentConfig(
        name="persistence_report_test",
        seed=999,
        episodes=3,
        parallel_workers=1,
        environment=EnvironmentSpec(
            width=10,
            height=10,
            max_steps=15,
        ),
        agents=[
            AgentSpec(agent_id="ag_1", agent_type="simple_reflex"),
        ],
        protocol=EvaluationProtocol(
            eval_seed_range=(300, 302),
        ),
    )

    runner = ExperimentRunner(use_parallel=False)
    record = runner.run(config)

    # Save and reload JSON record
    out_file = tmp_path / "suite_record.json"
    record.save(out_file)
    assert out_file.exists()

    loaded = ExperimentSuiteRecord.load(out_file)
    assert loaded.suite_id == record.suite_id
    assert loaded.config_hash == record.config_hash
    assert len(loaded.records) == 3

    # Report generation
    md_report = MarkdownReportGenerator.generate_report(
        experiment_name=loaded.experiment_name,
        config_hash=loaded.config_hash,
        git_commit=loaded.git_commit,
        total_episodes=loaded.total_runs,
        metrics=loaded.aggregated_metrics,
        runtime_seconds=1.25,
        throughput_ticks_per_sec=150.0,
    )

    assert "# Experimental Report:" in md_report
    assert "Mandatory Limitations" in md_report
    assert "Low sample size warning" in md_report
    assert "survival_rate" in md_report
