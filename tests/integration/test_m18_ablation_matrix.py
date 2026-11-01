"""Integration tests for M18 Ablation Matrix expansion and factor analysis (SYS-16, SYS-17)."""

from autogen.experiments.ablation import AblationRunner
from autogen.experiments.config import (
    AgentSpec,
    EnvironmentSpec,
    EvaluationProtocol,
    ExperimentConfig,
)
from autogen.experiments.runner import ExperimentRunner


def test_ablation_matrix_execution():
    base_config = ExperimentConfig(
        name="hybrid_ablation_test",
        seed=777,
        episodes=3,
        parallel_workers=1,
        environment=EnvironmentSpec(
            width=10,
            height=10,
            max_steps=15,
        ),
        agents=[
            AgentSpec(
                agent_id="agent_under_test",
                agent_type="simple_reflex",
                policy_params={"use_memory": True, "use_planning": True},
            ),
        ],
        protocol=EvaluationProtocol(
            eval_seed_range=(701, 703),
            confidence_level=0.95,
        ),
    )

    ablation_matrix = {
        "use_memory": [False],
        "use_planning": [False],
    }

    ablation_runner = AblationRunner(runner=ExperimentRunner(use_parallel=False))
    result_record = ablation_runner.run_ablation(base_config, ablation_matrix)

    assert result_record.experiment_name.startswith("Ablation:")
    # Baseline + 2 ablated variants = 3 summary entries
    assert len(result_record.ablation_summary) == 3
    assert result_record.ablation_summary[0]["name"] == "baseline_full"
    assert result_record.ablation_summary[1]["name"] == "ablation_use_memory_False"
    assert result_record.ablation_summary[2]["name"] == "ablation_use_planning_False"

    # Comparisons were generated against baseline
    assert len(result_record.comparisons) == 2

