"""Tests for systematic cognitive subsystem ablations and Cohen's d effect sizes (SYS-15, Module M17)."""

from __future__ import annotations

from autogen.agents.hybrid.config import HybridAgentConfig
from autogen.agents.hybrid.hybrid_agent import HybridCognitiveAgent
from autogen.research.ablation_arena import AblationArena, compute_cohens_d


def test_cohens_d_effect_size_calculation() -> None:
    """Verifies Cohen's d computation on known synthetic samples."""
    # Group 1 with mean 10, std ~ 1; Group 2 with mean 8, std ~ 1 -> d ~ 2.0 (large effect)
    g1 = [10.0, 10.5, 9.5, 10.2, 9.8]
    g2 = [8.0, 8.5, 7.5, 8.2, 7.8]
    d = compute_cohens_d(g1, g2)
    assert d > 1.5

    # Identical groups -> d == 0.0
    assert compute_cohens_d(g1, g1) == 0.0


def test_ablation_arena_study_execution() -> None:
    """Verifies that AblationArena executes Full and ablated variants and computes effect sizes."""
    arena = AblationArena(
        eval_seeds=[101, 202, 303, 404, 505],
        max_steps_per_episode=15,
        grid_size=12,
    )

    results = arena.run_ablation_study()

    assert "Full" in results
    assert "No_Shield" in results
    assert "No_Bayes" in results
    assert "No_Options" in results

    full_res = results["Full"]
    no_shield_res = results["No_Shield"]

    # In hazard environments, the Full architecture with Safety Shield should equal or outperform No_Shield
    assert full_res.survival_rate >= no_shield_res.survival_rate
    assert full_res.mean_shield_vetoes > 0.0


def test_ablation_toggles_integrity() -> None:
    """Verifies that configuration flags correctly toggle subsystems on the HybridCognitiveAgent."""
    cfg_no_bayes = HybridAgentConfig(enable_bayes=False)
    agent = HybridCognitiveAgent(agent_id="test_hybrid", config=cfg_no_bayes)

    assert not agent.config.enable_bayes
    assert agent.config.enable_safety_shield
    assert agent.config.enable_options
